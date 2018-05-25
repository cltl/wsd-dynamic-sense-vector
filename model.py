import numpy as np
import tensorflow as tf
import time
import sys
import pickle
from tqdm._tqdm import tqdm
import random

float_dtype = tf.float32

class DummyModelTrain(object):
    '''
    This is for testing GPU usage only. This model runs very trivial operations
    on GPU therefore its running time is mostly on CPU. Compared to WSDModel,
    this model should run much faster, otherwise you're spending too much time
    on CPU.
    '''

    def __init__(self, config):
        self._x = tf.placeholder(tf.int32, shape=[None, None], name='x')
        self._y = tf.placeholder(tf.int32, shape=[None], name='y')
        self._subvocab = tf.placeholder(tf.int32, shape=[None], name='subvocab')
        
        self._cost = tf.reduce_mean(tf.reduce_sum(self._x, axis=1) - self._y) + tf.reduce_mean(self._subvocab)
        self._train_op = tf.reduce_mean(tf.reduce_sum(self._x, axis=1) - self._y) + tf.reduce_mean(self._subvocab)
        self._initial_state = tf.reduce_mean(self._x)
        
    def trace_timeline(self):
        pass

    def train_epoch(self, session, data, verbose=False):
        sentence_lens = np.array([x.shape[1] for x, _, _ in data])
        samples = np.random.choice(len(data), size=len(data), 
                                   p=sentence_lens/sentence_lens.sum())
        for batch_id in samples:
            x, subvocab, target_id = data[batch_id]
            i =  np.random.randint(x.shape[1])
            y = x[:,i].copy() # copy content
            x[:,i] = target_id
            feed_dict = {self._x: x, self._y: y, self._subvocab: subvocab}
            session.run(self._initial_state, feed_dict)
            session.run([self._cost, self._train_op], feed_dict)
            x[:,i] = y # restore the data
        return 0.1234
    
    def print_device_placement(self):
        pass


class WSDModel(object):
    """A LSTM WSD model designed for fast training."""

    def __init__(self, config, optimized=False, reuse_variables=False, use_eos=False):
        self.config = config
        self.optimized = optimized
        self.reuse_variables = reuse_variables
        self.use_eos = use_eos
        self._build_inputs()
        self._build_word_embeddings()
        self._build_lstm_output()
        self._build_context_embs()
        self._build_logits()
        self._build_output()
        self._build_cost()
        self.run_options = self.run_metadata = None

    def _build_inputs(self):
        # the names are for later reference when the model is loaded
        self._x = tf.placeholder(tf.int32, shape=[None, None], name='x')
        self._y = tf.placeholder(tf.int32, shape=[None], name='y')
        # they might be used or not, doesn't hurt
        self._subvocab = tf.placeholder(tf.int32, shape=[None], name='subvocab')
        self._lens = tf.placeholder(tf.int32, shape=[None], name='lens')

    def _build_word_embeddings(self):
        E_words = tf.get_variable("word_embedding", 
                [self.config.vocab_size, self.config.emb_dims], dtype=float_dtype)
        self._word_embs = tf.nn.embedding_lookup(E_words, self._x)

    def _build_lstm_output(self):
        cell = tf.contrib.rnn.LSTMCell(num_units=self.config.hidden_size,
                                       state_is_tuple=True, reuse=self.reuse_variables)
        if self.optimized and self.config.assume_same_lengths:
            outputs, _ = tf.nn.dynamic_rnn(cell, self._word_embs, 
                                           dtype=float_dtype)
            self._lstm_output = outputs[:,-1]
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, self._word_embs, 
                                           sequence_length=self._lens,
                                           dtype=float_dtype)
            last_output_indices = tf.stack([tf.range(tf.shape(self._x)[0]), self._lens-1], axis=1)
            self._lstm_output = tf.gather_nd(outputs, last_output_indices)
        self._initial_state = cell.zero_state(tf.shape(self._x)[0], float_dtype)

    def _build_context_embs(self):
        context_layer_weights = tf.get_variable("context_layer_weights",
                [self.config.hidden_size, self.config.emb_dims], dtype=float_dtype)
        self._predicted_context_embs = tf.matmul(self._lstm_output, context_layer_weights, 
                                                 name='predicted_context_embs')
    
    def _build_logits(self):
        E_contexts = tf.get_variable("context_embedding", 
                [self.config.vocab_size, self.config.emb_dims], dtype=float_dtype)
        if self.optimized and self.config.sampled_softmax:
            subcontexts = tf.nn.embedding_lookup(E_contexts, self._subvocab)
            self._logits = tf.matmul(self._predicted_context_embs, tf.transpose(subcontexts))
        else:
            self._logits = tf.matmul(self._predicted_context_embs, tf.transpose(E_contexts))
    
    def _build_cost(self):
        self._cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._logits, labels=self._y))
        self._accuracy = tf.reduce_mean(tf.cast(
                tf.equal(self._y, self._y_hat), tf.float32))
        self._hit_at_100 = tf.reduce_mean(tf.cast(
                tf.nn.in_top_k(self._logits, self._y, 100), float_dtype))
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          self.config.max_grad_norm)
        optimizer = tf.train.AdagradOptimizer(self.config.learning_rate)
        self._global_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                global_step=self._global_step)
    
    def _build_output(self):
        self._y_hat = tf.argmax(self._logits, axis=1, output_type=tf.int32)
        self._probs = tf.nn.softmax(self._logits, axis=1)
    
    def trace_timeline(self):
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

    def train_epoch(self, session, data, target_id, verbose=False):
        """ Run the model through the data once, update parameters """
        total_cost = 0.0
        total_rows = 0
        
        # resample the batches so that each token has equal chance to become target
        # another effect is to randomize the order of batches
        if self.config.optimized_batches:
            sentence_lens = np.array([x.shape[1] for x, _, _, _ in data])
            samples = np.random.choice(len(data), size=len(data), 
                                       p=sentence_lens/sentence_lens.sum())
        else:
            samples = np.random.choice(len(data), size=len(data))
        for batch_no, batch_id in enumerate(samples):
            x, y_all, subvocab, lens = data[batch_id]
            batch_size = x.shape[0]
            # lens is subtracted by 1, if needed, to avoid selecting <eos> as target
            i = np.mod(np.random.randint(1000000, size=batch_size), 
                       lens-(1 if self.use_eos else 0))
            one_to_n = np.arange(batch_size)
            y = y_all[one_to_n,i]
            old_xi = x[one_to_n,i].copy() # old_xi might be different from y because of subvocab
            x[one_to_n,i] = target_id
    
            # self._lens may be used or not depends on assume_same_lengths option
            feed_dict = {self._x: x, self._y: y, self._subvocab: subvocab, self._lens: lens}
            state = session.run(self._initial_state, feed_dict)
            c, h = self._initial_state
            feed_dict[c] = state.c
            feed_dict[h] = state.h
    
            batch_cost, _ = session.run([self._cost, self._train_op], feed_dict,
                                        options=self.run_options, 
                                        run_metadata=self.run_metadata)
            x[one_to_n,i] = old_xi # restore the data
    
            total_cost += batch_cost * batch_size # because the cost is averaged
            total_rows += batch_size              # over rows in a batch
            
            if verbose and (batch_no+1) % 1000 == 0:
                print("\tfinished %d of %d batches, sample batch cost: %.7f" 
                      %(batch_no+1, len(samples), batch_cost))
        return total_cost / total_rows
    
    def print_device_placement(self):
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            print("******** Start of device placement ********")
            sess.run(tf.global_variables_initializer())
            x = np.random.randint(10, size=(100, 10))
            y = np.random.randint(10, size=100)
            subvocab = np.random.randint(100, size=10) 
            feed_dict = {self._x: x, self._y: y, self._subvocab : subvocab}
            state = sess.run(self._initial_state, feed_dict)
            c, h = self._initial_state
            feed_dict[c], feed_dict[h] = state.c, state.h
            sess.run(self._train_op, feed_dict)
            print("******** End of device placement ********")

    def measure_dev_cost(self, session, data, target_id):
        # make sure that we measure against the same dataset every time we call this method
        rng = np.random.RandomState(925)
        total_examples = 0
        total_cost = 0.0
        total_hit = 0.0
        for batch_x, _, _, batch_lens in data:
            batch_size = len(batch_lens)
            # lens is subtracted by 1, if needed, to avoid selecting <eos> as target
            target_indices = np.mod(rng.randint(1000000, size=batch_size), 
                                    batch_lens-(1 if self.use_eos else 0))
            one_to_n = np.arange(batch_size)
            batch_y = batch_x[one_to_n, target_indices]
            batch_x[one_to_n, target_indices] = target_id
            # initialize the RNN
            feed_dict = { self._x: batch_x, self._y: batch_y, self._lens: batch_lens}
            state = session.run(self._initial_state, feed_dict)
            c, h = self._initial_state
            feed_dict[c] = state.c
            feed_dict[h] = state.h
            # now it's time to evaluate
            cost, hit_at_100 = session.run([self._cost, self._hit_at_100], feed_dict)        
            total_cost += cost * batch_size
            total_hit += hit_at_100 * batch_size
            total_examples += batch_size
            # restore the data
            batch_x[one_to_n, target_indices] = batch_y
        return total_cost / total_examples, total_hit / total_examples


class WSIModel(WSDModel):
    """A LSTM word sense induction (WSI) model designed for fast training."""

    def _build_logits(self):
        E_contexts = tf.get_variable("context_embedding", 
                [self.config.vocab_size, self.config.num_senses, self.config.emb_dims], 
                dtype=float_dtype)
        if self.optimized and self.config.sampled_softmax:
            subcontexts = tf.nn.embedding_lookup(E_contexts, self._subvocab)
            subvocab_size = tf.shape(self._subvocab)[0]
            sense_logits = tf.matmul(self._predicted_context_embs, tf.transpose(
                    tf.reshape(subcontexts, (-1, self.config.emb_dims))))
            self._logits = tf.reduce_max(tf.reshape(sense_logits, 
                    (-1, subvocab_size, self.config.num_senses)), axis=2)
        else:
            sense_logits = tf.matmul(self._predicted_context_embs, tf.transpose(
                tf.reshape(E_contexts, (-1, self.config.emb_dims))))
            self._logits = tf.reduce_max(tf.reshape(sense_logits,
                    (-1, self.config.vocab_size, self.config.num_senses)), axis=2)
        

class HDNModel(WSDModel):

    def __init__(self, config, reuse_variables=False):
        self._build_masks(config)
        WSIModel.__init__(self, config, optimized=False, 
                          reuse_variables=reuse_variables, use_eos=True)

    def _build_masks(self, config):
        with open(config.hdn_vocab_path, 'rb') as f:
            self.hdn2id = pickle.load(f) 
        with open(config.hdn_list_vocab_path, 'rb') as f:
            self.hdn_list2id = pickle.load(f) 
        masks = np.zeros((len(self.hdn_list2id)+1, len(self.hdn2id)), dtype=np.float32)
        for hdn_list, id1 in self.hdn_list2id.items():
            for hdn in hdn_list:
                masks[id1, self.hdn2id[hdn]] = 1
        self._unmask_index = len(self.hdn_list2id)
        masks[self._unmask_index,:] = 1
        self._masks = tf.constant(masks, dtype=tf.bool)

    def _build_inputs(self):
        WSIModel._build_inputs(self)
        self._candidates_list = tf.placeholder(tf.int32, shape=[None], name='candidate_list')

    def _build_logits(self):
        E_contexts = tf.get_variable("context_embedding", 
                [len(self.hdn2id), self.config.num_senses, self.config.emb_dims], 
                dtype=float_dtype)
        sense_logits = tf.matmul(self._predicted_context_embs, tf.transpose(
                tf.reshape(E_contexts, (-1, self.config.emb_dims))))
        self._unmasked_logits = tf.reduce_max(tf.reshape(sense_logits,
                (-1, len(self.hdn2id), self.config.num_senses)), axis=2,
                name='unmasked_logits')
        my_masks = tf.nn.embedding_lookup(self._masks, self._candidates_list)
        minus_inf = tf.ones_like(self._unmasked_logits)*(-np.inf)
        self._logits = tf.where(my_masks, self._unmasked_logits, minus_inf)

    @classmethod
    def gen_batches(cls, data, batch_size, word2id, name="noname"):
        buffer, indices = data
        batch_boundaries = [0]
        max_len = -1
        for i, l in enumerate(indices['sent_len']):
            new_max_len = max(l, max_len)
            if (i-batch_boundaries[-1]+1)*(new_max_len+1) > batch_size:
                batch_boundaries.append(i)
                max_len = l
            else:
                max_len = new_max_len
        batch_boundaries.append(len(indices))
        
        batches = []
        starts = tqdm(batch_boundaries[:-1], unit='batch',
                      desc='Preparing "%s" batches' %name)
        stops = batch_boundaries[1:]
        for start, stop in zip(starts, stops):
            my_indices = indices.iloc[start:stop]
            x = np.empty((len(my_indices), my_indices['sent_len'].max()+1), 
                         dtype=np.int32)
            word_targets = np.empty((len(my_indices)), dtype=np.int32)
            assert x.size <= batch_size
            x.fill(word2id['<pad>'])
            for i, (_, row) in enumerate(my_indices.iterrows()):
                x[i,:row['sent_len']] = buffer[row['sent_start']:row['sent_stop']]
                x[i,row['sent_len']] = word2id['<eos>']
                word_targets[i] = x[i,row['word_index']]
                x[i,row['word_index']] = word2id['<target>']
            batches.append((x, my_indices['sent_len'].values,
                            my_indices['candidates'].values, 
                            my_indices['hdn'].values, word_targets))
        return batches

    def _run_lstm(self, session, out_vars, batch):
        x, lens, candidates, y, _ = batch
        feed_dict = {self._x: x, self._y: y, 
                     self._candidates_list: candidates, 
                     self._lens: lens}
        state = session.run(self._initial_state, feed_dict)
        c, h = self._initial_state
        feed_dict[c] = state.c
        feed_dict[h] = state.h
        return session.run(out_vars, feed_dict,
                           options=self.run_options, 
                           run_metadata=self.run_metadata)

    def train_epoch(self, session, batches, verbose=False):
        costs, weights = [], []
        random.shuffle(batches)
        for batch in tqdm(batches, unit='batch', desc="Training"):
            batch_cost, _ = self._run_lstm(session, 
                                           [self._cost, self._train_op], 
                                           batch)
            batch_size = len(batches[0])
            costs.append(batch_cost)
            weights.append(batch_size)
        return np.average(costs, weights=weights)

    def measure_dev_cost(self, session, batches):
        results, weights = [], []
        for batch in tqdm(batches, unit='batch', desc="Evaluating"):
            results.append(self._run_lstm(session, [self._cost, self._accuracy], batch))
            weights.append(len(batch[0]))
        return np.average(results, axis=0, weights=weights)
        
    def _predict(self, session, data, word2id, compute_probs=True, compute_y=True):
        probs, y_hat = [], []
        out_vars = [self._probs if compute_probs else [0],
                    self._y_hat if compute_y else [0]]
        batches = self._gen_batches(data, self.config.predict_batch_size, word2id)
        if len(batches) > 10:
            batches = tqdm(batches, desc="Predicting", unit="batch")
        start = 0
        for x, lens, candidates, _, y in batches:
            candidates = candidates.copy()
            candidates[candidates == -1] = self._unmask_index
            batch = (x, lens, candidates, None)
            probs_batch, y_hat_batch = self._run_lstm(session, out_vars, batch)
            probs.append(probs_batch)
            y_hat.append(y_hat_batch)
            start += len(x)
        assert start == len(y_hat) == len(probs)
        return np.vstack(probs), np.vstack(y_hat)
        
    def predict(self, sess, data, word2id):
        _, y_hat = self._predict(sess, data, word2id, compute_probs=False)
        return y_hat
        
    def predict_proba(self, sess, data, word2id):
        probs, _ = self._predict(sess, data, word2id, compute_y=False)
        return probs
    
            
def from_npz_to_batches(npz, full_vocab, prepare_subvocabs):
    batches = []
    num_batches = sum(1 for key in npz if key.startswith('batch'))
    for i in range(num_batches):
#         if i >= 10: break # for debugging
        sentences, lens = npz['batch%d' %i], npz['lens%d' %i]
        if prepare_subvocabs: 
            batch_vocab, inverse = np.unique(sentences, return_inverse=True)
            outputs = inverse.reshape(sentences.shape)
            sys.stderr.write('Batch %d of %d, vocab size: %d (%.2f%% of original)\n'
                             %(i, num_batches, batch_vocab.size, 
                               batch_vocab.size*100.0/len(full_vocab)))
        else:
            outputs, batch_vocab = sentences, np.empty(0)
        batches.append((sentences, outputs, batch_vocab, lens))
    return batches
            
            
def load_data(FLAGS, prepare_subvocabs=False):
    sys.stderr.write('Loading data from data_path=%s, vocab_path=%s, dev_path=%s...\n'
                     %(FLAGS.data_path, getattr(FLAGS, 'vocab_path', ''), getattr(FLAGS, 'dev_path', '')))
    full_vocab = np.load(FLAGS.vocab_path if getattr(FLAGS, 'vocab_path', '') != ''
                         else FLAGS.data_path + '.index.pkl')
    train = np.load(FLAGS.data_path + '.train.npz')
    train_batches = from_npz_to_batches(train, full_vocab, prepare_subvocabs)
    dev = np.load(FLAGS.dev_path if getattr(FLAGS, 'dev_path', '') != '' 
                  else FLAGS.data_path + '.dev.npz')
    dev_batches = from_npz_to_batches(dev, full_vocab, False)
    sys.stderr.write('Loading data... Done.\n')
    return full_vocab, train_batches, dev_batches


def train_model(m_train, m_evaluate, FLAGS, config):
    vocab, train_batches, dev_batches = load_data(FLAGS, prepare_subvocabs=config.sampled_softmax)
    target_id = vocab['<target>']

    best_cost = None # don't know how to update this within a managed session yet
    stagnant_count = tf.get_variable("stagnant_count", initializer=0, dtype=tf.int32, trainable=False)
    reset_stag = tf.assign(stagnant_count, 0)
    inc_stag = tf.assign_add(stagnant_count, 1)
    epoch = tf.get_variable("epoch", initializer=0, dtype=tf.int32, trainable=False)
    inc_epoch = tf.assign_add(epoch, 1)
    
    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep if hasattr(FLAGS, 'max_to_keep') else 1)
    best_model_saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=FLAGS.save_path, saver=saver) #, save_model_secs=60) # for testing
    with sv.managed_session() as sess:
        start_time = time.time()
        for i in range(sess.run(epoch), config.max_epoch):
            # only turn it on after 5 epochs because first epochs spend time 
            # on GPU initialization routines
            if hasattr(FLAGS, 'trace_timeline') and FLAGS.trace_timeline and i == 5: 
                m_train.trace_timeline() # start tracing timeline
            print("Epoch #%d:" % (i + 1))
#             train_cost = 0 # for debugging
            train_cost = m_train.train_epoch(sess, train_batches, target_id, verbose=True)
            print("Epoch #%d finished:" %(i + 1))
            print("\tTrain cost: %.3f" %train_cost) 
            saver.save(sess, FLAGS.save_path, global_step=i)
            if m_evaluate:
                dev_cost, hit_at_100 = m_evaluate.measure_dev_cost(sess, dev_batches, target_id)
                print("\tDev cost: %.3f, hit@100: %.1f%%" %(dev_cost, hit_at_100))
                if best_cost is None or dev_cost < best_cost:
                    best_cost = dev_cost
                    save_path = best_model_saver.save(sess, FLAGS.save_path + '-best-model')
                    print("\tSaved best model to %s" %save_path)
                    sess.run(reset_stag)
                else:
                    sess.run(inc_stag)
                    if (config.max_stagnant_count > 0 and 
                        sess.run(stagnant_count) >= config.max_stagnant_count):
                        print("Stopped early because development cost "
                              "didn't decrease for %d consecutive epochs." 
                              %config.max_stagnant_count)
                        break
            print("\tElapsed time: %.1f minutes" %((time.time()-start_time)/60))
            sess.run(inc_epoch)
