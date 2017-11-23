import numpy as np
import tensorflow as tf
import time
import sys

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

    def __init__(self, config, optimized=False, reuse_variables=False):
        self.config = config
        self.optimized = optimized
        self.reuse_variables = reuse_variables
        self._build_inputs()
        self._build_word_embeddings()
        self._build_lstm_output()
        self._build_context_embs()
        self._build_logits()
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
        self._hit_at_100 = tf.reduce_mean(tf.cast(
                tf.nn.in_top_k(self._logits, self._y, 100), float_dtype))
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          self.config.max_grad_norm)
        optimizer = tf.train.AdagradOptimizer(self.config.learning_rate)
        self._global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                global_step=self._global_step)
    
    def trace_timeline(self):
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

    def train_epoch(self, session, data, target_id, verbose=False):
        """Runs the model on the given data."""
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
            i =  np.random.randint(x.shape[1])
            y = y_all[:,i]
            old_xi = x[:,i].copy()
            x[:,i] = target_id
    
            feed_dict = {self._x: x, self._y: y, self._subvocab: subvocab, self._lens: lens}
            state = session.run(self._initial_state, feed_dict)
            c, h = self._initial_state
            feed_dict[c] = state.c
            feed_dict[h] = state.h
    
            batch_cost, _ = session.run([self._cost, self._train_op], feed_dict,
                                        options=self.run_options, 
                                        run_metadata=self.run_metadata)
            x[:,i] = old_xi # restore the data
    
            total_cost += batch_cost * x.shape[0] # because the cost is averaged
            total_rows += x.shape[0]              # over rows in a batch
            
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

    def measure_dev_cost(self, session, data, lens, target_id, max_batch_size=1000):
        # resample the sentences so that each token has equal chance to become target
        samples = np.random.choice(len(data), size=len(data), p=lens/lens.sum())
        # copying is needed here because one sentence might be chosen multiple times
        # injecting targets into multiple references of the same sentence will
        # create a mess
        x, lens2 = data[samples].copy(), lens[samples] 
        # sample targets
        target_indices = np.mod(np.random.randint(1000000, size=lens2.shape), lens2)
        one_to_n = np.arange(lens2.size)
        y = x[one_to_n, target_indices]
        x[one_to_n, target_indices] = target_id
        total_cost = 0.0
        total_hit = 0.0
        for batch_start in range(0, len(x), max_batch_size):
            batch_end = min(len(x), batch_start + max_batch_size)
            batch_x = x[batch_start:batch_end]
            batch_y = y[batch_start:batch_end]
            batch_lens = lens2[batch_start:batch_end]
            batch_size = batch_end - batch_start
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
        return total_cost / len(x), total_hit / len(x)

class WSIModel(WSDModel):
    """A LSTM word sense induction (WSI) model designed for fast training."""

    def _build_logits(self):
        if self.optimized and self.config.sampled_softmax:
            E_contexts = tf.get_variable("context_embedding", 
                    [self.config.vocab_size, self.config.num_senses, self.config.emb_dims], 
                    dtype=float_dtype)
            subcontexts = tf.nn.embedding_lookup(E_contexts, self._subvocab)
            subvocab_size = tf.shape(self._subvocab)[0]
            sense_logits = tf.matmul(self._predicted_context_embs, tf.transpose(
                    tf.reshape(subcontexts, (-1, self.config.emb_dims))))
            self._logits = tf.reduce_max(tf.reshape(sense_logits, 
                    (-1, subvocab_size, self.config.num_senses)), axis=2)
        else:
            E_contexts = tf.get_variable("context_embedding")
            sense_logits = tf.matmul(self._predicted_context_embs, tf.transpose(
                tf.reshape(E_contexts, (-1, self.config.emb_dims))))
            self._logits = tf.reduce_max(tf.reshape(sense_logits,
                    (-1, self.config.vocab_size, self.config.num_senses)), axis=2)
            
def load_data(FLAGS, prepare_subvocabs=False):
    sys.stderr.write('Loading data...\n')
    full_vocab = np.load(FLAGS.vocab_path if hasattr(FLAGS, 'vocab_path')
                         else FLAGS.data_path + '.index.pkl')
    train = np.load(FLAGS.data_path + '.train.npz')
    train_batches = []
    num_batches = sum(1 for key in train if key.startswith('batch'))
    for i in range(num_batches):
        sentences, lens = train['batch%d' %i], train['lens%d' %i]
        if prepare_subvocabs: 
            batch_vocab, inverse = np.unique(sentences, return_inverse=True)
            outputs = inverse.reshape(sentences.shape)
            sys.stderr.write('Batch %d of %d, vocab size: %d (%.2f%% of original)\n'
                             %(i, num_batches, batch_vocab.size, batch_vocab.size*100.0/len(full_vocab)))
        else:
            outputs, batch_vocab = sentences, np.empty(0)
        train_batches.append((sentences, outputs, batch_vocab, lens))
#         if i >= 10: break # for debugging
    dev = np.load(FLAGS.data_path + '.dev.npz')
    sys.stderr.write('Loading data... Done.\n')
    return full_vocab, train_batches, dev['data'], dev['lens']

def train_model(m_train, m_evaluate, FLAGS, config):
    vocab, train_batches, dev_data, dev_lens = load_data(FLAGS, prepare_subvocabs=config.sampled_softmax)
    target_id = vocab['<target>']

    best_cost = None # don't know how to update this within a managed session yet
    stagnant_count = tf.get_variable("stagnant_count", initializer=0, dtype=tf.int32, trainable=False)
    reset_stag = tf.assign(stagnant_count, 0)
    inc_stag = tf.assign_add(stagnant_count, 1)
    epoch = tf.get_variable("epoch", initializer=0, dtype=tf.int32, trainable=False)
    inc_epoch = tf.assign_add(epoch, 1)
    
    saver = tf.train.Saver(max_to_keep=0)
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
                dev_cost, hit_at_100 = m_evaluate.measure_dev_cost(sess, dev_data, dev_lens, target_id)
                print("\tDev cost: %.3f, hit@100: %.1f%%" %(dev_cost, hit_at_100))
                if best_cost is None or dev_cost < best_cost:
                    best_cost = dev_cost
                    save_path = saver.save(sess, FLAGS.save_path + '-best-model')
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
