import numpy as np
import tensorflow as tf
import time
import sys

class DummyModelTrain(object):
    '''
    This is for testing GPU usage only. This model runs very trivial operations
    on GPU therefore its running time is mostly on CPU. Compared to WSDModelTrain,
    this model should run much faster, otherwise you're spending too much time
    on CPU.
    '''

    def __init__(self, config, float_dtype):
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

class WSDModelTrain(object):
    """A LSTM WSD model designed for fast training."""

    def __init__(self, config, float_dtype):
        self._sents_val = tf.placeholder(tf.int32, shape=[None])
        self._subvocabs_val = tf.placeholder(tf.int32, shape=[None])
        self._indices_val = tf.placeholder(tf.int32, shape=[None, 6])

        self._sents = tf.Variable(self._sents_val, name='data_sents', 
                                  trainable=False, collections=[], validate_shape=False)
        self._subvocabs = tf.Variable(self._subvocabs_val, name='data_subvocabs',
                                      trainable=False, collections=[], validate_shape=False)
        self._indices = tf.Variable(self._indices_val, name='data_indices',
                                    trainable=False, collections=[], validate_shape=False) 
        
        i, = tf.train.slice_input_producer([self._indices])
        data = tf.reshape(self._sents[i[0]:i[0]+i[1]*i[2]], (i[1], i[2]))
        self._subvocab = self._subvocabs[i[3]:i[3]+i[4]]
        target_id = i[5]
        col = tf.random_uniform((1,), maxval=tf.shape(data)[1], dtype=tf.int32)
        self._y = data[:, col[0]]
        data_tmp = tf.Variable(0, dtype=tf.int32)
        data_tmp = tf.assign(data_tmp, tf.transpose(data), validate_shape=False)
        col_of_target_ids = tf.fill((tf.shape(data)[0],), target_id)
        self._x = tf.transpose(tf.scatter_nd_update(data_tmp, [col], [col_of_target_ids]))
        
        E_words = tf.get_variable("word_embedding", 
                [config.vocab_size, config.emb_dims], dtype=float_dtype)
        word_embs = tf.nn.embedding_lookup(E_words, self._x)
        cell = tf.contrib.rnn.LSTMCell(num_units=config.hidden_size,
                                       state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell, word_embs, dtype=float_dtype)
        context_layer_weights = tf.get_variable("context_layer_weights",
                [config.hidden_size, config.emb_dims], dtype=float_dtype)
        self._predicted_context_embs = tf.matmul(outputs[:,-1], context_layer_weights, 
                                                 name='predicted_context_embs')
        E_contexts = tf.get_variable("context_embedding", 
                [config.vocab_size, config.emb_dims], dtype=float_dtype)
        subcontexts = tf.nn.embedding_lookup(E_contexts, self._subvocab)
        pre_probs = tf.matmul(self._predicted_context_embs, tf.transpose(subcontexts))
        
        self._cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pre_probs, labels=self._y))

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdagradOptimizer(config.learning_rate)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

        self.run_options = self.run_metadata = None
        
    def init_data(self, sess, sents_val, subvocabs_val, indices_val, verbose=False):
        start_time = time.time()
        if verbose: sys.stdout.write('Loading data... ')
        sess.run(self._sents.initializer, feed_dict={self._sents_val: sents_val})
        sess.run(self._subvocabs.initializer, feed_dict={self._subvocabs_val: subvocabs_val})
        sess.run(self._indices.initializer, feed_dict={self._indices_val: indices_val})
        self._num_batches = sess.run(tf.shape(self._indices)[0])
        elapsed_time = (time.time()-start_time)/60
        if verbose: sys.stdout.write('Done (%f min).\n' %elapsed_time)
    
    def trace_timeline(self):
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

    def train_epoch(self, session, verbose=False):
        """Runs the model on the given data."""
        total_cost = 0.0
        total_rows = 0
        for batch_no in range(self._num_batches):
            batch_cost, y_val, _ = session.run([self._cost, self._y, self._train_op],
                                                options=self.run_options, 
                                                run_metadata=self.run_metadata)
            total_cost += batch_cost * y_val.shape[0] # because the cost is averaged
            total_rows += y_val.shape[0]              # over rows in a batch
            
            if verbose and (batch_no+1) % 100 == 0:
                print("\tsample batch cost: %.7f" %batch_cost)
        return total_cost / total_rows
    
    def print_device_placement(self):
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            print("******** Start of device placement ********")
            sents = np.random.randint(10, size=100)
            subvocabs = np.random.randint(10, size=90)
            indices = np.array([[0, 4, 20, 0, 40, 3], [0, 2, 10, 40, 50, 2]])
            self.init_data(sess, sents, subvocabs, indices)
            sess.run(tf.global_variables_initializer())
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(self._train_op)
            coord.request_stop()
            coord.join(threads)
            print("******** End of device placement ********")


class WSDModelEvaluate(object):
    """A LSTM WSD model designed for accurate evaluation, sharing parameters
    with @WSDModelTrain."""

    def __init__(self, config, float_dtype):
        self._x = tf.placeholder(tf.int32, shape=[None, None], name='x')
        self._lens = tf.placeholder(tf.int32, shape=[None], name='lens')
        self._y = tf.placeholder(tf.int32, shape=[None], name='y')
        
        E_words = tf.get_variable("word_embedding", 
                [config.vocab_size, config.emb_dims], dtype=float_dtype)
        word_embs = tf.nn.embedding_lookup(E_words, self._x)
        cell = tf.contrib.rnn.LSTMCell(num_units=config.hidden_size,
                                       state_is_tuple=True, reuse=True)
        outputs, _ = tf.nn.dynamic_rnn(cell, word_embs, 
                                       sequence_length=self._lens,
                                       dtype=float_dtype)
        context_layer_weights = tf.get_variable("context_layer_weights",
                [config.hidden_size, config.emb_dims], dtype=float_dtype)
        last_output_indices = tf.stack([tf.range(tf.shape(self._x)[0]), self._lens-1], axis=1)
        last_outputs = tf.gather_nd(outputs, last_output_indices)
        self._predicted_context_embs = tf.matmul(last_outputs, context_layer_weights, 
                                                 name='predicted_context_embs')
        E_contexts = tf.get_variable("context_embedding", 
                [config.vocab_size, config.emb_dims], dtype=float_dtype)
        pre_probs = tf.matmul(self._predicted_context_embs, tf.transpose(E_contexts))
        
        pre_probs_32 = (tf.cast(pre_probs, tf.float32) if float_dtype != tf.float32 
                        else pre_probs)
        self._hit_at_100 = tf.reduce_mean(tf.cast(
                tf.nn.in_top_k(pre_probs_32, self._y, 100), float_dtype))
        self._cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pre_probs, labels=self._y))
        self._initial_state = cell.zero_state(tf.shape(self._x)[0], float_dtype)
        
    def measure_dev_cost(self, session, data, lens, target_id):
        # resample the sentences so that each token has equal chance to become target
        samples = np.random.choice(len(data), size=len(data), p=lens/lens.sum())
        data2, lens2 = data[samples].copy(), lens[samples]
        # sample targets
        target_indices = np.mod(np.random.randint(1000000, size=lens2.shape), lens2)
        targets = data2[np.arange(lens2.size), target_indices]
        data2[np.arange(lens2.size), target_indices] = target_id
        # initialize the RNN
        feed_dict = { self._x: data2, self._y: targets, self._lens: lens2}
        state = session.run(self._initial_state, feed_dict)
        c, h = self._initial_state
        feed_dict[c] = state.c
        feed_dict[h] = state.h
        # now it's time to evaluate
        cost, hit_at_100 = session.run([self._cost, self._hit_at_100], feed_dict)        
        return cost, hit_at_100


class WSIModelTrain(WSDModelTrain):
    """A LSTM word sense induction (WSI) model designed for fast training."""

    def __init__(self, config, float_dtype, sense_num=4):
        self._x = tf.placeholder(tf.int32, shape=[None, None], name='x')
        self._y = tf.placeholder(tf.int32, shape=[None], name='y')
        self._subvocab = tf.placeholder(tf.int32, shape=[None], name='subvocab')
        
        E_words = tf.get_variable("word_embedding", 
                [config.vocab_size, config.emb_dims], dtype=float_dtype)
        word_embs = tf.nn.embedding_lookup(E_words, self._x)
        cell = tf.contrib.rnn.LSTMCell(num_units=config.hidden_size,
                                       state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell, word_embs, dtype=float_dtype)
        context_layer_weights = tf.get_variable("context_layer_weights",
                [config.hidden_size, config.emb_dims], dtype=float_dtype)
        self._predicted_context_embs = tf.matmul(outputs[:,-1], context_layer_weights, 
                                                 name='predicted_context_embs')
        E_contexts = tf.get_variable("context_embedding", 
                [config.vocab_size, sense_num, config.emb_dims], dtype=float_dtype)
        subcontexts = tf.nn.embedding_lookup(E_contexts, self._subvocab)
        subvocab_size = tf.shape(self._subvocab)[0]
        pre_probs_senses = tf.matmul(self._predicted_context_embs, 
                tf.transpose(tf.reshape(subcontexts, (subvocab_size*sense_num, config.emb_dims))))
        pre_probs = tf.reduce_max(tf.reshape(pre_probs_senses, (-1, subvocab_size, sense_num)), axis=2)
        
        self._cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pre_probs, labels=self._y))

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdagradOptimizer(config.learning_rate)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

        self._initial_state = cell.zero_state(tf.shape(self._x)[0], float_dtype)
        self.run_options = self.run_metadata = None
        