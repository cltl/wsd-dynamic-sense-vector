import numpy as np
import tensorflow as tf

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
        self._x = tf.placeholder(tf.int32, shape=[None, None])
        self._y = tf.placeholder(tf.int32, shape=[None])
        self._subvocab = tf.placeholder(tf.int32, shape=[None])

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
        self._initial_state = cell.zero_state(tf.shape(self._x)[0], float_dtype)

        self.run_options = self.run_metadata = None
    
    def trace_timeline(self):
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

    def train_epoch(self, session, data, target_id, verbose=False):
        """Runs the model on the given data."""
        total_cost = 0.0
        total_rows = 0
        
        # resample the batches so that each token has equal chance to become target
        # another effect is to randomize the order of batches
        sentence_lens = np.array([x.shape[1] for x, _, _, in data])
        samples = np.random.choice(len(data), size=len(data), 
                                   p=sentence_lens/sentence_lens.sum())
        for batch_no, batch_id in enumerate(samples):
            x, y_all, subvocab = data[batch_id]
            i =  np.random.randint(x.shape[1])
            y = y_all[:,i]
            old_xi = x[:,i].copy()
            x[:,i] = target_id
    
            feed_dict = {self._x: x, self._y: y, self._subvocab: subvocab}
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
        # copying is needed here because one sentence might be chosen multiple times
        # injecting targets into multiple references of the same sentence will
        # create a mess
        x, lens2 = data[samples].copy(), lens[samples] 
        # sample targets
        target_indices = np.mod(np.random.randint(1000000, size=lens2.shape), lens2)
        one_to_n = np.arange(lens2.size)
        y = x[one_to_n, target_indices]
        x[one_to_n, target_indices] = target_id
        max_batch_size = 1000
        total_cost = 0.0
        total_hit = 0.0
        for batch_start in range(0, len(x), max_batch_size):
            batch_end = min(len(x), batch_start+max_batch_size)
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
        