"""
Adopted from TensorFlow LSTM demo: 

    https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py

Also borrow some parts from this guide:

    http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

"""
import time

import numpy as np
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class WSDModel(object):
    """The PTB model."""

    def __init__(self, config):
        self._x = tf.placeholder(tf.int32, shape=[None, None], name='x')
        self._y = tf.placeholder(tf.int32, shape=[None])
        
        E_words = tf.get_variable("word_embedding", 
                [config.vocab_size, config.emb_dims], dtype=data_type())
        word_embs = tf.nn.embedding_lookup(E_words, self._x)
        cell = tf.contrib.rnn.LSTMCell(num_units=config.hidden_size,
                                       state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell, word_embs, dtype=data_type())
        context_layer_weights = tf.get_variable("context_layer_weights",
                [config.hidden_size, config.emb_dims], dtype=data_type())
        self._predicted_context_embs = tf.matmul(outputs[:,-1], context_layer_weights, 
                                                 name='predicted_context_embs')
        E_contexts = tf.get_variable("context_embedding", 
                [config.vocab_size, config.emb_dims], dtype=data_type())
        pre_probs = tf.matmul(self._predicted_context_embs, tf.transpose(E_contexts))
        
        self._cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pre_probs, labels=self._y))

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdagradOptimizer(config.learning_rate)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._initial_state = cell.zero_state(tf.shape(self._x)[0], data_type())

    @property
    def predict(self):
        return self._predicted_context_embs

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.1
  max_grad_norm = 5
  hidden_size = 100
  max_epoch = 10
  batch_size = 20
  vocab_size = None # to be assigned
  emb_dims = 10


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 0.1
  max_grad_norm = 5
  hidden_size = 200
  max_epoch = 39
  batch_size = 20
  vocab_size = None # to be assigned
  emb_dims = 100


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 0.1
  max_grad_norm = 10
  hidden_size = 512
  max_epoch = 55
  batch_size = 20
  vocab_size = None # to be assigned
  emb_dims = 128


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 0.1
  max_grad_norm = 1
  hidden_size = 2
  max_epoch = 1
  batch_size = 20
  vocab_size = None # to be assigned


def train_epoch(session, model, data, target_id, verbose=False):
    """Runs the model on the given data."""
    total_cost = 0.0
    total_rows = 0

    fetches = { "cost": model.cost, "eval_op": model.train_op }
    for batch_no, x in enumerate(data):
        i =  np.random.randint(x.shape[1])
        y = x[:,i].copy() # copy content
        x[:,i] = target_id

        state = session.run(model.initial_state, {model.x: x})
        feed_dict = {model.x: x, model.y: y}
        c, h = model.initial_state
        feed_dict[c] = state.c
        feed_dict[h] = state.h

        vals = session.run(fetches, feed_dict)
        batch_cost = vals["cost"]
        x[:,i] = y

        total_cost += batch_cost * x.shape[0]
        total_rows += x.shape[0]
        
        if verbose and (batch_no+1) % 100 == 0:
            print("batch #%d cost: %.7f" %(batch_no+1, batch_cost))
    return total_cost / total_rows


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def print_device_placement(model):
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print("******** Start of device placement ********")
        sess.run(tf.initialize_all_variables())
        x, y = np.zeros((100, 10)), np.zeros(100)
        c, h = model.initial_state
        state = sess.run(model.initial_state, {model.x: x})
        feed_dict = {model.x: x, model.y: y, c: state.c, h: state.h}
        print(sess.run(model.train_op, feed_dict))
        print("******** End of device placement ********")

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to the base path of "
                         "prepared input (e.g. output/gigaword)")
    vocab = np.load(FLAGS.data_path + '.index.pkl')
    target_id = vocab['<target>']
    train = np.load(FLAGS.data_path + '.train.npz')
    train_batches = [train['batch%d' %i] for i in range(len(train.keys()))]
    config = get_config()
    config.vocab_size = len(vocab)
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = WSDModel(config=config)
    print_device_placement(m)
    with tf.Session() as session:
        saver = tf.train.Saver()
        start_time = time.time()
        session.run(tf.initialize_all_variables())
        for i in range(config.max_epoch):
            print("Epoch: %d" % (i + 1))
            train_cost = 0
            train_cost = train_epoch(session, m, train_batches, target_id, verbose=True)
            print("Epoch: %d -> Train cost: %.3f, elapsed time: %.1f minutes" % 
                  (i + 1, train_cost, (time.time()-start_time)/60))
        if FLAGS.save_path:
            print("Saved model to %s." %saver.save(session, FLAGS.save_path))

if __name__ == "__main__":
    tf.app.run()
