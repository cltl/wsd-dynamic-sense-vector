"""
Adopted from TensorFlow LSTM demo: 

    https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py

Also borrow some parts from this guide:

    http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

"""
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import sys
from model import WSDModelTrain, WSDModelEvaluate, DummyModelTrain

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "small",
    "A type of model. Possible options are: small, medium, large, google.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("trace_timeline", False,
                  "Trace execution time to find out bottlenecks.")
FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.1
  max_grad_norm = 5
  hidden_size = 100
  max_epoch = 10
  emb_dims = 10


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 0.1
  max_grad_norm = 5
  hidden_size = 200
  max_epoch = 39
  emb_dims = 100


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 0.1
  max_grad_norm = 10
  hidden_size = 512
  max_epoch = 100
  emb_dims = 128


class GoogleConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 0.1
  max_grad_norm = 5
  hidden_size = 2048
  max_epoch = 1000
  emb_dims = 512


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 0.1
  max_grad_norm = 1
  hidden_size = 2
  max_epoch = 1
  batch_size = 20

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "google":
    return GoogleConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
    
def load_data():
    vocab = np.load(FLAGS.data_path + '.index.pkl')
    target_id = vocab['<target>']
    train = np.load(FLAGS.data_path + '.train.npz')
    dev = np.load(FLAGS.data_path + '.dev.npz')
    return (vocab, train['sents'], train['vocabs'], train['indices'], 
            dev['data'], dev['lens'], target_id)

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to the base path of "
                         "prepared input (e.g. output/gigaword)")
    vocab, train_sents, train_vocabs, train_indices, dev_data, dev_lens, target_id = load_data()
    config = get_config()
    config.vocab_size = len(vocab)
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m_train = WSDModelTrain(config, data_type())
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
        m_evaluate = WSDModelEvaluate(config, data_type())
    m_train.print_device_placement()
    with tf.Session() as session:
        saver = tf.train.Saver()
        start_time = time.time()
        m_train.init_data(session, train_sents, train_vocabs, train_indices, verbose=True)
        sys.stdout.write("Initializing variables.... ")
        session.run(tf.global_variables_initializer())
        sys.stdout.write("Done.\n")
        best_cost = None
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        for i in range(config.max_epoch):
            # only turn it on after 5 epochs because first epochs spend time 
            # on GPU initialization routines
            if FLAGS.trace_timeline and i == 5: 
                m_train.trace_timeline()
            print("Epoch: %d" % (i + 1))
#             train_cost = 0 # for debugging
            train_cost = m_train.train_epoch(session, verbose=True)
            dev_cost, hit_at_100 = m_evaluate.measure_dev_cost(session, dev_data, dev_lens, target_id)
            print("Epoch: %d finished, elapsed time: %.1f minutes" % 
                  (i + 1, (time.time()-start_time)/60))
            print("\tTrain cost: %.3f" %train_cost)
            print("\tDev cost: %.3f, hit@100: %.1f%%" %(dev_cost, hit_at_100))
            if best_cost is None or dev_cost < best_cost:
                best_cost = dev_cost
#                 save_start = time.time()
                print("\tSaved best model to %s" %saver.save(session, FLAGS.save_path))
#                 print("\tTime on saving: %f sec" %(time.time()-save_start))
        coord.request_stop()
        coord.join(threads)
    if FLAGS.trace_timeline:
        tl = timeline.Timeline(m_train.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        timeline_path = 'output/timeline.json'
        with open(timeline_path, 'w') as f: f.write(ctf)
        print('Timeline written to %s' %timeline_path)

if __name__ == "__main__":
    tf.app.run()
