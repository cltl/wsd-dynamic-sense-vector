"""
Adopted from TensorFlow LSTM demo: 

    https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py

Also borrow some parts from this guide:

    http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

"""
import numpy as np
import tensorflow as tf
from model import WSIModel, train_model
from configs import get_config
import random

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("seed", 192, 
                     "A random seed to make sure the experiment is repeatable")
flags.DEFINE_string("model", "small",
                    "A type of model. Possible options are: small, medium, large, google.")
flags.DEFINE_string("data_path", None,
                    "Where the training/valid data is stored.")
flags.DEFINE_string("dev_path", '',
                    "Where the valid data is stored, if it cannot be inferred from data_path.")
flags.DEFINE_string("vocab_path", '',
                    "Where the vocabulary is stored, if it cannot be inferred from data_path.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("trace_timeline", False,
                  "Trace execution time to find out bottlenecks.")
flags.DEFINE_integer("max_to_keep", 1, 
                     "Number of models (at different epochs) to keep around")
FLAGS = flags.FLAGS

def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(random.randint(0, 10**6))
    tf.set_random_seed(random.randint(0, 10**6))
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to the base path of "
                         "prepared input (e.g. output/gigaword)")
    config = get_config(FLAGS)
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m_train = WSIModel(config, optimized=True)
    with tf.variable_scope("Model", reuse=True):
        m_evaluate = WSIModel(config, reuse_variables=True)
    train_model(m_train, m_evaluate, FLAGS, config)

if __name__ == "__main__":
    tf.app.run()
