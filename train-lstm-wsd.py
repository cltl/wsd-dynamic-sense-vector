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
from model import WSDModel, train_model
from configs import get_config

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
FLAGS = flags.FLAGS

def main(_):
    tf.set_random_seed(FLAGS.seed)
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to the base path of "
                         "prepared input (e.g. output/gigaword)")
    config = get_config(FLAGS)
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m_train = WSDModel(config, optimized=True)
    with tf.variable_scope("Model", reuse=True):
        m_evaluate = WSDModel(config, reuse_variables=True)
#     m_train.print_device_placement() # for debugging
    train_model(m_train, m_evaluate, FLAGS, config)

    if FLAGS.trace_timeline:
        tl = timeline.Timeline(m_train.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        timeline_path = 'output/timeline.json'
        with open(timeline_path, 'w') as f: f.write(ctf)
        print('Timeline written to %s' %timeline_path)

if __name__ == "__main__":
    tf.app.run()
