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
from model import WSDModelTrain, WSDModelEvaluate, DummyModelTrain, train_model
from configs import get_config, data_type

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "small",
    "A type of model. Possible options are: small, medium, large, google.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("trace_timeline", False,
                  "Trace execution time to find out bottlenecks.")
FLAGS = flags.FLAGS

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to the base path of "
                         "prepared input (e.g. output/gigaword)")
    config = get_config()
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m_train = WSDModelTrain(config, data_type())
    with tf.variable_scope("Model", reuse=True):
        m_evaluate = WSDModelEvaluate(config, data_type())
    m_train.print_device_placement()
    train_model(m_train, m_evaluate, FLAGS, config)

    if FLAGS.trace_timeline:
        tl = timeline.Timeline(m_train.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        timeline_path = 'output/timeline.json'
        with open(timeline_path, 'w') as f: f.write(ctf)
        print('Timeline written to %s' %timeline_path)

if __name__ == "__main__":
    tf.app.run()
