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
from configs import get_config, SmallConfig
from version import version
import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("config", "",
                    "Choose the type of optimization to test. Possible options are: "
                    "baseline, same-length, sampled-softmax, optimized-batches. "
                    "If not provided, test all possible configs.")
FLAGS = flags.FLAGS

class Baseline(SmallConfig):
    name = 'baseline'
    assume_same_lengths = False
    sampled_softmax = False
    optimized_batches = False
    max_epoch = 10
    
class AssumeSameLengths(Baseline):
    name = 'same-length'
    assume_same_lengths = True
    sampled_softmax = False
    optimized_batches = False
    
class SampledSoftmax(Baseline):
    name = 'sampled-softmax'
    assume_same_lengths = True
    sampled_softmax = True
    optimized_batches = False
    
class OptimizedBatches(Baseline):
    name = 'optimized-batches'
    assume_same_lengths = True
    sampled_softmax = True
    optimized_batches = True
    
all_configs = (Baseline, AssumeSameLengths, SampledSoftmax, OptimizedBatches)

def main(_):
    tf.set_random_seed(252)
    if FLAGS.config:
        config, = [cf for cf in all_configs if cf.name == FLAGS.config]
        configs = [config]
    else:
        configs = all_configs
        
    gigaword_for_lstm_wsd_path = os.path.join('preprocessed-data', '2017-11-16-a9618a6', 'gigaword-for-lstm-wsd')
    for config in configs:
        if config.optimized_batches:
            FLAGS.data_path = gigaword_for_lstm_wsd_path
        else:
            FLAGS.data_path = gigaword_for_lstm_wsd_path + '-shuffled'
            FLAGS.vocab_path = gigaword_for_lstm_wsd_path + '.index.pkl'
        FLAGS.save_path = os.path.join('output', version, 'speedups-%s' %config.name)
            
        tf.reset_default_graph()
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m_train = WSDModel(config, optimized=True)
        with tf.variable_scope("Model", reuse=True):
            m_evaluate = WSDModel(config, reuse_variables=True)
        train_model(m_train, m_evaluate, FLAGS, config)

if __name__ == "__main__":
    tf.app.run()
