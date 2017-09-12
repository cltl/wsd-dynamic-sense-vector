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
from configs import get_config, GoogleConfig

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
FLAGS = flags.FLAGS

class Baseline(GoogleConfig):
    training_on_same_length_sentences = False
    sampled_softmax = False
    optimized_batches = False
    max_epoch = 1
    
class SeparateTrainingAndEvaluating(Baseline):
    training_on_same_length_sentences = True
    sampled_softmax = False
    optimized_batches = False
    
class SampledSoftmax(Baseline):
    training_on_same_length_sentences = True
    sampled_softmax = True
    optimized_batches = False
    
class OptimizedBatches(Baseline):
    training_on_same_length_sentences = True
    sampled_softmax = True
    optimized_batches = True

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to the base path of "
                         "prepared input (e.g. output/gigaword)")
    for config in (Baseline, SeparateTrainingAndEvaluating, 
                   SampledSoftmax, OptimizedBatches):
        tf.reset_default_graph()
        config = GoogleConfig
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m_train = WSDModelTrain(config)
        if config is Baseline: 
            # do it only once. that's enough 
            m_train.print_device_placement()
        train_model(m_train, None, FLAGS, config)

if __name__ == "__main__":
    tf.app.run()
