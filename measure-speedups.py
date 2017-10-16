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
from configs import get_config, GoogleConfig, gigaword_for_lstm_wsd_path

flags = tf.flags
logging = tf.logging

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


class WSDModelTrainUnoptimized(WSDModelTrain):

    def __init__(self, config):
        super(WSDModelTrainUnoptimized, self).__init__(config)
        self.config = config
        
    def _build_logits(self, config):
        if self.config.sampled_softmax:
            super(WSDModelTrainUnoptimized, self)._build_logits(config)
        else:
            E_contexts = tf.get_variable("context_embedding")
            self._logits = tf.matmul(self._predicted_context_embs, tf.transpose(E_contexts))

    def train_epoch(self, session, data, target_id, verbose=False, lens=None):
        if config.training_on_same_length_sentences:
            return super(WSDModelTrainUnoptimized, self).train_epoch(session, data, target_id, verbose)
        else:
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

def main(_):
    for config in (Baseline, SeparateTrainingAndEvaluating, 
                   SampledSoftmax, OptimizedBatches):
        if config.optimized_batches:
            FLAGS.data_path = gigaword_for_lstm_wsd_path + '.train.npz'
        else:
            FLAGS.data_path = gigaword_for_lstm_wsd_path + '.train-shuffled.npz'
        tf.reset_default_graph()
        config = GoogleConfig
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)
        if config.training_on_same_length_sentences:
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m_train = WSDModelTrain(config)
        else:
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m_train = WSDModelEvaluate(config)
        train_model(m_train, None, FLAGS, config)

if __name__ == "__main__":
    tf.app.run()
