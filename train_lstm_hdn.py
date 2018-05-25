"""
Adopted from TensorFlow LSTM demo: 

    https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py

Also borrow some parts from this guide:

    http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

"""
import numpy as np
import tensorflow as tf
from model import HDNModel
from configs import LargeConfig
import random
import pandas as pd
from version import version
import generate_hdn_datasets
from block_timer.timer import Timer


class MyConfig(LargeConfig):
    gigaword_path_pattern = generate_hdn_datasets.inp_pattern
    word_vocab_path = generate_hdn_datasets.word_vocab_path
    hdn_path_pattern = 'output/gigaword-hdn-%s.2018-05-18-f48a06c.pkl'
    num_senses = 16
    predict_batch_size = 32000
    train_batch_size = 8000


def prepare_batches(buffer, indices, batch_size, word2id, name='noname'):
    # make training and evaluating more efficient
    indices = indices.sort_values(by='sent_len', axis='index')
    batches = HDNModel.gen_batches((buffer, indices), batch_size, word2id,
                                   name=name)
    return batches, indices


def train_hdn_model(model, config):
    word2id = np.load(config.word_vocab_path)
    with Timer('Read Gigaword training set from %s' %(config.gigaword_path_pattern %'train')):
        buffer_train = np.load(config.gigaword_path_pattern %'train')['buffer']
    with Timer('Read HDN training indices from %s' %(config.hdn_path_pattern %'train')):
        hdn_train = pd.read_pickle(config.hdn_path_pattern %'train')
    with Timer('Read Gigaword development set from %s' %(config.gigaword_path_pattern %'dev')):
        buffer_dev = np.load(config.gigaword_path_pattern %'dev')['buffer']
    with Timer('Read HDN dev indices from %s' %(config.hdn_path_pattern %'dev')):
        hdn_dev = pd.read_pickle(config.hdn_path_pattern %'dev')
    train_batches, _ = prepare_batches(buffer_train, hdn_train, 
                                       config.train_batch_size, word2id,
                                       name='train')
    dev_batches, _ = prepare_batches(buffer_dev, hdn_dev, 
                                     config.predict_batch_size, word2id,
                                     name='dev')

    best_acc = None # don't know how to update this within a managed session yet
    stagnant_count = tf.get_variable("stagnant_count", initializer=0, dtype=tf.int32, trainable=False)
    reset_stag = tf.assign(stagnant_count, 0)
    inc_stag = tf.assign_add(stagnant_count, 1)
    epoch = tf.get_variable("epoch", initializer=0, dtype=tf.int32, trainable=False)
    inc_epoch = tf.assign_add(epoch, 1)
    
    saver = tf.train.Saver(max_to_keep=1)
    best_model_saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=config.save_path, saver=saver)
    with sv.managed_session() as sess, Timer("Training HDN model"):
        for i in range(sess.run(epoch), config.max_epoch):
            print("Epoch #%d:" % (i + 1))
            train_cost = model.train_epoch(sess, train_batches, word2id)
            dev_cost, dev_acc = model.measure_dev_cost(sess, dev_batches)
            print("\tTrain cost: %.3f" %train_cost) 
            print("\tDev cost: %.3f, accuracy: %.1f%%" %(dev_cost, dev_acc*100))
            if best_acc is None or dev_acc > best_acc:
                best_acc = dev_acc
                save_path = best_model_saver.save(sess, config.save_path + '-best-model')
                print("\tSaved best model to %s" %save_path)
                sess.run(reset_stag)
            else:
                sess.run(inc_stag)
                if (config.max_stagnant_count > 0 and 
                    sess.run(stagnant_count) >= config.max_stagnant_count):
                    print("Stopped early because development accuracy "
                          "didn't increase for %d consecutive epochs." 
                          %config.max_stagnant_count)
                    break
            sess.run(inc_epoch)


def main(_):
    random.seed(29352729)
    np.random.seed(random.randint(0, 10**6))
    tf.set_random_seed(random.randint(0, 10**6))
    config = MyConfig()
    config.save_path = 'output/hdn-large.%s' %version
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        model = HDNModel(config)
    train_hdn_model(model, config)
    

if __name__ == "__main__":
    tf.app.run()
