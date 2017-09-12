'''
Created on 13 Jun 2017

Read a simple text file (one sentence per line) and produce these files:

- <fname>.index.pkl: vocabulary as a dictionary (word -> index)
- <fname>.train.npz: training batches (each batch contains roughly the same
number of tokens but differing number of sentences depends on sentence length)
- <fname>.dev.npz: development dataset (as big as one epoch)

@author: Minh Le
'''
import codecs
import sys
import collections
import os

from time import time
import pickle
import re
import numpy as np
import subprocess
from tensorflow.contrib.labeled_tensor import batch
from random import Random
from collections import Counter
from utils import progress
from configs import preprocessed_gigaword_path, preprocessed_data_dir

dev_sents = 20000 # absolute maximum
dev_portion = 0.01 # relative maximum
batch_size = 64000 # words
vocab_size = 10**6
min_count = 5

special_symbols = ['<target>', '<unkn>', '<pad>']

rng = Random(2958207520)


def _build_vocab(filename):
    sys.stderr.write('Building vocabulary...\n')
    counter = collections.Counter()
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in progress(f):
            words = line.strip().split()
            counter.update(words)
    sys.stderr.write('Total unique words: %d\n' %len(counter))
    for sym in special_symbols: assert sym not in counter
    words = special_symbols + [w for w, c in counter.most_common(vocab_size) 
                               if c >= min_count] 
    sys.stderr.write('Retained %d words\n' %len(words))
    word2id = dict((words[i], i) for i in range(len(words)))
    sys.stderr.write('Building vocabulary... Done.\n')
    return word2id, words

def sort_sentences(inp_path, out_path):
    cmd = ('cat %s | python3 scripts/sentlen.py --min 6 --max 100 '
           '| sort -T output -k1,1g -k2 | uniq > %s'
           %(inp_path, out_path))
    sys.stderr.write('%s\n' %cmd)
    status = subprocess.call(cmd, shell=True)
    assert status == 0

def lookup_and_iter_sents(filename, word_to_id):
    unkn_id = word2id['<unkn>']
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            words = line.strip().split()
            yield [word_to_id.get(word) or unkn_id for word in words]
            
class PadFunc(object):
    
    dry_run=False
    
    def __init__(self):
        self.total = 0
        self.pads = 0
    def __call__(self, sents, max_len, pad_id):
        if self.dry_run:
            arr = np.empty(0)
            value_count = sum(1 for s in sents for _ in s)
            size = len(sents) * max_len
        else:
            arr = np.zeros((len(sents), max_len), dtype=np.int32)
            size = arr.size
            arr.fill(pad_id)
            value_count = 0
            for i, s in enumerate(sents):
                for j, v in enumerate(s):
                    arr[i,j] = v
                    value_count += 1
        self.pads += (size - value_count) 
        self.total += size
        return arr

def pad_batches(inp_path, word2id):
    sys.stderr.write('Dividing and padding...\n')
    pad = PadFunc()
    pad_id = word2id['<pad>']
    dev = []
    batches = {}
    last_max_len = 0
    last_batch = []
    with open(inp_path) as f: total_sents = sum(1 for line in f)
    for sent in progress(lookup_and_iter_sents(inp_path, word2id)):
        if (len(dev) < dev_sents and len(dev) < dev_portion*total_sents 
                and rng.random() < 0.01):
            dev.append(sent)
        else:
            new_size = (len(last_batch)+1) * max(last_max_len,len(sent))
            if new_size > batch_size:
                batches['batch%d' %len(batches)] = pad(last_batch, last_max_len, pad_id)
                last_max_len = 0
                last_batch = []
            last_max_len = max(last_max_len, len(sent))
            last_batch.append(sent)
    if last_batch:
        batches['batch%d' %len(batches)] = pad(last_batch, last_max_len, pad_id)
    dev_lens = np.array([len(s) for s in dev], dtype=np.int32)
    dev_padded = PadFunc()(dev, max(dev_lens), pad_id)
    sys.stderr.write('Dividing and padding... Done.\n')
    sizes = np.array([b.size for b in batches.values()])
    if len(batches) >= 2:
        sys.stderr.write('Divided into %d batches (%d elements each, std=%d, '
                         'except last batch of %d).\n'
                         %(len(batches), sizes[:-1].mean(), sizes[:-1].std(), sizes[-1]))
    else:
        assert len(batches) == 1
        sys.stderr.write('Created 1 batch of %d elements.\n' %sizes[0])
    sys.stderr.write('Added %d elements as padding (%.2f%%).\n' 
                     %(pad.pads, pad.pads*100.0/pad.total))
    sys.stderr.write('Consumed roughly %.2f GiB.\n' 
                     %(pad.total*4/float(2**30)))
    return batches, dev_padded, dev_lens



def shuffle_and_pad_batches(inp_path, word2id):
    sys.stderr.write('Reading lengths...\n')
    lens = []
    with codecs.open(inp_path, 'r', 'utf-8') as f:
        for line in progress(f):
            lens.append(line.count(' ') + 1)
    sys.stderr.write('Reading lengths... Done.\n')
    
    sys.stderr.write('Calculating batch shapes... ')
    indices = list(range(len(lens)))
    rng.shuffle(indices)
    total_sents = len(lens)
    batches = {}
    dev_lens = []
    last_max_len = 0
    last_batch = []
    sent2batch = {}
    for sent_id in indices:
        l = lens[sent_id]
        if (len(dev_lens) < dev_sents and len(dev_lens) < dev_portion*total_sents 
                and rng.random() < 0.01):
            dev_lens.append(l)
            sent2batch[sent_id] = 'dev'
        else:
            new_size = (len(last_batch)+1) * max(last_max_len,l)
            if new_size >= batch_size:
                batches['batch%d' %len(batches)] = np.empty((len(last_batch), last_max_len))
                last_max_len = 0
                last_batch = []
            last_max_len = max(last_max_len, l)
            last_batch.append(l)
            sent2batch[sent_id] = 'batch%d' %len(batches)
    if last_batch:
        batches['batch%d' %len(batches)] = np.empty((len(last_batch), last_max_len))
    dev = np.empty((len(dev_lens), max(dev_lens)))
    sys.stderr.write('Done.\n')
    
    sys.stderr.write('Dividing and padding...\n')
    pad_id = word2id['<pad>']
    for b in batches.values(): b.fill(pad_id)
    dev.fill(pad_id)
    nonpad_count = 0
    counter = Counter()
    for sent_id, sent in progress(enumerate(lookup_and_iter_sents(inp_path, word2id))):
        batch_name = sent2batch[sent_id]
        arr = dev if batch_name == 'dev' else batches[batch_name]
        arr[counter[batch_name],:len(sent)] = sent
        if batch_name != 'dev': nonpad_count += len(sent)
        counter[batch_name] += 1
    assert counter['dev'] == dev.shape[0]
    for batch_name, b in batches.items():
        assert counter[batch_name] == b.shape[0]
    sys.stderr.write('Dividing and padding... Done.\n')
    
    sizes = np.array([b.size for b in batches.values()])
    if len(batches) >= 2:
        sys.stderr.write('Divided into %d batches (%d elements each, std=%d, '
                         'except last batch of %d).\n'
                         %(len(batches), sizes[:-1].mean(), sizes[:-1].std(), sizes[-1]))
    else:
        assert len(batches) == 1
        sys.stderr.write('Created 1 batch of %d elements.\n' %sizes[0])
    total = sum(sizes)
    pad_count = total - nonpad_count
    sys.stderr.write('Added %d elements as padding (%.2f%%).\n' 
                     %(pad_count, pad_count*100.0/total))
    sys.stderr.write('Consumed roughly %.2f GiB.\n' 
                     %(total*4/float(2**30)))
    return batches, dev, dev_lens


if __name__ == '__main__':
    inp_path = preprocessed_gigaword_path
    out_path = os.path.join(preprocessed_data_dir, 'gigaword-for-lstm-wsd')
    
    index_path = out_path + '.index.pkl'
    if os.path.exists(index_path):
        sys.stderr.write('Reading vocabulary from %s... ' %index_path)
        with open(index_path, 'rb') as f: word2id = pickle.load(f)
        sys.stderr.write('Done.\n')
    else:
        assert os.path.isfile(inp_path)
        word2id, words = _build_vocab(inp_path)
        with open(index_path, 'wb') as f: pickle.dump(word2id, f)

    sorted_sents_path = inp_path + '.sorted'
    if os.path.exists(sorted_sents_path):
        sys.stderr.write('Sentences are already sorted at %s\n' %sorted_sents_path)
    else:
        sort_sentences(inp_path, sorted_sents_path)
    
    train_path = out_path + '.train.npz'
    dev_path = out_path + '.dev.npz'
    shuffled_train_path = out_path + '.train-shuffled.npz'
    shuffled_dev_path = out_path + '.dev-shuffled.npz'
    if os.path.exists(train_path):
        sys.stderr.write('Result already exists: %s. Skipped.\n' %train_path)
    else:
        batches, dev_data, dev_lens = pad_batches(sorted_sents_path, word2id)
        np.savez(train_path, **batches)
        np.savez(dev_path, data=dev_data, lens=dev_lens)
        
        batches, dev_data, dev_lens = shuffle_and_pad_batches(sorted_sents_path, word2id)
        np.savez(shuffled_train_path, **batches)
        np.savez(shuffled_dev_path, data=dev_data, lens=dev_lens)
