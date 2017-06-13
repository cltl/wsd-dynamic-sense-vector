'''
Created on 13 Jun 2017

Read a simple text file (one sentence per line) and produce these files:

- <fname>.idx: vocabulary as a dictionary(word -> index)
- <fname>.train.npz: training batches (each batch contains roughly the same
number of tokens but differing number of sentences depends on sentence length)
- <fname>.dev.pkl: development dataset (10000 sentences)

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

batch_size = 10**6
max_words = 10**6
min_count = 5

def progress(it):
    start = time()
    for i, val in enumerate(it):
        yield(val)
        if (i+1) % 1000000 == 0:
            sys.stderr.write('processed %d items, elapsed time: %.1f minutes...\n' 
                             %(i+1, (time()-start)/60))

def _build_vocab(filename):
    sys.stderr.write('Building vocabulary...\n')
    counter = collections.Counter()
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in progress(f):
            words = line.strip().split()
            counter.update(words)
    sys.stderr.write('Total unique words: %d\n' %len(counter))
    words = [w for w, c in counter.most_common(max_words) if c >= min_count] 
    sys.stderr.write('Retained %d words\n' %len(words))
    word2id = dict((words[i], i) for i in range(len(words)))
    sys.stderr.write('Building vocabulary... Done.\n')
    return word2id, words

def _file_to_sents(filename, word_to_id):
    sys.stderr.write('Reading sentences and converting words to indices...\n')
    unkn_id = add_special_symbol(word2id, '<unkn>')
    sents = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in progress(f):
            words = line.strip().split()
            sents.append([word_to_id.get(word) or unkn_id for word in words])
    sys.stderr.write('Reading sentences and converting words to indices... Done.\n')
    return sents

class PadFunc(object):
    def __init__(self):
        self.total = 0
        self.pads = 0
    def __call__(self, sents, max_len, pad_id):
        for s in sents:
            while len(s) < max_len:
                s.append(pad_id)
                self.pads += 1
        self.total += len(sents)*max_len
        return np.array(sents)
pad = PadFunc()

def pad_batches(sents):
    sys.stderr.write('Dividing and padding...\n')
    add_special_symbol(word2id, '<target>')
    pad_id = add_special_symbol(word2id, '<pad>')
    splitting_point = len(sents) - 10000
    train, dev = sents[:splitting_point], sents[splitting_point:]
    train.sort(key=lambda s: len(s))
    batches = {}
    last_max_len = 0
    last_batch = []
    for sents in progress(train):
        last_max_len = max(last_max_len, len(sents))
        last_batch.append(sents)
        if len(last_batch)*last_max_len >= batch_size:
            batches['batch%d' %len(batches)] = pad(last_batch, last_max_len, pad_id)
            last_max_len = 0
            last_batch = []
    if last_max_len > 0:
        batches['batch%d' %len(batches)] = pad(last_batch, last_max_len, pad_id)
    sys.stderr.write('Dividing and padding... Done.\n')
    sizes = np.array([b.size for b in batches.values()])
    sys.stderr.write('Divided into %d batches (%d elements each, std=%d, '
                     'except last batch of %d)... Done.\n'
                     %(len(batches), sizes[:-1].mean(), sizes[:-1].std(), sizes[-1]))
    sys.stderr.write('Added %d elements as padding (%.2f%%).\n' 
                     %(pad.pads, pad.pads*100.0/pad.total))
    return batches, dev

def add_special_symbol(word2id, sym):
    assert sym not in word2id
    word2id[sym] = len(word2id)
    return word2id[sym]

if __name__ == '__main__':
    inp_path = sys.argv[1]
    assert os.path.isfile(inp_path)
    assert re.search(r'\.txt$', inp_path)
    
    index_path = re.sub('.txt$', '.index.pkl', inp_path)
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f: word2id = pickle.load(f)
        sys.stderr.write('Read vocabulary from %s.\n' %index_path)
    else:
        word2id, words = _build_vocab(inp_path)
        with open(index_path, 'wb') as f: pickle.dump(word2id, f)

    sents_path = re.sub('.txt$', '.sents.pkl', inp_path)
    if os.path.exists(sents_path):
        with open(sents_path, 'rb') as f: sents = pickle.load(f)
        sys.stderr.write('Read sentences from %s.\n' %sents_path)
    else:
        sents = _file_to_sents(inp_path, word2id)
        with open(sents_path, 'wb') as f: pickle.dump(sents, f)
    
    train_path = re.sub('.txt$', '.train.npz', inp_path)
    dev_path = re.sub('.txt$', '.dev.pkl', inp_path)
    batches, dev = pad_batches(sents)
    np.savez(train_path, **batches)
    with open(dev_path, 'wb') as f: pickle.dump(dev, f)
