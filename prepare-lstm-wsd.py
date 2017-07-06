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

batch_size = 128000 # words
vocab_size = 10**6
min_count = 5

special_symbols = ['<target>', '<unkn>', '<pad>']

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
            arr = np.zeros((len(sents), max_len), dtype=np.int64)
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
    dev_size = 0
    batches = []
    last_max_len = 0
    last_batch = []
    for sent in progress(lookup_and_iter_sents(inp_path, word2id)):
        if dev_size < batch_size and np.random.rand() < 0.01:
            dev.append(sent)
            dev_size += len(sent)
        else:
            last_max_len = max(last_max_len, len(sent))
            last_batch.append(sent)
            if len(last_batch)*last_max_len >= batch_size:
                batches.append(pad(last_batch, last_max_len, pad_id))
                last_max_len = 0
                last_batch = []
    if last_max_len > 0:
        batches.append(pad(last_batch, last_max_len, pad_id))
    dev_lens = np.array([len(s) for s in dev], dtype=np.int64)
    dev_padded = PadFunc()(dev, max(dev_lens), pad_id)
    sys.stderr.write('Dividing and padding... Done.\n')
    sizes = np.array([b.size for b in batches])
    if len(batches) >= 2:
        sys.stderr.write('Divided into %d batches (%d elements each, std=%d, '
                         'except last batch of %d).\n'
                         %(len(batches), sizes[:-1].mean(), sizes[:-1].std(), sizes[-1]))
    else:
        assert len(batches) == 1
        sys.stderr.write('Created 1 batch of %d elements.\n' %sizes[0])
    sys.stderr.write('Added %d elements as padding (%.2f%%).\n' 
                     %(pad.pads, pad.pads*100.0/pad.total))
    return batches, dev_padded, dev_lens

def reduce_vocab(batches, target_id, full_vocab_size):
    train_sents = []
    train_vocabs = []
    train_targets = []
    for i, sentences in enumerate(batches):
        batch_vocab, inverse = np.unique(sentences, return_inverse=True)
        sentences = inverse.reshape(sentences.shape)
        batch_vocab = np.append(batch_vocab, target_id)
        local_target_id = batch_vocab.size-1
        sys.stderr.write('Batch #%d vocab: %d (%.2f%%)\n'
                         %(i, batch_vocab.size, batch_vocab.size*100.0/full_vocab_size))
        train_sents.append(sentences)
        train_vocabs.append(batch_vocab)
        train_targets.append(local_target_id)
    return train_sents, train_vocabs, train_targets

def serialize(train_data_path, train_index_path, 
              train_sents, train_vocabs, train_targets):
    assert len(train_sents) == len(train_vocabs) == len(train_targets)
    sys.stderr.write('Flattening... ')
    total_size = sum(s.size for s in train_sents) + sum(v.size for v in train_vocabs)
    data = np.memmap(train_data_path, dtype=np.int64, mode='w+', shape=(total_size,))
    last_index = 0
    indices = np.zeros((len(train_sents), 6), dtype=np.int64)
    for i in range(len(train_sents)):
        indices[i,0] = last_index
        indices[i,1], indices[i,2] = train_sents[i].shape
        data[last_index:last_index+train_sents[i].size] = np.ravel(train_sents[i])
        last_index += train_sents[i].size
        
        indices[i,3] = last_index
        indices[i,4], = train_vocabs[i].shape
        data[last_index:last_index+train_vocabs[i].size] = train_vocabs[i]
        last_index += train_vocabs[i].size
        
        indices[i,5] = train_targets[i]
    assert last_index == data.size
    assert np.all(indices >= 0) # check for integer overflow
    total_size = data.size*8 + indices.size*8
    sys.stderr.write('Done.\n')
    sys.stderr.write('Training data consumes roughly %.2f GiB.\n' 
                     %(total_size/float(2**30)))
    del data # write to file
    np.save(train_index_path, indices)

if __name__ == '__main__':
    inp_path, out_path = sys.argv[1:]
    assert os.path.isfile(inp_path)
    
    index_path = out_path + '.index.pkl'
    if os.path.exists(index_path):
        sys.stderr.write('Reading vocabulary from %s... ' %index_path)
        with open(index_path, 'rb') as f: word2id = pickle.load(f)
        sys.stderr.write('Done.\n')
    else:
        word2id, words = _build_vocab(inp_path)
        with open(index_path, 'wb') as f: pickle.dump(word2id, f)

    sorted_sents_path = inp_path + '.sorted'
    if os.path.exists(sorted_sents_path):
        sys.stderr.write('Sentences are already sorted at %s\n' %sorted_sents_path)
    else:
        sort_sentences(inp_path, sorted_sents_path)
    
    train_data_path = out_path + '.train.data.npy'
    train_index_path = out_path + '.train.index.npy'
    dev_path = out_path + '.dev.npz'
    if os.path.exists(train_data_path):
        sys.stderr.write('Result already exists: %s. Skipped.\n' %train_data_path)
    else:
        batches, dev_data, dev_lens = pad_batches(sorted_sents_path, word2id)
        np.savez(dev_path, data=dev_data, lens=dev_lens)
        serialize(train_data_path, train_index_path,
                  *reduce_vocab(batches, word2id['<target>'], len(word2id)))
