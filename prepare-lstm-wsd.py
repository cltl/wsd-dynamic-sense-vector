'''
Created on 13 Jun 2017

Read a simple text file (one sentence per line) and produce these files:

- <fname>.index.pkl: vocabulary as a dictionary (word -> index)
- <fname>.train.npz: training batches (each batch contains roughly the same
number of tokens but differing number of sentences depends on sentence length)
- <fname>.dev.npz: development dataset (as big as one epoch)
- 

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
from random import Random
from collections import Counter
from utils import progress, count_lines_fast
from configs import preprocessed_gigaword_path, output_dir
from version import version

dev_sents = 20000 # absolute maximum
dev_portion = 0.01 # relative maximum
# if you get OOM (out of memory) error, reduce this number
batch_size = 60000 # words
vocab_size = 10**6
min_count = 5

inp_path = preprocessed_gigaword_path
# inp_path = 'preprocessed-data/gigaword_1m-sents.txt' # for debugging    
out_dir = os.path.join('preprocessed-data', version)
out_path = os.path.join(out_dir, 'gigaword-for-lstm-wsd')

special_symbols = ['<target>', '<unkn>', '<pad>']

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
    start = time()
    cmd = ('cat %s | python3 scripts/sentlen.py --min 6 --max 100 '
           '| sort -T %s -k1,1g -k2 | uniq > %s'
           %(inp_path, output_dir, out_path))
    sys.stderr.write('%s\n' %cmd)
    status = subprocess.call(cmd, shell=True)
    sys.stderr.write('sorting finished after %.1f minutes...\n' %((time()-start)/60))
    assert status == 0

def lookup_and_iter_sents(filename, word2id, include_ids=None, exclude_ids=None):
    unkn_id = word2id['<unkn>']
    with codecs.open(filename, 'r', 'utf-8') as f:
        for sent_id, line in enumerate(f):
            if ((include_ids is None or sent_id in include_ids) and 
                (exclude_ids is None or sent_id not in exclude_ids)):
                words = line.strip().split()
                yield [word2id.get(word) or unkn_id for word in words]
            
def pad(sents, max_len, pad_id):
    arr = np.empty((len(sents), max_len), dtype=np.int32)
    arr.fill(pad_id)
    for i, s in enumerate(sents):
        arr[i, :len(s)] = s
    return arr

def pad_batches(inp_path, word2id, include_ids, exclude_ids, max_sents=-1):
    sys.stderr.write('Dividing and padding...\n')
    pad_id = word2id['<pad>']
    batches = {}
    sent_lens = []
    curr_max_len = 0
    curr_batch = []
    batch_id = 0
    for sent in progress(lookup_and_iter_sents(inp_path, word2id,
                                               include_ids, exclude_ids)):
        new_size = (len(curr_batch)+1) * max(curr_max_len,len(sent))
        if new_size > batch_size or (max_sents > 0 and len(curr_batch) >= max_sents):
            batches['batch%d' %batch_id] = pad(curr_batch, curr_max_len, pad_id)
            batches['lens%d' %batch_id] = np.array([len(s) for s in curr_batch], dtype=np.int32)
            batch_id += 1
            curr_max_len = 0
            curr_batch = []
        curr_max_len = max(curr_max_len, len(sent))
        curr_batch.append(sent)
        sent_lens.append(len(sent))
    if curr_batch:
        batches['batch%d' %batch_id] = pad(curr_batch, curr_max_len, pad_id)
        batches['lens%d' %batch_id] = np.array([len(s) for s in curr_batch], dtype=np.int32)
        batch_id += 1 # important to count num batches correctly
    sent_lens = np.array(sent_lens, dtype=np.int32)
    sys.stderr.write('Dividing and padding... Done.\n')
    sizes = np.array([batches['batch%d'%i].size for i in range(batch_id)])
    if batch_id >= 2:
        sys.stderr.write('Divided into %d batches (%d elements each, std=%d, '
                         'except last batch of %d).\n'
                         %(batch_id, sizes[:-1].mean(), sizes[:-1].std(), sizes[-1]))
    else:
        assert batch_id == 1
        sys.stderr.write('Created 1 batch of %d elements.\n' %sizes[0])
    sys.stderr.write('Sentence lengths: %.5f (std=%.5f)\n' 
                     %(sent_lens.mean(), sent_lens.std()))
    return batches


def shuffle_and_pad_batches(inp_path, word2id, dev_sent_ids):
    sys.stderr.write('Reading lengths...\n')
    lens = []
    with codecs.open(inp_path, 'r', 'utf-8') as f:
        for line in progress(f, label='sentences'):
            # this is different from counting the blank spaces because some words
            # are separated by double spaces and there might be an additional
            # whitespace at the end of a line
            lens.append(len(line.strip().split()))
    lens = np.array(lens, dtype=np.int32)
    sys.stderr.write('Reading lengths... Done.\n')
    
    sys.stderr.write('Calculating batch shapes...\n')
    indices = list(range(len(lens)))
    rng = Random(29)
    rng.shuffle(indices)
    total_sents = len(lens)
    batches = {}
    curr_max_len = 0
    curr_batch_lens = []
    sent2batch = {}
    batch_id = 0
    for sent_id in progress(indices, label='sentences'):
        l = lens[sent_id]
        if sent_id not in dev_sent_ids:
            new_size = (len(curr_batch_lens)+1) * max(curr_max_len,l)
            if new_size >= batch_size:
                batches['batch%d' %batch_id] = \
                        np.empty((len(curr_batch_lens), max(curr_batch_lens)), dtype=np.int32)
                batches['lens%d' %batch_id] = np.array(curr_batch_lens, dtype=np.int32)
                batch_id += 1
                curr_max_len = 0
                curr_batch_lens = []
            curr_max_len = max(curr_max_len, l)
            curr_batch_lens.append(l)
            sent2batch[sent_id] = 'batch%d' %batch_id
    if curr_batch_lens:
        batches['batch%d' %batch_id] = \
                np.empty((len(curr_batch_lens), max(curr_batch_lens)), dtype=np.int32)
        batches['lens%d' %batch_id] = np.array(curr_batch_lens, dtype=np.int32)
        batch_id += 1 # important to count num batches correctly
    sys.stderr.write('Calculating batch shapes... Done.\n')
    
    sys.stderr.write('Dividing and padding...\n')
    pad_id = word2id['<pad>']
    for i in range(batch_id): batches['batch%d'%i].fill(pad_id)
    nonpad_count = 0
    sent_counter = Counter()
    for sent_id, sent in progress(enumerate(lookup_and_iter_sents(inp_path, word2id)), label='sentences'):
        assert lens[sent_id] == len(sent)
        batch_name = sent2batch.get(sent_id)
        if batch_name is not None: # could be in dev set
            batches[batch_name][sent_counter[batch_name],:len(sent)] = sent
            nonpad_count += len(sent)
            sent_counter[batch_name] += 1
    # check that we filled all arrays
    for batch_name in sent_counter:
        assert sent_counter[batch_name] == batches[batch_name].shape[0]
    sys.stderr.write('Dividing and padding... Done.\n')
    
    sizes = np.array([batches['batch%d'%i].size for i in range(batch_id)])
    if batch_id >= 2:
        sys.stderr.write('Divided into %d batches (%d elements each, std=%d, '
                         'except last batch of %d).\n'
                         %(batch_id, sizes[:-1].mean(), sizes[:-1].std(), sizes[-1]))
    else:
        assert batch_id == 1
        sys.stderr.write('Created 1 batch of %d elements.\n' %sizes[0])
    total = sum(sizes)
    pad_count = total - nonpad_count
    sys.stderr.write('Sentence lengths: %.5f (std=%.5f)\n' 
                     %(lens.mean(), lens.std()))
    return batches

def run():
    os.makedirs(out_dir, exist_ok=True)
    index_path = out_path + '.index.pkl'
    if os.path.exists(index_path):
        sys.stderr.write('Reading vocabulary from %s... ' %index_path)
        with open(index_path, 'rb') as f: word2id = pickle.load(f)
        sys.stderr.write('Done.\n')
    else:
        assert os.path.isfile(inp_path)
        word2id, words = _build_vocab(inp_path)
        with open(index_path, 'wb') as f: pickle.dump(word2id, f)

    sorted_sents_path = out_path + '.sorted'
    if os.path.exists(sorted_sents_path):
        sys.stderr.write('Sentences are already sorted at %s\n' %sorted_sents_path)
    else:
        sort_sentences(inp_path, sorted_sents_path)
        
    total_sents = count_lines_fast(sorted_sents_path)
    real_num_dev_sents = int(min(dev_sents, dev_portion*total_sents))
    np.random.seed(918)
    dev_sent_ids = set(np.random.choice(total_sents, size=real_num_dev_sents, replace=False))
    
    train_path = out_path + '.train.npz'
    dev_path = out_path + '.dev.npz'
    shuffled_train_path = out_path + '-shuffled.train.npz'
    if os.path.exists(shuffled_train_path):
        sys.stderr.write('Result already exists: %s. Skipped.\n' %shuffled_train_path)
    else:
        print("- Training set:")
        batches = pad_batches(sorted_sents_path, word2id, None, dev_sent_ids)
        np.savez(train_path, **batches)
        print("- Development set:")
        batches = pad_batches(sorted_sents_path, word2id, dev_sent_ids, None, 768)
        np.savez(dev_path, **batches)
        print("- Shuffled training set:")
        batches = shuffle_and_pad_batches(sorted_sents_path, word2id, dev_sent_ids)
        np.savez(shuffled_train_path, **batches)
            
    for percent in (1, 10, 25, 50, 75):
        num_lines = int(percent / 100.0 * total_sents)
        sampled_ids = set(np.random.choice(total_sents, size=num_lines, replace=False))
        pc_train_path = out_path + ('_%02d-pc.train.npz' %percent)
        if os.path.exists(pc_train_path):
            sys.stderr.write('%02d%% dataset already exists: %s. Skipped.\n' %pc_train_path)
        else:
            print("- Reduced training set (%02d%%):" %percent)
            batches = pad_batches(sorted_sents_path, word2id, sampled_ids, dev_sent_ids)
            np.savez(pc_train_path, **batches)

if __name__ == '__main__':
    run()
