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

def lookup_and_iter_sents(filename, word2id):
    unkn_id = word2id['<unkn>']
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            words = line.strip().split()
            yield [word2id.get(word) or unkn_id for word in words]
            
def pad(sents, max_len, pad_id):
    arr = np.empty((len(sents), max_len), dtype=np.int32)
    arr.fill(pad_id)
    for i, s in enumerate(sents):
        arr[i, :len(s)] = s
    return arr

def pad_batches(inp_path, word2id, dev_sent_ids):
    sys.stderr.write('Dividing and padding...\n')
    pad_id = word2id['<pad>']
    dev = []
    batches = {}
    sent_lens = []
    curr_max_len = 0
    curr_batch = []
    batch_id = 0
    for sent_id, sent in enumerate(progress(lookup_and_iter_sents(inp_path, word2id))):
        if sent_id in dev_sent_ids:
            dev.append(sent)
        else:
            new_size = (len(curr_batch)+1) * max(curr_max_len,len(sent))
            if new_size > batch_size:
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
    dev_lens = np.array([len(s) for s in dev], dtype=np.int32)
    dev_padded = pad(dev, max(dev_lens), pad_id)
    sys.stderr.write('Dividing and padding... Done.\n')
    sizes = np.array([batches['batch%d'%i].size for i in range(batch_id)])
    if batch_id >= 2:
        sys.stderr.write('Divided into %d batches (%d elements each, std=%d, '
                         'except last batch of %d).\n'
                         %(batch_id, sizes[:-1].mean(), sizes[:-1].std(), sizes[-1]))
    else:
        assert batch_id == 1
        sys.stderr.write('Created 1 batch of %d elements.\n' %sizes[0])
    sys.stderr.write('Development set contains %d sentences\n' %len(dev_lens))
    sys.stderr.write('Sentence lengths: %.5f (std=%.5f)\n' 
                     %(sent_lens.mean(), sent_lens.std()))
    sys.stderr.write('Sentence lengths in development set: %.5f (std=%.5f)\n' 
                     %(dev_lens.mean(), dev_lens.std()))
    return batches, dev_padded, dev_lens


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
    
    sys.stderr.write('Calculating batch shapes... ')
    indices = list(range(len(lens)))
    rng = Random(29)
    rng.shuffle(indices)
    total_sents = len(lens)
    batches = {}
    dev_lens = []
    curr_max_len = 0
    curr_batch_lens = []
    sent2batch = {}
    batch_id = 0
    for sent_id in progress(indices, label='sentences'):
        l = lens[sent_id]
        if sent_id in dev_sent_ids:
            dev_lens.append(l)
            sent2batch[sent_id] = 'dev'
        else:
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
    dev_lens = np.array(dev_lens, dtype=np.int32)
    dev_data = np.empty((len(dev_lens), max(dev_lens)), dtype=np.int32)
    sys.stderr.write('Done.\n')
    
    sys.stderr.write('Dividing and padding...\n')
    pad_id = word2id['<pad>']
    for i in range(batch_id): batches['batch%d'%i].fill(pad_id)
    dev_data.fill(pad_id)
    nonpad_count = 0
    sent_counter = Counter()
    for sent_id, sent in progress(enumerate(lookup_and_iter_sents(inp_path, word2id)), label='sentences'):
        assert lens[sent_id] == len(sent)
        batch_name = sent2batch[sent_id]
        arr = dev_data if batch_name == 'dev' else batches[batch_name]
        arr[sent_counter[batch_name],:len(sent)] = sent
        if batch_name != 'dev': nonpad_count += len(sent)
        sent_counter[batch_name] += 1
    # check that we filled all arrays
    for batch_name in sent_counter:
        arr = dev_data if batch_name == 'dev' else batches[batch_name]
        assert sent_counter[batch_name] == arr.shape[0]
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
    sys.stderr.write('Development set contains %d sentences\n' %len(dev_lens))
    sys.stderr.write('Sentence lengths: %.5f (std=%.5f)\n' 
                     %(lens.mean(), lens.std()))
    sys.stderr.write('Sentence lengths in development set: %.5f (std=%.5f)\n' 
                     %(dev_lens.mean(), dev_lens.std()))
    return batches, dev_data, dev_lens

def run(inp_path, out_path, shuffle=True):
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
    
    train_path = out_path + '.train.npz'
    dev_path = out_path + '.dev.npz'
    shuffled_train_path = out_path + '-shuffled.train.npz'
    shuffled_dev_path = out_path + '-shuffled.dev.npz'
    if os.path.exists(dev_path):
        sys.stderr.write('Result already exists: %s. Skipped.\n' %dev_path)
    else:
        total_sents = count_lines_fast(sorted_sents_path)
        real_num_dev_sents = int(min(dev_sents, dev_portion*total_sents))
        np.random.seed(918)
        dev_sent_ids = set(np.random.choice(total_sents, size=real_num_dev_sents, replace=False))
        
        batches, dev_data, dev_lens = pad_batches(sorted_sents_path, word2id, dev_sent_ids)
        np.savez(train_path, **batches)
        np.savez(dev_path, data=dev_data, lens=dev_lens)
        
        if shuffle:
            batches, dev_data, dev_lens = shuffle_and_pad_batches(sorted_sents_path, word2id, dev_sent_ids)
            np.savez(shuffled_train_path, **batches)
            np.savez(shuffled_dev_path, data=dev_data, lens=dev_lens)
    
def copy_lines(num_lines, src_path, dest_path):
    with open(src_path, 'rb') as f_src, open(dest_path, 'wb') as f_dest:
        for _ in range(num_lines):
            f_dest.write(f_src.readline())

if __name__ == '__main__':
    inp_path = preprocessed_gigaword_path
#     inp_path = 'preprocessed-data/gigaword_1m-sents.txt' # for debugging    
    out_dir = os.path.join('preprocessed-data', version)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gigaword-for-lstm-wsd')
    run(inp_path, out_path, shuffle=True)
    
    total_lines = count_lines_fast(inp_path)
    for percent in (1, 10, 25, 50, 75):
        num_lines = int(percent / 100.0 * total_lines)
        
        inp_path_pc = os.path.join(out_dir, 'gigaword_%02d-pc.txt' %percent)
        out_path_pc = os.path.join(out_dir, 'gigaword-for-lstm-wsd_%02d-pc' %percent)
        
        copy_lines(num_lines, inp_path, inp_path_pc)
        run(inp_path_pc, out_path_pc, shuffle=False)