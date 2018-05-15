'''
This 
'''

from tqdm import tqdm
from version import version
import gzip
import numpy as np
import os
import pickle
import collections
from configs import special_symbols
from utils import count_lines_fast


gigaword_pattern = os.path.join('output', 'gigaword-%s.2018-05-10-9fd479f.txt.gz')
vocab_size = 10**6
min_count = 5


def _build_vocab(filename):
    counter = collections.Counter()
    num_lines = count_lines_fast(filename)
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        lines = tqdm(f, desc='Building vocabulary', unit='line',
                     miniters=100000, total=num_lines)
        for line in lines:
            words = line.strip().split()
            counter.update(words)
    print('Total unique words: %d' %len(counter))
    for sym in special_symbols: assert sym not in counter
    words = special_symbols + [w for w, c in counter.most_common(vocab_size) 
                               if c >= min_count] 
    print('Retained %d words' %len(words))
    word2id = dict((words[i], i) for i in range(len(words)))
    return word2id, words


def build_vocab(inp_path, index_path):
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f: 
            word2id = pickle.load(f)
        print('Vocabulary read from %s' %index_path)
    else:
        word2id, _ = _build_vocab(inp_path)
        with open(index_path, 'wb') as f: 
            pickle.dump(word2id, f)
        print("Vocabulary written to %s" %index_path)
    return word2id


def vectorize(inp_path, out_path, word2id, name='noname'):
    if os.path.exists(out_path):
        print('Found result at %s. Skipped.' %out_path)
    else:
        sents = []
        num_lines = count_lines_fast(inp_path)
        with gzip.open(inp_path, 'rt', encoding='utf-8') as f:
            lines = tqdm(f, unit='line', total=num_lines, miniters=100000,
                         desc='Vectorizing "%s"' %name)
            for line in lines:
                words = line.strip().split()
                words = [word2id.get(w) or word2id['<unkn>'] for w in words]
                sents.append(words)
        total_len = sum(len(s) for s in sents)
        buf = np.empty(total_len, dtype=np.int32)
        sent_index = np.empty((len(sents), 2), dtype=np.int32)
        start = 0
        for i, sent in enumerate(sents):
            end = start+len(sent)
            buf[start:end] = sent
            sent_index[i] = (start, end)
            start = end
        assert end == total_len
        np.savez(out_path, buffer=buf, sent_index=sent_index)
        print('Result saved to %s' %out_path)
        

def run():
    index_path = os.path.join('output', 'vocab.%s.pkl' %version)
    out_pattern = os.path.join('output', 'gigaword-%%s.%s.npz' %version)

    word2id = build_vocab(gigaword_pattern %'train', index_path)
    for ds in ('train', 'dev'):
        vectorize(gigaword_pattern %ds, out_pattern %ds, word2id, name=ds)


if __name__ == '__main__':
    run()
    