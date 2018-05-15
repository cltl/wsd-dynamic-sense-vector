from nltk.corpus import wordnet as wn
from collections import defaultdict
from tqdm import tqdm
import random
from version import version
import gzip
import os
from hdn import extract_all_hdns, all_noun_lemmas
from evaluate.wn_utils import synset2identifier
import vectorize_gigaword
import numpy as np
from random import sample
from utils import count_lines_fast
import sys


inp_pattern = vectorize_gigaword.gigaword_pattern


def monosemous_noun2related_hdns(all_hdns):
    sys.stdout.write('Looking for monosemous nouns... ')
    lemmas_without_hdn = set()
    mono2hdn = defaultdict(list)
    for lemma in all_noun_lemmas:
        synsets = wn.synsets(lemma=lemma, pos='n')
        if len(synsets) == 1: # monosemous
            for s in synsets:
                paths = s.hypernym_paths()
                for h in paths[0]:
                    id_ = synset2identifier(h, '30')
                    if id_ in all_hdns:
                        mono2hdn[lemma].append(id_)
            if not mono2hdn[lemma]:
                lemmas_without_hdn.add(lemma)
    sys.stdout.write('Done.\n')
    
    print('All noun lemmas:', len(all_noun_lemmas))
    print('Monosemous noun lemmas: %d (%.1f%%)'
          %(len(mono2hdn), len(mono2hdn) / len(all_noun_lemmas) * 100))
    mwes = [m for m in mono2hdn if '_' in m]
    print('Monosemous multi-word-expressions: %d (%.1f%% of monosemous)'
          %(len(mwes), len(mwes) / len(mono2hdn) * 100))
    print('\tSamples: %s' %', '.join(sample(mwes, 5)))
    print('Lemmas that are not associated with any HDN: %s' 
          %', '.join(lemmas_without_hdn))
    num_hdns = [len(hdns) for hdns in mono2hdn.values()]
    print('Number of HDNs per lemma: mean=%.1f, median=%.1f, std=%.1f)'
          %(np.mean(num_hdns), np.median(num_hdns), np.std(num_hdns)))
    
    return mono2hdn


def convert_gigaword(inp_path, out_path, all_hdns, mono2hdn, 
                     name='noname', random_seed=None):
    '''
    Turn Gigaword into a labeled dataset of HDNs by using monosemous nouns.
    The output file contains lines of the following format:
        <TARGET_HDN> <SPACE> <CANDIDATE_HDN_LIST> <SPACE> <SENTENCE>
    where the sentence is a list of words separated by a space and the candidate
    HDNs are separated by a slash.   
    '''
    if os.path.exists(out_path):
        print('Transformed Gigaword found at %s' %out_path)
        return

    r = random.Random(random_seed)
    used_hdn_lists = set()
    available_hdn_lists = set()
    num_examples = 0
    num_lines = count_lines_fast(inp_path)
    tmp_path = out_path + '.tmp'
    with gzip.open(inp_path, 'rt') as f, \
            gzip.open(tmp_path, 'wt') as f_out:
        for line in tqdm(f, unit='sentence', total=num_lines, miniters=10000,
                         desc='Transforming "%s"' %name):
            sent = line.split()
            for i, word in enumerate(sent):
                if mono2hdn.get(word):
                    new_sent = sent[:]
                    new_sent[i] = '<target>'
                    for hdn in mono2hdn[word]:
                        available_hdn_lists.update(all_hdns[hdn])
                    hdn = r.choice(mono2hdn[word])
                    hdn_list = r.choice(all_hdns[hdn])
                    used_hdn_lists.add(hdn_list)
                    f_out.write(' '.join((hdn, '/'.join(hdn_list), ' '.join(new_sent))))
                    f_out.write('\n')
                    num_examples += 1
                    
    os.rename(tmp_path, out_path)
    print('Result is written to %s' %out_path)
    print('Number of examples: %d' %(num_examples))
    print('Number of HDNs used:', len(used_hdn_lists))
    print('Number of HDNs availble in GigaWord:', len(available_hdn_lists))


def run():
    out_pattern = 'output/gigaword-hdn-%%s.%s.txt.gz' %version
    all_hdns = extract_all_hdns()
    mono2hdn = monosemous_noun2related_hdns(all_hdns)
    for ds, seed in [('train', 203840), ('dev', 39420528323)]:
        convert_gigaword(inp_pattern %ds, out_pattern %ds, all_hdns, mono2hdn,
                         name=ds, random_seed=seed)


if __name__ == '__main__':
    run()