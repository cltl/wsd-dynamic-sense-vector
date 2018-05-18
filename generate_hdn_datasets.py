'''
Generate train and development sets that contain HDN examples. This is a 
'''

from nltk.corpus import wordnet as wn
from collections import defaultdict
from tqdm import tqdm
import random
from version import version
import os
from hdn import extract_all_hdns, all_noun_lemmas
import pandas as pd
from evaluate.wn_utils import synset2identifier
import numpy as np
from random import sample
import sys
import re
import pickle


inp_pattern = 'output/gigaword-%s.2018-05-10-7d764e7.npz'
word_vocab_path = 'output/vocab.2018-05-10-7d764e7.pkl'


def monosemous_noun2related_hdns(all_hdns):
    '''
    Returns a map (monosemous word -> list(HDN subsumers)).
    '''
    sys.stdout.write('Looking for monosemous nouns... ')
    lemmas_without_hdn = set()
    mono2hdn = defaultdict(list)
    blacklist = ['while', 'why', 'might'] # list of ambiguous words
    for lemma in all_noun_lemmas.difference(blacklist):
        synsets = wn.synsets(lemma=lemma) # take all parts-of-speech
        if (len(synsets) == 1 and # monosemous
            not synsets[0].instance_hypernyms() and  # ignore proper names
            not re.match(r'[A-Z][a-z]*', lemma)): # ignore proper names 
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


def extract_evidences(mono2hdn, all_hdns):
    '''
    A monosemous lemma provides evidence for the disambiguation of certain 
    lists of candidates
    e.g. lemma W of synset S is under hypernym A within list (A, B, C)
    ==> every occurence of W provides a training example for A against B and C
    If a lemma happens to be under two hypernyms of the same list, we don't 
    know which one to choose so we can't use it as a training example. This is 
    called a conflict. Conflicts are excluded from the evidence map.
    
    @return: map(lemma:str -> list(<evidence>) in which <evidence> is a map contains
    two keys: 'hdn' is the correct HDN to predict, 'candidates' are all 
    competing HDNs for the given lemma.
    '''
    num_conflicts = 0
    mono2evidence = defaultdict(list)
    for mono, hdns in tqdm(mono2hdn.items(), desc="Extracting evidences"):
        evidences = defaultdict(set)
        for hdn in hdns:
            for hdn_list in all_hdns[hdn]:
                evidences[hdn_list].add(hdn)
        for key, val in evidences.items():
            if len(val) == 1: # there's no conflict
                mono2evidence[mono].append({'hdn': list(val)[0], 'candidates': key})
            elif len(val) >= 2:
                num_conflicts += 1
    num_evidences = sum(len(v) for v in mono2evidence.values())
    print('Found %d evidences, ignored %d conflicts' %(num_evidences, num_conflicts))
    return mono2evidence
    

def convert_gigaword(inp_path, out_path, id2word, mono2evidence,
                     hdn2id, hdn_list2id, name='noname', random_seed=None):
    if os.path.exists(out_path):
        print('Transformed dataset "%s" found at %s' %(name, out_path))
        return

    r = random.Random(random_seed)
    arrs = np.load(inp_path)    
    buffer = arrs['buffer']
    sent_index_tqdm = tqdm(arrs['sent_index'], 
                           desc='Extracting monosemous words from "%s"' %name)
    mono_in_sent = [{'sent_start': start, 'sent_stop': stop, 'word': w} 
                    for start, stop in sent_index_tqdm
                    for w, word_id in enumerate(buffer[start:stop])
                    if id2word[word_id] in mono2evidence]
    hdn_examples = []
    for ex in tqdm(mono_in_sent, desc='Extracting examples from "%s"' %name):
        word_as_id = buffer[ex['sent_start'] + ex['word']]
        if mono2evidence[id2word[word_as_id]]:
            for evidence in r.sample(mono2evidence[id2word[word_as_id]], 1):
                candidates_as_id = hdn_list2id[evidence['candidates']]
                hdn_as_id = hdn2id[evidence['hdn']]
                hdn_examples.append({'sent_start': ex['sent_start'], 
                                     'sent_stop': ex['sent_stop'], 
                                     'sent_len': ex['sent_stop'] - ex['sent_start'], 
                                     'word_index': ex['word'],
                                     'hdn': hdn_as_id,
                                     'candidates': candidates_as_id})
    hdn_examples = pd.DataFrame(hdn_examples)
    hdn_examples.to_pickle(out_path)
    print('Transformed dataset "%s" written to %s' %(name, out_path))


def build_hdn_vocab(hdns):
    hdn_vocab_path = os.path.join('output', 'hdn-vocab.%s.pkl' %version)
    hdn_list_vocab_path = os.path.join('output', 'hdn-list-vocab.%s.pkl' %version)
    
    hdn2id = {hdn: i for i, hdn in enumerate(sorted(hdns))}
    with open(hdn_vocab_path, 'wb') as f:
        pickle.dump(hdn2id, f)
    print('HDN vocab written to %s' %hdn_vocab_path)
        
    hdn_lists = sorted(set(hdn_list 
                       for hdn_lists in hdns.values() 
                       for hdn_list in hdn_lists))
    hdn_list2id = {hdn_list: i for i, hdn_list in enumerate(hdn_lists)}
    with open(hdn_list_vocab_path, 'wb') as f:
        pickle.dump(hdn_list2id, f)
    print('HDN list vocab written to %s' %hdn_list_vocab_path)
    
    return hdn2id, hdn_list2id


def run():
    out_pattern = os.path.join('output', 'gigaword-hdn-%%s.%s.pkl' %version)
    
    all_hdns = extract_all_hdns()
    hdn2id, hdn_list2id = build_hdn_vocab(all_hdns)
    mono2hdn = monosemous_noun2related_hdns(all_hdns)
    mono2evidence = extract_evidences(mono2hdn, all_hdns)
    word2id = np.load(word_vocab_path)
    id2word = {i: w for w, i in word2id.items()}
    
    for ds, seed in [('train', 203840), ('dev', 39420528323)]:
        convert_gigaword(inp_pattern %ds, out_pattern %ds, id2word,
                         mono2evidence, hdn2id, hdn_list2id,
                         name=ds, random_seed=seed)


if __name__ == '__main__':
    run()