from nltk.corpus import wordnet as wn
from collections import defaultdict, Counter
from tqdm import tqdm
import random
from itertools import islice
from version import version
import gzip
import sys
import numpy as np
import os
import pickle
import collections
import codecs
from configs import special_symbols
from sklearn.model_selection._split import train_test_split


no_lines_gigaword = 175771829 # no need to re-count it
dev_sents = 20000 # absolute maximum
dev_portion = 0.01 # relative maximum
train_max_sents = 1.5e8
# if you get OOM (out of memory) error, reduce this number
batch_size = 60000 # words
vocab_size = 10**6
min_count = 5


def synset2identifier(synset, wn_version):
    """
    return synset identifier of
    nltk.corpus.reader.wordnet.Synset instance

    :param nltk.corpus.reader.wordnet.Synset synset: a wordnet synset
    :param str wn_version: supported: '171 | 21 | 30'

    :rtype: str
    :return: eng-VERSION-OFFSET-POS (n | v | r | a)
    e.g.
    """
    offset = str(synset.offset())
    offset_8_char = offset.zfill(8)

    pos = synset.pos()
    if pos == 'j':
        pos = 'a'

    identifier = 'eng-{wn_version}-{offset_8_char}-{pos}'.format_map(locals())

    return identifier

def synsets_graph_info(wn_instance, wn_version, lemma, pos):
    """
    extract:
    1. hyponym under lowest least common subsumer

    :param nltk.corpus.reader.wordnet.WordNetCorpusReader wn_instance: instance
    of nltk.corpus.reader.wordnet.WordNetCorpusReader
    :param str wn_version: supported: '171' | '21' | '30'
    :param str lemma: a lemma
    :param str pos: a pos

    :rtype: dict
    :return: mapping synset_id
        -> 'under_lcs' -> under_lcs identifier
        -> 'path_to_under_lcs' -> [sy1_iden, sy2_iden, sy3_iden, ...]
    """
    sy_id2under_lcs_info = dict()

    synsets = wn_instance.synsets(lemma, pos=pos)

    synsets = set(synsets)

    if len(synsets) == 1:
        sy_obj = synsets.pop()
        target_sy_iden = synset2identifier(sy_obj, wn_version)
        sy_id2under_lcs_info[target_sy_iden] = {'under_lcs': None,
                                                'under_lcs_obj': None,
                                                'sy_obj' : sy_obj,
                                                'path_to_under_lcs': []}
        return sy_id2under_lcs_info


    for sy1 in synsets:

        target_sy_iden = synset2identifier(sy1, wn_version)

        min_path_distance = 100
        closest_lcs = None

        for sy2 in synsets:
            if sy1 != sy2:
                try:
                    lcs_s = sy1.lowest_common_hypernyms(sy2, simulate_root=True)
                    lcs = lcs_s[0]
                except:
                    lcs = None
                    print('wordnet error', sy1, sy2)

                path_distance = sy1.shortest_path_distance(lcs, simulate_root=True)

                if path_distance < min_path_distance:
                    closest_lcs = lcs
                    min_path_distance = path_distance

        under_lcs = None
        for hypernym_path in sy1.hypernym_paths():
            for first, second in  zip(hypernym_path, hypernym_path[1:]):
                if first == closest_lcs:
                    under_lcs = second

                    index_under_lcs = hypernym_path.index(under_lcs)
                    path_to_under_lcs = hypernym_path[index_under_lcs + 1:-1]

                    under_lcs_iden = synset2identifier(under_lcs, wn_version)
                    path_to_under_lcs_idens = [synset2identifier(synset, wn_version)
                                               for synset in path_to_under_lcs]

                    sy_id2under_lcs_info[target_sy_iden] = {'under_lcs': under_lcs_iden,
                                                            'under_lcs_obj': under_lcs,
                                                            'sy_obj' : sy1,
                                                            'path_to_under_lcs': path_to_under_lcs_idens}

    return sy_id2under_lcs_info


def extract_all_hdns():
    all_noun_lemmas = {lemma.name()
                          for synset in wn.all_synsets('n')
                          for lemma in synset.lemmas()}
    all_hdns = defaultdict(set)
    for lemma in tqdm(all_noun_lemmas, desc='looking for HDNs'):
        graph_info = synsets_graph_info(wn_instance=wn,
                                    wn_version='30',
                                    lemma=lemma,
                                    pos='n')
        hdns = tuple(sorted(info['under_lcs'] # sorted to avoid arbitrary order
                            for synset, info in graph_info.items() 
                            if info['under_lcs']))
        for hdn in hdns:
            all_hdns[hdn].add(hdns)
    for hdn in all_hdns:
        all_hdns[hdn] = list(all_hdns[hdn])
    return all_noun_lemmas, all_hdns


def convert_gigaword(gigaword_path, output_path, all_noun_lemmas, all_hdns):
    '''
    Turn Gigaword into a labeled dataset of HDNs by using monosemous nouns.
    The output file contains lines of the following format:
        <TARGET_WORD> <SPACE> <CANDIDATES> <SPACE> <SENTENCE>
    where the sentence is a list of words separate by a space.   
    '''
    if os.path.exists(output_path):
        print('Transformed Gigaword found at %s' %output_path)
        return all_hdns
    
    # # Getting monosemous words
    monosemous_noun2related_hdns = defaultdict(list)
    for lemma in all_noun_lemmas:
        synsets = wn.synsets(lemma=lemma, pos='n')
        if len(synsets) == 1: # monsemous
            for s in synsets:
                paths = s.hypernym_paths()
                for h in paths[0]:
                    id_ = synset2identifier(h, '30')
                    if id_ in all_hdns:
                        monosemous_noun2related_hdns[lemma].append(id_)

    print('Monosemous multi-word-expressions:',
          len([m for m in monosemous_noun2related_hdns if '_' not in m]))
    print('All monosemous words:', len(monosemous_noun2related_hdns))
    print('All lemmas:', len(all_noun_lemmas))
    print('Proportion monosemous/all:', 
          len(monosemous_noun2related_hdns) / len(all_noun_lemmas))

    # # Turn Gigaword into a supervised training set
    r = random.Random(5328952)
    used_hdn_lists = set()
    available_hdn_lists = set()
    num_examples = 0
    with open(gigaword_path) as f, gzip.open(output_path, 'wt') as f_out:
        for line in tqdm(f, total=no_lines_gigaword, desc='transforming Gigaword'):
            sent = line.split()
            for i, word in enumerate(sent):
                if word in monosemous_noun2related_hdns:
                    new_sent = sent[:]
                    new_sent[i] = '<target>'
                    for hdn in monosemous_noun2related_hdns[word]:
                        available_hdn_lists.update(all_hdns[hdn])
                    hdn = r.choice(monosemous_noun2related_hdns[word])
                    hdn_list = r.choice(all_hdns[hdn])
                    used_hdn_lists.add(hdn_list)
                    f_out.write(' '.join((hdn, '/'.join(hdn_list), ' '.join(new_sent))))
                    f_out.write('\n')
                    num_examples += 1
    print('Written %d examples to %s' %(num_examples, output_path))
    print('Used HDNs:', len(used_hdn_lists))
    print('HDNs availble in GigaWord:', len(available_hdn_lists))
    print('HDNs in WordNet:', len(available_hdn_lists))

    return all_hdns


def shuffle_and_pad_batches(inp_path, word2id, hdn2id, hdn_list2id):
    eos_id, pad_id, unkn_id = word2id.get('<eos>'), word2id['<pad>'], word2id['<unkn>']
    lens = []
    with gzip.open(inp_path, 'rt') as f:
        for line in tqdm(f, total=no_lines_gigaword, desc='reading lengths'):
            target, candidates, sent = line.strip().split(maxsplit=2)
            # this is different from counting the blank spaces because some words
            # are separated by double spaces and there might be an additional
            # whitespace at the end of a line
            lens.append(len(sent.split()))
    lens = np.array(lens, dtype=np.int32)
    
    all_indices = list(range(len(lens)))
    actual_dev_size = min(dev_portion, dev_sents/len(lens))
    actual_train_size = min(train_max_sents / len(lens), 1-actual_dev_size)
    train_indices, train_lens, dev_indices, _ = \
            train_test_split(all_indices, lens, test_size=actual_dev_size,
                             train_size=actual_train_size)
    train_lens, train_indices = zip(*sorted(zip(train_lens, train_indices)))
    dev_indices = set(dev_indices)
    
    batches = {}
    curr_max_len = 0
    curr_batch_lens = []
    sent2batch = {}
    batch_id = 0
    for l, sent_id in tqdm(zip(train_lens, train_indices), 
                           desc='calculating batch shapes'):
        if sent_id in dev_indices:
            pass # TODO
        else:
            new_size = (len(curr_batch_lens)+1) * max(curr_max_len,l)
            if new_size >= batch_size:
                max_len = max(curr_batch_lens)
                if eos_id is not None:
                    max_len += 1
                batches['batch%d' %batch_id] = \
                        np.empty((len(curr_batch_lens), max_len), dtype=np.int32)
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
    for i in range(batch_id):
        batches['batch%d'%i].fill(pad_id)
    
    nonpad_count = 0
    sent_counter = Counter()
    with gzip.open(inp_path, 'rt') as f:
        for sent_id, line in tqdm(enumerate(f), 
                                  total=no_lines_gigaword, 
                                  desc='dividing and padding'):
            target, candidates, sent = line.strip().split(maxsplit=2)
            candidates = tuple(candidates.split('/'))
            sent = sent.split()
            
            target = hdn2id[target]
#             candidate_list_id = hdn_list2id[candidates]
            sent = [word2id.get(w, unkn_id) for w in sent]
            
            assert lens[sent_id] == len(sent)
            batch_name = sent2batch.get(sent_id)
            if batch_name is not None: # could be in dev set
                batches[batch_name][sent_counter[batch_name], :len(sent)] = sent
                if eos_id is not None:
                    batches[batch_name][sent_counter[batch_name], len(sent)] = eos_id
                nonpad_count += len(sent)
                sent_counter[batch_name] += 1
    # check that we filled all arrays
    for batch_name in sent_counter:
        assert sent_counter[batch_name] == batches[batch_name].shape[0]
        
    sizes = np.array([batches['batch%d'%i].size for i in range(batch_id)])
    if batch_id >= 2:
        sys.stderr.write('Divided into %d batches (%d elements each, std=%d, '
                         'except last batch of %d).\n'
                         %(batch_id, sizes[:-1].mean(), sizes[:-1].std(), sizes[-1]))
    else:
        assert batch_id == 1
        sys.stderr.write('Created 1 batch of %d elements.\n' %sizes[0])
    sys.stderr.write('Sentence lengths: %.5f (std=%.5f)\n' 
                     %(lens.mean(), lens.std()))
    return batches


if __name__ == '__main__':
    gigaword_path = 'preprocessed-data/gigaword.txt'
    index_path = 'output/hdn-input-vocab.%s.pkl' %version
    converted_gigaword_path = 'output/gigaword-hdn-training.%s.txt.gz' %version
    
    all_noun_lemmas, all_hdns = extract_all_hdns()
    all_hdns = convert_gigaword(gigaword_path, converted_gigaword_path, 
                                all_noun_lemmas, all_hdns)
    all_hdn_lists = list(set(hdn_list 
                             for hdn_lists in all_hdns.values()
                             for hdn_list in hdn_lists))
    hdn_list2id = {hdn_list: i for i, hdn_list in enumerate(all_hdn_lists)}
    hdn2id = {hdn: i for i, hdn in enumerate(all_hdns.keys())}
    
    word2id = build_vocab(gigaword_path, index_path)
    shuffle_and_pad_batches(converted_gigaword_path, word2id, hdn2id, hdn_list2id)

