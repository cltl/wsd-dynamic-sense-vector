from nltk.corpus import wordnet as wn
from collections import defaultdict
from tqdm import tqdm
import random
from itertools import islice
from version import version

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

if __name__ == '__main__':
    gigaword_path = 'preprocessed-data/gigaword.txt'
    output_path = 'output/gigaword-hdn-training.%s.txt' %version
    
    id2synset = {synset2identifier(s, '30'): s for s in wn.all_synsets()}
    all_noun_lemmas = {lemma.name()
                          for synset in wn.all_synsets('n')
                          for lemma in synset.lemmas()}
    
    all_hdns = defaultdict(set)
    hdn_list2lemma = defaultdict(list)
    for lemma in all_noun_lemmas:
        graph_info = synsets_graph_info(wn_instance=wn,
                                    wn_version='30',
                                    lemma=lemma,
                                    pos='n')
        hdns = tuple([info['under_lcs'] 
                     for synset, info in graph_info.items() 
                     if info['under_lcs']])
        hdn_list2lemma[hdns].append(lemma)
        for hdn in hdns:
            all_hdns[hdn].add(hdns)
    for hdn in all_hdns:
        all_hdns[hdn] = list(all_hdns[hdn])
    
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

    # # Find all monsemous words in Gigaword
    found_monosemous = []
    with open(gigaword_path) as f:
        for line in tqdm(f, total=175771829):
            for word in line.split():
                if word in monosemous_noun2related_hdns:
                    found_monosemous.append(word)
    print('Found monosemous words: ', len(found_monosemous))
    
    # # Turn Gigaword into a supervised training set
    r = random.Random(5328952)
    used_hdn_lists = set()
    available_hdn_lists = set()
    with open(gigaword_path) as f, open(output_path, 'w') as f_out:
        for line in tqdm(f, total=175771829):
            sent = line.split()
            for i, word in enumerate(sent):
                if word in monosemous_noun2related_hdns:
                    new_sent = sent[:]
                    new_sent[i] = '<target>'
                    for hdn in monosemous_noun2related_hdns[word]:
                        available_hdn_lists.update(all_hdns[hdn])
                    for _ in range(2):
                        hdn = r.choice(monosemous_noun2related_hdns[word])
                        hdn_list = r.choice(all_hdns[hdn])
                        used_hdn_lists.add(hdn_list)
                        f_out.write(' '.join((hdn, '/'.join(hdn_list), ' '.join(new_sent))))
                        f_out.write('\n')
    print('Used HDNs:', len(used_hdn_lists))
    print('HDNs availble in GigaWord:', len(available_hdn_lists))
    all_hdn_lists = set(hdn_list 
                        for hdn_lists in all_hdns.values()
                        for hdn_list in hdn_lists)
    print('HDNs in WordNet:', len(available_hdn_lists))
    
    l = all_hdn_lists.difference(used_hdn_lists)
    print("Sample HDN lists that don't occur in our dataset")
    for hdn_list in random.sample(l, 10):
        print(','.join(hdn_list2lemma[hdn_list]), '-->', 
              '/'.join(id2synset[h].name() for h in hdn_list))    
