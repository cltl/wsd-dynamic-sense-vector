from nltk.corpus import wordnet as wn
from tqdm import tqdm
from collections import defaultdict
from evaluate.wn_utils import synsets_graph_info


all_noun_lemmas = {lemma.name()
                   for synset in wn.all_synsets('n')
                   for lemma in synset.lemmas()}


def extract_all_hdns():
    '''
    Highest-disambiguating nodes (HDNs) are defined as synsets right below
    the lowest common subsumer of the synsets of a noun lemma that needs to
    be disambiguated. This method extract all such unique nodes, associated with
    HDN lists (the competing HDNs for a particular lemma).
    
    @return: dict (HDN -> tuple(tuple(HDN)))
    '''
    all_hdns = defaultdict(set)
    for lemma in tqdm(all_noun_lemmas, unit='lemma', miniters=10000, 
                      desc='Looking for HDNs'):
        graph_info = synsets_graph_info(wn_instance=wn,
                                    wn_version='30',
                                    lemma=lemma,
                                    pos='n')
        hdn_list = tuple(sorted(info['under_lcs'] # sorted to avoid arbitrary order
                            for synset, info in graph_info.items() 
                            if info['under_lcs']))
        for hdn in hdn_list:
            all_hdns[hdn].add(hdn_list)
    for hdn in all_hdns:
        all_hdns[hdn] = tuple(all_hdns[hdn])
    return all_hdns
