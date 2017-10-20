import nltk
import itertools
from collections import defaultdict


def get_synset2domain(path_wn20_to_domain,
                      path_wn20_to_wn30):
    """
    create mapping between wn30 and domain and vice versa

    :param str path_wn20_to_domain: wn-domains-3.2-20070223 file
    :param str path_wn20_to_wn30: wn20-30.noun file from upc mappings

    :rtype: tuple
    :return: (wn30_domain, domain_wn30)
    """
    wn30_domain = dict()
    domain_wn30 = defaultdict(set)

    wn20_wn30 = dict()
    with open(path_wn20_to_wn30) as infile:
        for line in infile:
            split = line.strip().split()
            if len(split) == 3:
                offset_20, *values = line.strip().split()
                offset_30 = ''
                conf = 0.0
                for index in range(0, len(values), 2):
                    an_offset = values[index]
                    a_conf = float(values[index + 1])
                    if a_conf > conf:
                        offset_30 = an_offset
                        conf = a_conf
                wn20_wn30[offset_20 + '-n'] = offset_30 + '-n'

    with open(path_wn20_to_domain) as infile:
        for line in infile:
            sy_id, domain = line.strip().split('\t')
            if all([sy_id in wn20_wn30,
                    sy_id.endswith('n')]):
                wn30 = wn20_wn30[sy_id]

                wn30_domain['eng-30-' + wn30] = domain
                domain_wn30[domain].add('eng-30-' + wn30)

    return wn30_domain, domain_wn30


def generate_training_instances(sentence_lemmas, annotations):
    """
    given the lemmas in a sentence with its annotations (can be more than one)
    generate all training instances for that sentence
    
    e.g. 
    sentence_lemmas = ['the', 'man',            'meeting', 'woman']
    annotations =     [[],    ['1', '2' , '3'], ['4'],     ['5', '6']]
    
    would result in
    the man---1 meeting---4 woman---5
    the man---2 meeting woman---6
    the man---3 meeting woman
    
    :param list sentence_lemmas: see above
    :param list annotations: see above
    
    :rtype: set
    :return: set of strings (representing annotated sentences)
    """
    instances = set()
    for one_annotated_sent in itertools.zip_longest(*annotations):
        a_sentence = []
        for index, annotation in enumerate(one_annotated_sent):
            lemma = sentence_lemmas[index]
            if annotation is not None:
                a_sentence.append(lemma + '---' + annotation)
            else:
                a_sentence.append(lemma)

        instances.add(' '.join(a_sentence))
    
    return instances


def generate_training_instances_v2(sentence_tokens,
                                   sentence_lemmas,
                                   sentence_pos,
                                   annotations):
    """
    given the lemmas in a sentence with its annotations (can be more than one)
    generate all training instances for that sentence

    e.g. 
    sentence_tokens = ['the', 'man',            'meets',   'women']
    sentence_lemmas = ['the', 'man',            'meet',    'woman']
    sentence_pos    = ['',    'n',              'v',       'n']
    annotations =     [[],    ['1', '2' ],      ['4'],     ['5', '6']]

    would result in
    ('man', 'n', '1', ['the', 'man', 'meets', 'women'], 'the man---1 meets women', 1)
    ('man', 'n', '2', ['the', 'man', 'meets', 'women'], 'the man---2 meets women', 1)
    ('meet', 'v', '4', ['the', 'man', 'meets', 'women'], 'the man meets---4 women', 2)
    ('woman', 'n', '5', ['the', 'man', 'meets', 'women'], 'the man meets women---5', 3)
    ('woman', 'n', '6', ['the', 'man', 'meets', 'women'], 'the man meets women---6', 3)

    :param list sentence_tokens: see above
    :param list sentence_lemmas: see above
    :param list sentence_pos: see above
    :param list annotations: see above

    :rtype: generator
    :return: generator of (target_lemma, 
                           target_pos, 
                           token_annotation, 
                           sentence_tokens, 
                           training_example, 
                           target_index)
    """
    for target_index, token_annotations in enumerate(annotations):

        target_lemma = sentence_lemmas[target_index]
        target_pos = sentence_pos[target_index]

        for token_annotation in token_annotations:

            a_sentence = []
            for index, token in enumerate(sentence_tokens):

                if index == target_index:
                    a_sentence.append(token + '---' + token_annotation)
                else:
                    a_sentence.append(token)

            training_example = ' '.join(a_sentence)

            yield (target_lemma,
                   target_pos,
                   token_annotation,
                   sentence_tokens,
                   training_example,
                   target_index)

def load_lemma_pos2offsets(path_to_index_sense):
    '''
    given with index.sense from wordnet distributions such as
    casuistical%3:01:01:: 03053657 1 0

    this function returns a dictionary mapping (lemma,pos)
    to the offsets they can refer to.

    :param str path_to_index_sense: path to wordnet index.sense file

    :rtype: collections.defaultdict
    :return: mapping of (lemma,pos) to offsets they can refer to
    '''
    lemmapos2offsets = defaultdict(set)
    with open(path_to_index_sense) as infile:
        for line in infile:
            key, offset, sqr, freq = line.strip().split()
            lemma, info = key.split('%')

            if info.startswith('1'):
                pos = 'n'
            elif info.startswith('2'):
                pos = 'v'
            elif info.startswith('4'):
                pos = 'r'
            else:
                pos = 'a'

            lemmapos2offsets[(lemma, pos)].add(offset)

    return lemmapos2offsets


def levenshtein(s, t):
    ''' 
    source: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

    >>> levenshtein('house', 'home')
    2

    @type  s: str
    @param s: a string (for example 'house')

    @type  t: str
    @param t: a string (for example ('home')

    @rtype: int
    @param: levenshtein distance
    '''

    if s == t:
        return 0

    elif len(s) == 0:
        return len(t)
    elif len(t) == 0:
        return len(s)

    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)

    for i in range(len(v0)):
        v0[i] = i

    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(t)]

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


def get_synset2sensekeys(wn, target_lemma, pos):
    """

    :param str target_lemma: e.g. cat
    :param str pos: n v a r

    :rtype: dict
    :return: mapping from synset identifier -> sensekey

    """
    synset2sensekeys = dict()
    for synset in wn.synsets(target_lemma, pos):
        sy_id = synset2identifier(synset, '30')
        for lemma in synset.lemmas():
            if lemma.key().startswith(target_lemma + '%'):
                synset2sensekeys[sy_id] = lemma.key()

    return synset2sensekeys
