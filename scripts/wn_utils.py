import nltk

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
        target_sy_iden = synset2identifier(synsets.pop(), wn_version)
        sy_id2under_lcs_info[target_sy_iden] = {'under_lcs': None,
                                                'path_to_under_lcs': []}
        return sy_id2under_lcs_info


    for sy1 in synsets:

        target_sy_iden = synset2identifier(sy1, wn_version)

        min_path_distance = 100
        closest_lcs = None

        for sy2 in synsets:
            if sy1 != sy2:
                try:
                    lcs_s = sy1.lowest_common_hypernyms(sy2)
                    lcs = lcs_s[0]
                except:
                    lcs = None
                    print('wordnet error', sy1, sy2)

                path_distance = sy1.shortest_path_distance(lcs)

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
                                                            'path_to_under_lcs': path_to_under_lcs_idens}

    return sy_id2under_lcs_info
