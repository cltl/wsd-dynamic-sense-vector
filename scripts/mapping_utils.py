#from semantic_class_manager import BLC
from WordNetMapper import WordNetMapper



def map_sensekey_to_sensekey(instance_id, mapping):
    return mapping[instance_id]

def map_instance_id2synset(instance_id, mapping):
    return {mapping[instance_id]}

def map_instance_id2direct_hypernym(synset_id, mapping):
    return {mapping[synset_id]}

def map_instance_id2blc20(synset_id, mapping):
    return {mapping[synset_id]}


def get_synset2hypernym(wn):
    """
    """
    synset2hypernym = dict()
    for synset in wn.all_synsets(pos='n'):

        synset_id = synset2identifier(synset, '30')
        hypernym_id = None

        hypernyms = synset.hypernyms()
        if hypernyms:
            hypernym_id = synset2identifier(hypernyms[0], '30')

        synset2hypernym[synset_id] = hypernym_id

    return synset2hypernym

def get_synset2blc20(wn):
    """
    
    :param wn: 
    :return: 
    """
    blc_20_obj = BLC(20, 'all')
    synset2blc20 = dict()

    for synset in wn.all_synsets(pos='n'):
        synset_id = synset2identifier(synset, '30')

        offset = str(synset.offset()).zfill(8)
        results = blc_20_obj.get_classes_for_synset_pos(offset, 'n', '30')

        synset2blc20[synset_id] = results[0]

    return synset2blc20

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

def get_lemma_pos_of_sensekey(sense_key):
    """
    lemma and pos are determined for a wordnet sense key

    >>> get_lemma_pos_of_sensekey('life%1:09:00::')
    ('life', 'n')

    :param str sense_key: wordnet sense key

    :rtype: tuple
    :return: (lemma, n | v | r | a | u)
    """
    if '%' not in sense_key:
        return '', 'u'

    lemma, information = sense_key.split('%')
    int_pos = information[0]

    if int_pos == '1':
        this_pos = 'n'
    elif int_pos == '2':
        this_pos = 'v'
    elif int_pos in {'3', '5'}:
        this_pos = 'a'
    elif int_pos == '4':
        this_pos = 'r'
    else:
        this_pos = 'u'

    return lemma, this_pos


def load_mapping_sensekey2offset(path_to_index_sense, wn_version):
    """
    extract mapping sensekey2offset from index.sense file

    :param str path_to_index_sense: path to wordnet index.sense file
    :param str wn_version: 171 | 21 | 30

    :rtype: dict
    :return: mapping sensekey (str) -> eng-WN_VERSION-OFFSET-POS    
    """
    sensekey2offset = dict()

    with open(path_to_index_sense) as infile:
        for line in infile:
            sensekey, synset_offset, sense_number, tag_cnt = line.strip().split()

            lemma, pos = get_lemma_pos_of_sensekey(sensekey)

            assert pos != 'u'
            assert len(synset_offset) == 8

            identifier = 'eng-{wn_version}-{synset_offset}-{pos}'.format_map(locals())

            sensekey2offset[sensekey] = identifier

    return sensekey2offset


def load_instance_id2offset(mapping_path, sensekey2offset, target_wn_version, debug=False):
    """
    load mapping between instance_id -> wordnet offset

    :param str mapping_path: path to mapping instance_id -> sensekeys
    :param dict sensekey2offset: see output function load_mapping_sensekey2offset
    :param str target_wn_version: version to which sensekeys should be mapped

    :rtype: dict
    :return: instance_id -> offset
    """
    instance_id2offset = dict()
    instance_id2sensekeys = dict()

    more_than_one_offset = 0
    no_offsets = 0

    if target_wn_version != '30':
        my_mapper = WordNetMapper()


    with open(mapping_path) as infile:
        for line in infile:
            instance_id, *sensekeys = line.strip().split()

            if target_wn_version != '30':
                new_sensekeys = []
                for sensekey in sensekeys:
                    try:
                        mapped_key = my_mapper.map_lexkey_to_lexkey(sensekey, '30', target_wn_version)
                    except ValueError:
                        continue

                    new_sensekeys.append(mapped_key)

                sensekeys = []
                sensekeys.extend(new_sensekeys)
    
            instance_id2sensekeys[instance_id] = sensekeys

            offsets = {sensekey2offset[sensekey]
                       for sensekey in sensekeys
                       if sensekey in sensekey2offset}

            if len(offsets) == 1:
                instance_id2offset[instance_id] = offsets.pop()

            elif len(offsets) >= 2:
                more_than_one_offset += 1
                #print('2> offsets available for %s: %s' % (instance_id, offsets))

                # we just take one of the n possible offsets
                instance_id2offset[instance_id] = offsets.pop()

            elif len(offsets) == 0:
                #print('no offsets available for %s' % instance_id)
                no_offsets += 1


    return instance_id2offset, instance_id2sensekeys
