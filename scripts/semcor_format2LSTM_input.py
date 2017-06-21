
# TODO run for semcor + omsti on computer science server
# TODO log some extra information

import os
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn
from lxml import html, etree
import wn_utils


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


def load_instance_id2offset(mapping_path, sensekey2offset, debug=False):
    """
    load mapping between instance_id -> wordnet offset

    :param str mapping_path: path to mapping instance_id -> sensekeys
    :param dict sensekey2offset: see output function load_mapping_sensekey2offset

    :rtype: dict
    :return: instance_id -> offset
    """
    instance_id2offset = dict()

    more_than_one_offset = 0
    no_offsets = 0

    with open(mapping_path) as infile:
        for line in infile:
            instance_id, *sensekeys = line.strip().split()

            offsets = {sensekey2offset[sensekey]
                       for sensekey in sensekeys
                       if sensekey in sensekey2offset}

            if len(offsets) == 1:
                instance_id2offset[instance_id] = offsets.pop()

            elif len(offsets) >= 2:
                more_than_one_offset += 1

                # we just take one of the n possible offsets
                instance_id2offset[instance_id] = offsets.pop()

            elif len(offsets) == 0:
                no_offsets += 1

    if debug:
        print('more than one offset', more_than_one_offset)
        print('no offset', no_offsets)

    return instance_id2offset


# experiment settings
wn_version = '30'
corpora_to_include = ['semcor', 'mun']  # semcor | mun
accepted_pos = {'NOUN'}
entailment_setting = 'any_hdn'  # lemma_hdn | any_hdn

if wn_version == '30':
    path_to_wn_dict_folder = str(wn._get_root) # change this for other wn versions
    path_to_wn_index_sense = os.path.join(path_to_wn_dict_folder, 'index.sense') # change this for other wn versions


if corpora_to_include == ['semcor', 'mun']:
    input_xml_path = '../data/WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml'
    input_mapping_path = '../data/WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'
elif corpora_to_include == ['semcor']:
    input_xml_path = '../data/WSD_Training_Corpora/SemCor/semcor.data.xml'
    input_mapping_path = '../data/WSD_Training_Corpora/SemCor/semcor.gold.key.txt'

synset_output_path = 'synset-' + '_'.join(corpora_to_include) + '.txt'
hdn_output_path = '-'.join(['hdn',
                            '_'.join(corpora_to_include),
                            '_'.join(accepted_pos),
                            entailment_setting]) + '.txt'


# precompute all hdns
lemma_pos2offsets = wn_utils.load_lemma_pos2offsets(path_to_wn_index_sense)
lemma_pos2graph_info = dict()
all_hdns = set()
for (lemma, pos), offsets in lemma_pos2offsets.items():
    if all([len(offsets) >= 2,
            pos == 'n']):
        graph_info = wn_utils.synsets_graph_info(wn_instance=wn,
                                                 wn_version='30',
                                                 lemma=lemma,
                                                 pos=pos)
        for sy_id, info in graph_info.items():
            hdn = info['under_lcs']

            if hdn is not None:
                all_hdns.add(hdn)

# precompute synset identifier -> path to root
sy_id2hypernyms = dict()
for synset in wn.all_synsets(pos='n'):
    synset_id = wn_utils.synset2identifier(synset, '30')
    hypernyms = {wn_utils.synset2identifier(hypernym, '30')
                 for hypernym_path in synset.hypernym_paths()
                 for hypernym in hypernym_path}

    sy_id2hypernyms[synset_id] = hypernyms



# load wn
my_wn_reader = WordNetCorpusReader(path_to_wn_dict_folder, None)

# load instance_id to offset(s) from .key.txt file
sensekey2offset = load_mapping_sensekey2offset(path_to_wn_index_sense,
                                               wn_version)

instance_id2offset = load_instance_id2offset(input_mapping_path,
                                             sensekey2offset,
                                             debug=False)

my_html_tree = html.parse(input_xml_path)

hdn_outfile = open(hdn_output_path, 'w')
synset_outfile = open(synset_output_path, 'w')

for corpus_node in my_html_tree.xpath('body/corpus'):

    # decide on whether to include corpus
    the_corpus = corpus_node.get('source')
    if the_corpus in corpora_to_include:

        # loop through sentences
        for sent_node in corpus_node.xpath('text/sentence'):

            sentence_lemmas = []
            synset_annotations = []
            hdn_annotations = []

            for child_el in sent_node.getchildren():

                lemma = child_el.get('lemma')
                pos = child_el.get('pos')

                assert lemma is not None

                sentence_lemmas.append(lemma)
                sent_synset_annotations = []
                sent_hdn_annotations = []

                if all([child_el.tag == 'instance',
                        pos in accepted_pos]):

                    instance_id = child_el.get('id')
                    synset_id = instance_id2offset[instance_id]
                    sent_synset_annotations.append(synset_id)

                    # option lemma-based hdn
                    if entailment_setting == 'lemma_hdn':
                        pos = synset_id[-1]
                        graph_info = wn_utils.synsets_graph_info(wn_instance=my_wn_reader,
                                                                 wn_version=wn_version,
                                                                 lemma=lemma,
                                                                 pos=pos)

                        if synset_id in graph_info:
                            hdn = graph_info[synset_id]['under_lcs']

                            if hdn is not None:
                                sent_hdn_annotations.append(hdn)

                    elif entailment_setting == 'any_hdn':
                        hypernyms = sy_id2hypernyms[synset_id]
                        for hypernym in hypernyms:
                            if hypernym in all_hdns:
                                sent_hdn_annotations.append(hypernym)

                synset_annotations.append(sent_synset_annotations)
                hdn_annotations.append(sent_hdn_annotations)

            for synset_sentence in wn_utils.generate_training_instances(sentence_lemmas,
                                                                        synset_annotations):
                synset_outfile.write(synset_sentence + '\n')

            for hdn_sentence in wn_utils.generate_training_instances(sentence_lemmas,
                                                                     hdn_annotations):
                hdn_outfile.write(hdn_sentence + '\n')

hdn_outfile.close()
synset_outfile.close()

