import argparse
import os
from lxml import html
from datetime import datetime

import mapping_utils
import wn_utils
from collections import defaultdict

parser = argparse.ArgumentParser(description='Map sensekey annotations to higher levels')
parser.add_argument('-i', dest='input_folder', required=True, help='path to WSD_Training_Corpora (http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip)')
parser.add_argument('-c', dest='corpora', required=True, help='supported: semcor | mun | semcor_mun')
parser.add_argument('-l', dest='abstraction_level', required=True, help='supported: sensekey')
parser.add_argument('-p', dest='accepted_pos', required=True, help='supported: NOUN')
parser.add_argument('-w', dest='wn_version', required=True, help='supported: 30')
parser.add_argument('-o', dest='output_folder', required=True, help='path where output wsd will be stored')
args = parser.parse_args()


# postprocess arguments
corpora_to_include = args.corpora.split('_')

# abstraction level
assert args.abstraction_level in {'sensekey'}, 'abstraction level %s is not supported (only: sensekey)' % args.abstraction_level

# pos
accepted_pos = set(args.accepted_pos.split('_'))

# input path
if 'mun' in corpora_to_include:
    input_xml_path = os.path.join(args.input_folder, 'SemCor+OMSTI/semcor+omsti.data.xml')
    input_mapping_path = os.path.join(args.input_folder, 'SemCor+OMSTI/semcor+omsti.gold.key.txt')
else:
    input_xml_path = os.path.join(args.input_folder, 'SemCor/semcor.data.xml')
    input_mapping_path = os.path.join(args.input_folder, 'SemCor/semcor.gold.key.txt')

assert os.path.exists(input_xml_path)
assert os.path.exists(input_mapping_path)


# wn version
assert args.wn_version in {'30'}, 'wordnet version: %s is not supported' % args.wn_version

if args.wn_version == '30':
    from nltk.corpus import wordnet as wn
    path_to_wn_dict_folder = str(wn._get_root())  # change this for other wn versions
    path_to_wn_index_sense = os.path.join(path_to_wn_dict_folder, 'index.sense')  # change this for other wn versions

# output path
output_path = os.path.join(args.output_folder, 'sensekey-' + '_'.join(corpora_to_include) + '.txt')
stats_path = output_path + '.stats'

corpora_to_include = set(corpora_to_include)

# load mapping dictionary and function
if args.abstraction_level == 'sensekey':

    # load instance_id to offset(s) from .key.txt file
    sensekey2offset = mapping_utils.load_mapping_sensekey2offset(path_to_wn_index_sense,
                                                                 args.wn_version)

    instance_id2offset, instance_id2sensekeys = mapping_utils.load_instance_id2offset(input_mapping_path,
                                                                                      sensekey2offset,
                                                                                      debug=False)

    the_mapping = instance_id2sensekeys
    the_mapping_function = mapping_utils.map_sensekey_to_sensekey


# load html
my_html_tree = html.parse(input_xml_path)
#outfile = open(output_path, 'w')
stats = defaultdict(int)


for corpus_node in my_html_tree.xpath('body/corpus'):

    # decide on whether to include corpus
    the_corpus = corpus_node.get('source')
    if the_corpus in corpora_to_include:

        # loop through sentences
        for sent_node in corpus_node.xpath('text/sentence'):

            sentence_tokens = []
            annotations = []

            for child_el in sent_node.getchildren():

                if child_el.sourceline % 10000 == 0:
                    print(child_el.sourceline, datetime.now())


                lemma = child_el.get('lemma')
                token = child_el.text
                pos = child_el.get('pos')

                assert lemma is not None
                assert token is not None

                sentence_tokens.append(token)
                sent_annotations = []

                if all([child_el.tag == 'instance',
                        pos in accepted_pos]):

                    instance_id = child_el.get('id')

                    mapped_sensekey = the_mapping_function(instance_id, the_mapping)

                    for a_mapping in mapped_sensekey:
                        sent_annotations.append(a_mapping)

                        stats[lemma] += 1

                annotations.append(sent_annotations)


            #for a_sentence in wn_utils.generate_training_instances(sentence_tokens,
            #                                                       annotations):
            #    outfile.write(a_sentence + '\n')

#outfile.close()

with open(stats_path, 'w') as outfile:
    outfile.write('number of unique lemma: %s\n' % len(stats))

    avg_num_instances = sum(stats.values()) / len(stats)
    outfile.write('avg number of training examples per lemma: %s\n' % round(avg_num_instances, 2))





















