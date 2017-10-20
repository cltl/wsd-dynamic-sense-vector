import argparse
import os
from lxml import html
from datetime import datetime

import mapping_utils
import wn_utils
from collections import defaultdict
import pandas

parser = argparse.ArgumentParser(description='Map sensekey annotations to higher levels')
parser.add_argument('-i', dest='input_folder', required=True, help='path to WSD_Training_Corpora (http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip)')
parser.add_argument('-c', dest='corpora', required=True, help='supported: semcor | mun | semcor_mun')
parser.add_argument('-l', dest='abstraction_level', required=True, help='supported: sensekey')
parser.add_argument('-d', dest='competition_df', required=True, help='dataframe of the competition used')
parser.add_argument('-p', dest='accepted_pos', required=True, help='supported: NOUN')
parser.add_argument('-w', dest='wn_version', required=True, help='supported: 30')
parser.add_argument('-o', dest='output_folder', required=True, help='path where output wsd will be stored')
args = parser.parse_args()

"""
python sense_annotations2lstm_format.py -i ../data/WSD_Training_Corpora -c semcor -l sensekey -d sem2013-aw.p -p NOUN -w 30 -o higher_level_annotations
"""

# postprocess arguments
corpora_to_include = args.corpora.split('_')

# abstraction level
assert args.abstraction_level in {'sensekey'}, 'abstraction level %s is not supported (only: sensekey)' % args.abstraction_level

# pos
accepted_pos = set(args.accepted_pos.split('_'))
pos_mapping = defaultdict(str)
pos_mapping['NOUN'] = 'n'

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
log_path = os.path.join(args.output_folder, 'sensekey-' + '_'.join(corpora_to_include) + '.log')
stats_path = os.path.join(args.output_folder, 'sensekey-' + '_'.join(corpora_to_include) + '.stats')


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


## load competition df and add column to it mapping to different semantic level
df = pandas.read_pickle(args.competition_df)
df_output_path = os.path.join(args.output_folder, 'sensekey-' + '_'.join(corpora_to_include) + '.bin')

column_name = 'synset2%s' % args.abstraction_level
df[column_name] = [None for index, row in df.iterrows()]

target_lemmas = set()
target_lemmas_pos = set()

for index, row in df.iterrows():
    sy2sensekeys = wn_utils.get_synset2sensekeys(wn, row['target_lemma'], row['pos'])
    df.set_value(index, column_name, sy2sensekeys)
    target_lemmas.add(row['target_lemma'])
    target_lemmas_pos.add((row['target_lemma'], row['pos']))

df.to_pickle(df_output_path)


# load html
print('started loading html', datetime.now())
my_html_tree = html.parse(input_xml_path)
print('finished loading html', datetime.now())
outfile = open(output_path, 'w')
stats = defaultdict(int)


# label propagation info
sc_lemma_pos2label_sent_index = defaultdict(list)
omsti_lemma_pos2label_sent_index = defaultdict(list)


for corpus_node in my_html_tree.xpath('body/corpus'):

    # decide on whether to include corpus
    the_corpus = corpus_node.get('source')
    if the_corpus in corpora_to_include:

        # loop through sentences
        for sent_number, sent_node in enumerate(corpus_node.xpath('text/sentence')):

            if sent_number % 10000 == 0:
                print(the_corpus, sent_number, datetime.now())

            sentence_tokens = []
            sentence_lemmas = []
            sentence_pos = []
            annotations = []

            for child_el in sent_node.getchildren():



                lemma = child_el.get('lemma')
                token = child_el.text
                pos = child_el.get('pos')
                mapped_pos = pos_mapping[pos]

                assert lemma is not None
                assert token is not None

                sentence_tokens.append(token)
                sentence_lemmas.append(lemma)
                sentence_pos.append(mapped_pos)

                sent_annotations = []

                if all([child_el.tag == 'instance',
                        pos in accepted_pos]):


                    instance_id = child_el.get('id')

                    mapped_sensekey = the_mapping_function(instance_id, the_mapping)

                    for a_mapping in mapped_sensekey:
                        sent_annotations.append(a_mapping)

                        stats[lemma] += 1

                annotations.append(sent_annotations)


            for sentence_info in wn_utils.generate_training_instances_v2(sentence_tokens,
                                                                         sentence_lemmas,
                                                                         sentence_pos,
                                                                         annotations):
                target_lemma, \
                target_pos, \
                token_annotation, \
                sentence_tokens, \
                training_example, \
                target_index = sentence_info

                if target_lemma in target_lemmas:
                    outfile.write(training_example + '\n')


                # add to label propagation dicts
                if the_corpus == 'semcor':
                    sc_lemma_pos2label_sent_index[(target_lemma, target_pos)].append((token_annotation, sentence_tokens, target_index))
                elif the_corpus == 'mun':
                    omsti_lemma_pos2label_sent_index[(target_lemma, target_pos)].append((-1, sentence_tokens, target_index))

outfile.close()

# postprocess for stats
with open(stats_path, 'w') as outfile:
    outfile.write('number of unique lemma: %s\n' % len(stats))

    avg_num_instances = sum(stats.values()) / len(stats)
    outfile.write('avg number of training examples per lemma: %s\n' % round(avg_num_instances, 2))

# postprocess for label propagation input
lp_input = dict()

with open(log_path, 'w') as outfile:

    headers = ['lemma', 'pos', '#_semcor', '#_omsti']
    outfile.write('\t'.join(headers) + '\n')

    for target_lemma, target_pos in target_lemmas_pos:

        sc_info = sc_lemma_pos2label_sent_index[(target_lemma, target_pos)]
        omsti_info = omsti_lemma_pos2label_sent_index[(target_lemma, target_pos)]


        outfile.write('\t'.join([target_lemma,
                                 target_pos,
                                 str(len(sc_info)),
                                 str(len(omsti_info))]) + '\n')

# add test instances to it




# save one dictionary for input label propagation






















