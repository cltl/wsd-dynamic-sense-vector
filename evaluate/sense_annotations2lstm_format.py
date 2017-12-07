import argparse
import os
from lxml import html
from datetime import datetime
import pickle
#from semantic_class_manager import BLC
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.corpus import wordnet as wn

from spacy.en import English
nlp = English()


import mapping_utils
import wn_utils
from collections import defaultdict
import pandas
from random import sample

parser = argparse.ArgumentParser(description='Map sensekey annotations to higher levels')
parser.add_argument('-i', dest='input_folder', required=True, help='path to WSD_Training_Corpora (http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip)')
parser.add_argument('-c', dest='corpora', required=True, help='supported: semcor | mun | semcor_mun')
parser.add_argument('-l', dest='abstraction_level', required=True, help='supported: sensekey | synset')
parser.add_argument('-d', dest='competition_df', required=True, help='dataframe of the competition used')
parser.add_argument('-p', dest='accepted_pos', required=True, help='supported: NOUN | VERB | ADJ | ADV')
parser.add_argument('-w', dest='wn_version', required=True, help='supported: 171 | 30')
parser.add_argument('-o', dest='output_folder', required=True, help='path where output wsd will be stored')
args = parser.parse_args()

"""
python3 sense_annotations2lstm_format.py -i ../data/WSD_Training_Corpora -c semcor -l sensekey -d sem2013-aw.p -p NOUN -w 30 -o higher_level_annotations
"""

print('start postprocessing command line args', datetime.now())


# postprocess arguments
corpora_to_include = args.corpora.split('_')

# abstraction level
assert args.abstraction_level in {'direct_hypernym', 'sensekey', 'synset', 'blc20'}, 'abstraction level %s is not supported (only: sensekey | synset | direct_hypernym | blc20)' % args.abstraction_level

# pos
accepted_pos = set(args.accepted_pos.split('_'))
pos_mapping = defaultdict(str)
pos_mapping['NOUN'] = 'n'
pos_mapping['VERB'] = 'v'
pos_mapping['ADJ'] = 'a'
pos_mapping['ADV'] = 'r'


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
assert args.wn_version in {'171', '30'}, 'wordnet version: %s is not supported' % args.wn_version

path_to_wn_dict_folder = str(wn._get_root())  # change this for other wn versions
path_to_wn_index_sense = os.path.join(path_to_wn_dict_folder, 'index.sense')  # change this for other wn versions

if args.wn_version == '171':
    cwd = os.path.dirname(os.path.realpath(__file__))
    path_to_wn_dict_folder = os.path.join(cwd, 'resources', 'wordnet_171', 'WordNet-1.7.1', 'dict')
    path_to_wn_index_sense = os.path.join(path_to_wn_dict_folder, 'index.sense')
    wn = WordNetCorpusReader(path_to_wn_dict_folder, None)
    

# output path
base_output_path = os.path.join(args.output_folder, 
                                args.abstraction_level + '-' + args.wn_version + '_' + '_'.join(corpora_to_include))
output_path = base_output_path + '.txt'
log_path = base_output_path + '.log'
stats_path = base_output_path + '.stats'
lp_path = base_output_path + '.lp'
gold_lp_path = base_output_path + 'lp_gold'
df_output_path = base_output_path + '.bin'
case_freq_path = base_output_path + '.case_freq'
plural_freq_path = base_output_path + '.plural_freq'


print('end postprocessing command line args', datetime.now())


print('start loading instance_id mappings', datetime.now())

corpora_to_include = set(corpora_to_include)


# load mapping dictionary and function

# load instance_id to offset(s) from .key.txt file
sensekey2offset = mapping_utils.load_mapping_sensekey2offset(path_to_wn_index_sense,
                                                             args.wn_version)

instance_id2offset, instance_id2sensekeys = mapping_utils.load_instance_id2offset(input_mapping_path,
                                                                                  sensekey2offset,
                                                                                  args.wn_version,
                                                                                  debug=False)

if args.abstraction_level in {'sensekey', 'synset'}:


    if args.abstraction_level == 'sensekey':
        the_mapping = instance_id2sensekeys
        the_mapping_function = mapping_utils.map_sensekey_to_sensekey
    elif args.abstraction_level == 'synset':
        the_mapping = instance_id2offset
        the_mapping_function = mapping_utils.map_instance_id2synset

elif args.abstraction_level == 'direct_hypernym':
    the_mapping = mapping_utils.get_synset2hypernym(wn)
    the_mapping_function = mapping_utils.map_instance_id2direct_hypernym
elif args.abstraction_level == 'blc20':
    the_mapping = mapping_utils.get_synset2blc20(wn)
    the_mapping_function = mapping_utils.map_instance_id2blc20


print('end loading instance_id mappings', datetime.now())


print('start updating df', datetime.now())


## load competition df and add column to it mapping to different semantic level
df = pandas.read_pickle(args.competition_df)

column_name = 'synset2%s' % args.abstraction_level
df[column_name] = [None for index, row in df.iterrows()]

target_lemmas = set()
target_lemmas_pos = set()

if args.abstraction_level == 'blc20':
    blc_20_obj = BLC(20, 'all')


for index, row in df.iterrows():

    if args.abstraction_level == 'sensekey':
        sy2sensekeys = wn_utils.get_synset2sensekeys(wn, row['target_lemma'], row['pos'])
        df.set_value(index, column_name, sy2sensekeys)

    elif args.abstraction_level == 'direct_hypernym':
        synsets = wn.synsets(row['target_lemma'], row['pos'])

        sy2hypernyms = dict()
        for synset in synsets:
            synset_id = mapping_utils.synset2identifier(synset, '30')
            hypernym_id = the_mapping[synset_id]
            sy2hypernyms[synset_id] = hypernym_id
        df.set_value(index, column_name, sy2hypernyms)

    elif args.abstraction_level == 'blc20':
        synsets = wn.synsets(row['target_lemma'], row['pos'])

        sy2blc20 = dict()
        for synset in synsets:
            synset_id = mapping_utils.synset2identifier(synset, '30')
            blc20_id = the_mapping[synset_id]
            sy2blc20[synset_id] = blc20_id
        df.set_value(index, column_name, sy2blc20)

    target_lemmas.add(row['target_lemma'])
    target_lemmas_pos.add((row['target_lemma'], row['pos']))


print('finished updating df', datetime.now())


# load html
print('started loading html', datetime.now())
my_html_tree = html.parse(input_xml_path)
print('finished loading html', datetime.now())
outfile = open(output_path, 'w')
stats = defaultdict(int)


# label propagation info
sc_lemma_pos2label_sent_index = defaultdict(list)
omsti_lemma_pos2label_sent_index = defaultdict(list)
lp_gold = defaultdict(list)

lemma_lower_pos2meaning2freq_uppercase = dict()
lemma_lower_pos2meaning2freq_plural = dict()


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
                    if instance_id not in instance_id2offset:
                        continue

                    synset_id = instance_id2offset[instance_id]

                    # determine key for mapping
                    if args.abstraction_level in {'direct_hypernym', 'blc20'}:
                        key = synset_id
                    else:
                        key = instance_id

                    mappings = the_mapping_function(key, the_mapping)

                    for a_mapping in mappings:
                        sent_annotations.append(a_mapping)

                        stats[lemma] += 1


                        # case sensitive information
                        if all([token.istitle(),                    # startswith capital letter
                                not instance_id.endswith('t000')]): # not first token in sentence
                            freq_key = (lemma.lower(), mapped_pos)
                            if freq_key not in lemma_lower_pos2meaning2freq_uppercase:
                                lemma_lower_pos2meaning2freq_uppercase[freq_key] = defaultdict(int)
                            lemma_lower_pos2meaning2freq_uppercase[freq_key][a_mapping] += 1

                        # plural information
                        doc = nlp(token)
                        if doc[0].tag_ in {'NNS', 'NNPS'}:
                            freq_key = (lemma.lower(), mapped_pos)
                            if freq_key not in lemma_lower_pos2meaning2freq_plural:
                                lemma_lower_pos2meaning2freq_plural[freq_key] = defaultdict(int)
                            lemma_lower_pos2meaning2freq_plural[freq_key][a_mapping] += 1


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

		# add gold lp info 
                lp_gold[(target_lemma, target_pos)].append((token_annotation, sentence_tokens, target_index))
		
                # add to label propagation dicts
                if the_corpus == 'semcor':
                    sc_lemma_pos2label_sent_index[(target_lemma, target_pos)].append((token_annotation, sentence_tokens, target_index))
                elif the_corpus == 'mun':
                    omsti_lemma_pos2label_sent_index[(target_lemma, target_pos)].append((None, sentence_tokens, target_index))

outfile.close()

# save case freq
with open(case_freq_path, 'wb') as outfile:
    pickle.dump(lemma_lower_pos2meaning2freq_uppercase, outfile)

with open(plural_freq_path, 'wb') as outfile:
    pickle.dump(lemma_lower_pos2meaning2freq_plural, outfile)


# postprocess for stats
with open(stats_path, 'w') as outfile:
    outfile.write('number of unique lemma: %s\n' % len(stats))

    avg_num_instances = sum(stats.values()) / len(stats)
    outfile.write('avg number of training examples per lemma: %s\n' % round(avg_num_instances, 2))

# postprocess for label propagation input
lp_input = dict()

with open(log_path, 'w') as outfile:

    headers = ['lemma', 'pos', '#_semcor', '#_omsti', '#_omsti_sample']
    outfile.write('\t'.join(headers) + '\n')

    for target_lemma, target_pos in target_lemmas_pos:



        sc_info = sc_lemma_pos2label_sent_index[(target_lemma, target_pos)]
        omsti_info = omsti_lemma_pos2label_sent_index[(target_lemma, target_pos)]

        num_sc = len(sc_info)

        num_omsti_before = len(omsti_info)
        if len(omsti_info) > 1000:
            omsti_info = sample(omsti_info, 1000)
        num_omsti_after = len(omsti_info)

        if all([num_sc > 10,
                num_omsti_after > 10]):
                
            lp_input[(target_lemma, target_pos)] = sc_info + omsti_info
            assert num_sc > 10
            assert num_omsti_after > 10

        outfile.write('\t'.join([target_lemma,
                                 target_pos,
                                 str(num_sc),
                                 str(num_omsti_before),
                                 str(num_omsti_after)]) + '\n')

# add test instances to it and save identifiers to df
df['lp_index'] = [None for index, row in df.iterrows()]

for index, row in df.iterrows():

    lemma = row['target_lemma']
    pos = row['pos']

    # check polysemy
    synsets = wn.synsets(lemma, pos)

    if all([len(synsets) >= 2,
            (lemma, pos) in lp_input]):

        target_id = row['token_ids'][0]
        target_index = None
        sentence = []
        for an_index, sentence_token in enumerate(row['sentence_tokens']):

            if sentence_token.token_id == target_id:
                target_index = an_index

            sentence.append(sentence_token.text)

        assert target_index is not None

        sc_and_omsti_info = lp_input[(lemma, pos)]
        len_before = len(sc_and_omsti_info)

        lp_index = len(sc_and_omsti_info)

        lp_input[(lemma, pos)].append((None, sentence, target_index))

        len_after = len(lp_input[(lemma, pos)])

        assert len_before != len_after
        assert lp_index is not None, 'lp index is None for %s' % target_id

    else: # monosemous
        lp_index = None

    df.set_value(index, 'lp_index', lp_index)


df.to_pickle(df_output_path)

# save one dictionary for input label propagation
with open(lp_path, 'wb') as outfile:
    pickle.dump(lp_input, outfile)


# save one dictionary with gold label propagation
with open(gold_lp_path, 'wb') as outfile:
    pickle.dump(lp_gold, outfile)





















