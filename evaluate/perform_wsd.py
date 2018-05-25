import numpy as np
import os
import tensorflow as tf
import json
import argparse
import pickle
import pandas
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from scipy import spatial
import morpho_utils
import tensor_utils as utils
import score_utils
import tsne_utils
import official_scorer
from wn_utils import synset2identifier

parser = argparse.ArgumentParser(description='Perform WSD using LSTM model')
parser.add_argument('-m', dest='model_path', required=True, help='path to model trained LSTM model')
parser.add_argument('-v', dest='vocab_path', required=True, help='path to LSTM vocabulary')
parser.add_argument('-c', dest='wsd_df_path', required=True, help='input path to dataframe wsd competition')
parser.add_argument('-l', dest='log_path', required=True, help='path where exp settings are stored')
parser.add_argument('-s', dest='sense_embeddings_path', required=True, help='path where sense embeddings are stored')
parser.add_argument('-o', dest='output_path', required=True, help='path where output wsd will be stored')
parser.add_argument('-r', dest='results', required=True, help='path where accuracy will be reported')
parser.add_argument('-g', dest='gran', required=True, help='sensekey | synset')
parser.add_argument('-f', dest='mfs_fallback', required=True, help='True or False')
parser.add_argument('-t', dest='path_case_freq', help='path to pickle with case freq')
parser.add_argument('-a', dest='use_case_strategy', help='set to True to use morphological strategy case')
parser.add_argument('-p', dest='path_plural_freq', help='path to pickle with plural freq')
parser.add_argument('-b', dest='use_number_strategy', help='set to True to use morphological strategy number')
parser.add_argument('-y', dest='path_lp', help='path to lp output')
parser.add_argument('-z', dest='use_lp', help='set to True to use label propagation') 


args = parser.parse_args()
args.mfs_fallback = args.mfs_fallback == 'True'
case_strategy = args.use_case_strategy == 'True'
number_strategy = args.use_number_strategy == 'True'
lp_strategy = args.use_lp == 'True'

case_freq = pickle.load(open(args.path_case_freq, 'rb'))
plural_freq = pickle.load(open(args.path_plural_freq, 'rb'))


lp_info = dict()
if lp_strategy:
    lp_info = pickle.load(open(args.path_lp, 'rb'))

the_wn_version = '30'
# load relevant wordnet
if '171' in args.wsd_df_path:
    the_wn_version = '171'
    cwd = os.path.dirname(os.path.realpath(__file__))
    path_to_wn_dict_folder = os.path.join(cwd, 'resources', 'wordnet_171', 'WordNet-1.7.1', 'dict')
    wn = WordNetCorpusReader(path_to_wn_dict_folder, None)


with open(args.sense_embeddings_path + '.freq', 'rb') as infile:
    meaning_freqs = pickle.load(infile)

with open(args.log_path, 'w') as outfile:
    json.dump(args.__dict__, outfile)


def lp_output(row, lp_info, candidate_synsets, debug=False):
    target_lemma = row['target_lemma']
    target_pos = row['pos']

    key = (target_lemma, target_pos)

    if key not in lp_info:
        if debug:
            print(target_lemma, target_pos, 'not in lp_info')
        return None

    lp_index = row['lp_index']
    if lp_index is None:
        print('lp_index is None')
        return None

    synset_identifier = lp_info[(target_lemma, target_pos)][lp_index]

    return synset_identifier

def extract_sentence_wsd_competition(row):
    """
    given row in dataframe (representing task instance)
    return raw sentence + index of target word to be disambiguated + lemma

    :param pandas.core.series.Series row: row in dataframe representing task instance

    :rtype: tuple
    :return: (target_index, sentence_tokens, lemma)
    """
    target_index = None
    lemma = None
    pos = None
    sentence_tokens = []

    for index, sentence_token in enumerate(row['sentence_tokens']):
        if sentence_token.token_id in row['token_ids']:
            target_index = index
            lemma = sentence_token.lemma
            pos = sentence_token.pos

        sentence_tokens.append(sentence_token.text)

    assert len(sentence_tokens) >= 2
    #assert pos is not None # only needed for sem2013-aw
    #assert lemma is not None, (lemma, pos)
    #assert target_index is not None

    return target_index, sentence_tokens, lemma, pos


def load_tensors(sess):
    x = sess.graph.get_tensor_by_name('Model/x:0')
    logits = sess.graph.get_tensor_by_name('Model/Max:0') # should have had a name
    lens = sess.graph.get_tensor_by_name('Model/lens:0')
    candidates = sess.graph.get_tensor_by_name('Model/candidate_list:0')
    
    return x, logits, lens, candidates


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

def find_hdns(lemma):
    graph_info = synsets_graph_info(wn_instance=wn,
                                wn_version='30',
                                lemma=lemma,
                                pos='n')
    hdn2synset = {info['under_lcs']: synset for synset, info in graph_info.items()}
    hdn_list = tuple(sorted(info['under_lcs'] # sorted to avoid arbitrary order
                        for info in graph_info.values() 
                        if info['under_lcs']))
    return hdn_list, hdn2synset


import generate_hdn_datasets
from block_timer.timer import Timer
import pickle

word_vocab_path = 'output/vocab.2018-05-10-7d764e7.pkl'
word2id = pickle.load(open(word_vocab_path, 'rb'))
hdn_vocab_path = 'output/hdn-vocab.2018-05-18-f48a06c.pkl'
hdn2id = pickle.load(open(hdn_vocab_path, 'rb'))
hdn_list_vocab_path = 'output/hdn-list-vocab.2018-05-18-f48a06c.pkl'
hdn_list2id = pickle.load(open(hdn_list_vocab_path, 'rb'))

def score_synsets(target_embedding, candidate_synsets, sense_embeddings, instance_id, lemma, pos, gran, synset2higher_level):
    """
    perform wsd

    :param numpy.ndarray target_embedding: predicted lstm embedding of sentence
    :param set candidate_synsets: candidate synset identifier of lemma, pos
    :param dict sense_embeddings: dictionary of sense embeddings

    :rtype: str
    :return: synset with highest cosine sim
    (if 2> options, one is picked)
    """
    highest_synsets = []
    highest_conf = -100
    synset_std = None
    candidate_freq = dict()
    strategy = 'lstm'

    for synset in candidate_synsets:
        if gran == 'synset':
            candidate = synset
            candidate_freq[synset] = meaning_freqs[candidate]
        elif gran in {'sensekey', 'blc20', 'direct_hypernym'}:
            candidate = None
            if synset in synset2higher_level:
                candidate = synset2higher_level[synset]
                candidate_freq[synset] = meaning_freqs[candidate]
                candidate_freq[synset] = meaning_freqs[candidate]
            else:
                candidate_freq[synset] = 0

        if candidate not in sense_embeddings:
            #print('%s %s %s: candidate %s missing in sense embeddings' % (instance_id, lemma, pos, candidate))
            continue

        cand_embedding, cand_std = sense_embeddings[candidate]
        sim = 1 - spatial.distance.cosine(cand_embedding, target_embedding)

        potentially_added_synset = (synset, cand_std)

        if sim == highest_conf:
            highest_synsets.append(potentially_added_synset)
        elif sim > highest_conf:
            highest_synsets = [potentially_added_synset]
            highest_conf = sim

    assert len(candidate_freq) == len(set(candidate_synsets)), (candidate_freq, candidate_synsets)

    if len(highest_synsets) == 1:
        highest_synset, synset_std = highest_synsets[0]
    elif len(highest_synsets) >= 2:
        highest_synset, synset_std = highest_synsets[0]
        #print('%s %s %s: 2> synsets with same conf %s: %s' % (instance_id, lemma, pos, highest_conf, highest_synsets))
    else:
        if args.mfs_fallback:
            highest_synset = candidate_synsets[0]
            synset_std = None
            #print('%s: no highest synset -> mfs' % instance_id)
            strategy = 'mfs_fallback'
        else:
            highest_synset = None
            synset_std = None
    return highest_synset, synset_std, candidate_freq, strategy


if __name__ == '__main__':
        
    # load wsd competition dataframe
    wsd_df = pandas.read_pickle(args.wsd_df_path)
    
    # add output column
    wsd_df['lstm_output'] = [None for _ in range(len(wsd_df))]
    wsd_df['target_embedding'] = [None for _ in range(len(wsd_df))]
    wsd_df['std_chosen_synset'] = [None for _ in range(len(wsd_df))]
    wsd_df['lstm_acc'] = [None for _ in range(len(wsd_df))]
    wsd_df['emb_freq'] = [None for _ in range(len(wsd_df))]
    wsd_df['#_cand_synsets'] = [None for _ in range(len(wsd_df))]
    wsd_df['#_new_cand_synsets'] = [None for _ in range(len(wsd_df))]
    wsd_df['gold_in_new_cand_synsets'] = [None for _ in range(len(wsd_df))]
    wsd_df['wsd_strategy'] = [None for _ in range(len(wsd_df))]
    wsd_df['num_embeddings'] = [None for _ in range(len(wsd_df))]
    wsd_df['has_gold_embedding'] = [None for _ in range(len(wsd_df))]
    
    # load sense embeddings
    with open(args.sense_embeddings_path, 'rb') as infile:
        sense_embeddings = pickle.load(infile)
    
    # num correct
    num_correct = 0
    
    vocab = np.load(args.vocab_path)
    with tf.Session() as sess:  # your session object
        path = 'output/hdn-large.2018-05-21-b1d1867-best-model'
        saver = tf.train.import_meta_graph(path + '.meta', clear_devices=True)
        saver.restore(sess, path)
        x, logits, lens, candidates = load_tensors(sess)
        
        #predicted_context_embs = sess.graph.get_tensor_by_name('Model/predicted_context_embs:0')
        #x = sess.graph.get_tensor_by_name('Model/Placeholder:0')
    
        for row_index, row in wsd_df.iterrows():
            target_index, sentence_tokens, lemma, pos =  extract_sentence_wsd_competition(row)
            instance_id = row['token_ids'][0]
            target_id = vocab['<target>']
            sentence_as_ids = [vocab.get(w) or vocab['<unkn>'] for w in sentence_tokens]
            sentence_as_ids[target_index] = target_id
    
            hdn_list, hdn2synset = find_hdns(lemma)
            feed_dict = {x: [sentence_as_ids],
                         lens: [len(sentence_as_ids)],
                         candidates: [hdn_list2id[hdn_list]]}
            target_embeddings = sess.run(logits, feed_dict=feed_dict)
            scores = [target_embeddings[0,hdn2id[hdn]] for hdn in hdn_list]
    
            meaning2confidence = {hdn2synset[hdn]: score for hdn, score in zip(hdn_list, scores)}

            # load token object
            token_obj = row['tokens'][0]
    
            # morphology reduced polysemy
            pos = row['pos']
            if the_wn_version in {'171'}:
                pos = None
            candidate_synsets, \
            new_candidate_synsets, \
            gold_in_candidates = morpho_utils.candidate_selection(wn,
                                                                  token=token_obj.text,
                                                                  target_lemma=row['target_lemma'],
                                                                  pos=row['pos'],
                                                                  morphofeat=token_obj.morphofeat,
                                                                  use_case=case_strategy,
                                                                  use_number=number_strategy,
                                                                  gold_lexkeys=row['lexkeys'],
                                                                  case_freq=case_freq,
                                                                  plural_freq=plural_freq,
                                                                  debug=False)
    
            the_chosen_candidates = [synset2identifier(synset, wn_version=the_wn_version)
                                     for synset in new_candidate_synsets]
    
            # get mapping to higher abstraction level
            synset2higher_level = dict()
            if args.gran in {'sensekey', 'blc20', 'direct_hypernym'}:
                label = 'synset2%s' % args.gran
                synset2higher_level = row[label]
    
            # determine wsd strategy used
            if len(candidate_synsets) == 1:
                wsd_strategy = 'monosemous'
            elif len(new_candidate_synsets) == 1:
                wsd_strategy = 'morphology_solved'
            elif len(candidate_synsets) == len(new_candidate_synsets):
                wsd_strategy = 'lstm'
            elif len(new_candidate_synsets) < len(candidate_synsets):
                wsd_strategy = 'morphology+lstm'
    
            # possibly include label propagation strategy
            if lp_strategy:
                lp_result = lp_output(row, lp_info, new_candidate_synsets, debug=True)
    
                print(lp_result, row['lp_index'], row['target_lemma'])
    
                if lp_result:
                    the_chosen_candidates = [lp_result]
                    wsd_strategy = 'lp'
    
            # perform wsd
            if len(the_chosen_candidates) >= 2:
                chosen_synset = max(the_chosen_candidates, key=lambda m: meaning2confidence[m])
                candidate_std = 0
                candidate_freq = 0
            else:
                chosen_synset = None
                candidate_std = None
                if the_chosen_candidates:
                    chosen_synset = the_chosen_candidates[0]
                candidate_freq = dict()
    
            # add to dataframe
            wsd_df.set_value(row_index, col='target_embedding', value=0)
            wsd_df.set_value(row_index, col='lstm_output', value=chosen_synset)
            wsd_df.set_value(row_index, col='std_chosen_synset', value=candidate_std)
    
            wsd_df.set_value(row_index, col='#_cand_synsets', value=len(candidate_synsets))
            wsd_df.set_value(row_index, col='#_new_cand_synsets', value=len(new_candidate_synsets))
            wsd_df.set_value(row_index, col='gold_in_new_cand_synsets', value=gold_in_candidates)
    
            # score it
            lstm_acc = chosen_synset in row['source_wn_engs'] # used to be wn30_engs
    
    
            has_gold_embedding = False
    
            for source_wn_eng in row['source_wn_engs']:
                if source_wn_eng in candidate_freq:
                    if candidate_freq[source_wn_eng]:
                        has_gold_embedding = True
    
            num_embeddings = 0
            for synset_id, freq in candidate_freq.items():
                if synset_id in sense_embeddings:
                    num_embeddings += 1
    
            wsd_df.set_value(row_index, col='has_gold_embedding', value=has_gold_embedding)
            wsd_df.set_value(row_index, col='num_embeddings', value=num_embeddings)
            wsd_df.set_value(row_index, col='lstm_acc', value=lstm_acc)
            wsd_df.set_value(row_index, col='emb_freq', value=candidate_freq)
            wsd_df.set_value(row_index, col='wsd_strategy', value=wsd_strategy)
    
            if lstm_acc:
                num_correct += 1
    
    print(num_correct)
    
    # save it
    wsd_df.to_pickle(args.output_path)
    
    with open(args.results, 'w') as outfile:
        outfile.write('%s' % num_correct)
    
    # json output path
    output_path_json = args.results.replace('.txt', '.json')
    
    results = score_utils.experiment_results(wsd_df, args.mfs_fallback, args.wsd_df_path)
    
    with open(output_path_json, 'w') as outfile:
        json.dump(results, outfile)
    
    # official scorer if possible
    exp_folder = args.results.replace('/results.txt', '')
    official_scorer.create_key_file(wn, exp_folder, debug=1)
    official_scorer.score_using_official_scorer(exp_folder, 
                                                scorer_folder='resources/WSD_Unified_Evaluation_Datasets')
    
    # write tsne visualizations
    visualize = False
    if visualize:
        output_folder = args.results.replace('/results.txt', '')
        tsne_utils.create_tsne_visualizations(output_folder,
                                              correct={False, True},
                                              meanings=True,
                                              instances=True,
                                              polysemy=range(2, 1000),
                                              num_embeddings=range(2, 1000))
