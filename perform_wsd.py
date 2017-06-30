import numpy as np
import tensorflow as tf
import argparse
import pickle
import pandas
from nltk.corpus import wordnet as wn
from scipy import spatial

parser = argparse.ArgumentParser(description='Perform WSD using LSTM model')
parser.add_argument('-m', dest='model_path', required=True, help='path to model trained LSTM model')
# model_path = '/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/lstm-wsd-small'
parser.add_argument('-v', dest='vocab_path', required=True, help='path to LSTM vocabulary')
# vocab_path = '/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/gigaword.1m-sents-lstm-wsd.index.pkl'
parser.add_argument('-c', dest='wsd_df_path', required=True, help='input path to dataframe wsd competition')
parser.add_argument('-s', dest='sense_embeddings_path', required=True, help='path where sense embeddings are stored')
parser.add_argument('-o', dest='output_path', required=True, help='path where output wsd will be stored')
args = parser.parse_args()


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
    assert pos is not None
    assert lemma is not None
    assert target_index is not None

    return target_index, sentence_tokens, lemma, pos


def score_synsets(target_embedding, candidate_synsets, sense_embeddings, instance_id, lemma, pos):
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
    highest_conf = 0.0

    for candidate in candidate_synsets:
        if candidate not in sense_embeddings:
            print('%s %s %s: candidate %s missing in sense embeddings' % (instance_id, lemma, pos, candidate))
            continue 

        cand_embedding = sense_embeddings[candidate]
        sim = 1 - spatial.distance.cosine(cand_embedding, target_embedding)

        if sim == highest_conf:
            highest_synsets.append(candidate)
        elif sim > highest_conf:
            highest_synsets = [candidate]
            highest_conf = sim

    if len(highest_synsets) == 1:
        highest_synset = highest_synsets[0]
    elif len(highest_synsets) >= 2:
        highest_synset = highest_synsets[0]
        print('%s %s %s: 2> synsets with same conf %s: %s' % (instance_id, lemma, pos, highest_conf, highest_synsets))
    else:
        highest_synset = None
        print('%s: no highest synset' % instance_id)
    return highest_synset


# load wsd competition dataframe
wsd_df = pandas.read_pickle(args.wsd_df_path)

# add output column
wsd_df['lstm_output'] = [None for _ in range(len(wsd_df))]
wsd_df['lstm_acc'] = [None for _ in range(len(wsd_df))]


# load sense embeddings
with open(args.sense_embeddings_path, 'rb') as infile:
    sense_embeddings = pickle.load(infile)

# num correct
num_correct = 0

vocab = np.load(args.vocab_path)
with tf.Session() as sess:  # your session object
    saver = tf.train.import_meta_graph(args.model_path + '.meta', clear_devices=True)
    saver.restore(sess, args.model_path)
    predicted_context_embs = sess.graph.get_tensor_by_name('Model/predicted_context_embs:0')
    x = sess.graph.get_tensor_by_name('Model/x:0')

    for row_index, row in wsd_df.iterrows():
        target_index, sentence_tokens, lemma, pos =  extract_sentence_wsd_competition(row)
        instance_id = row['token_ids'][0]
        target_id = vocab['<target>']
        sentence_as_ids = [vocab.get(w) or vocab['<unkn>'] for w in sentence_tokens]
        sentence_as_ids[target_index] = target_id
        target_embedding = sess.run(predicted_context_embs, {x: [sentence_as_ids]})[0]

        # load candidate synsets
        synsets = wn.synsets(lemma, pos=pos)
        candidate_synsets = {synset2identifier(synset, wn_version='30')
                             for synset in synsets}

        # perform wsd
        if len(candidate_synsets) >= 2:
            chosen_synset = score_synsets(target_embedding, candidate_synsets, sense_embeddings, instance_id, lemma, pos)
        else:
            chosen_synset = candidate_synsets.pop()

        # add to dataframe
        wsd_df.set_value(row_index, col='lstm_output', value=chosen_synset)

        # score it
        lstm_acc = chosen_synset in row['wn30_engs']
        wsd_df.set_value(row_index, col='lstm_acc', value=lstm_acc)
        
        if lstm_acc:
            num_correct += 1

print(num_correct)

# save it
wsd_df.to_pickle(args.output_path)









