import numpy as np
import tensorflow as tf
from collections import defaultdict 
import argparse
import pickle

parser = argparse.ArgumentParser(description='Trains meaning embeddings based on precomputed LSTM model')
parser.add_argument('-m', dest='model_path', required=True, help='path to model trained LSTM model')
# model_path = '/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/lstm-wsd-small'
parser.add_argument('-v', dest='vocab_path', required=True, help='path to LSTM vocabulary')
#vocab_path = '/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/gigaword.1m-sents-lstm-wsd.index.pkl'
parser.add_argument('-i', dest='input_path', required=True, help='input path with sense annotated sentences')
parser.add_argument('-o',dest='output_path', required=True, help='path where sense embeddings will be stored')
parser.add_argument('-t', dest='max_lines', required=True, help='maximum number of lines you want to train on')
args = parser.parse_args()

def ctx_embd_input(sentence):
    """
    given a annotated sentence, return
    each the sentence with only one annotation
    
    :param str sentence: a sentence with annotations
    (lemma---annotation)
    
    :rtype: generator
    :return: generator of input for the lstm (synset_id, sentence)
    """
    sent_split = sentence.split()

    annotation_indices = []
    tokens = []
    for index, token in enumerate(sent_split):
        token, *annotation = token.split('---')
        tokens.append(token)
        
        if annotation:
            annotation_indices.append((index, annotation[0]))
    
    return tokens, annotation_indices    
    
vocab = np.load(args.vocab_path)
synset2context_embds = defaultdict(list)
  
with tf.Session() as sess:  # your session object
    saver = tf.train.import_meta_graph(args.model_path + '.meta', clear_devices=True)
    saver.restore(sess, args.model_path)
    predicted_context_embs = sess.graph.get_tensor_by_name('Model/predicted_context_embs:0')
    x = sess.graph.get_tensor_by_name('Model/x:0')

    with open(args.input_path) as infile:
        for counter, line in enumerate(infile):
            if counter >= int(args.max_lines):
                break
            sentence = line.strip()
            tokens, annotation_indices = ctx_embd_input(sentence)
            for index, synset_id in annotation_indices:
                target_id = vocab['<target>']
                sentence_as_ids = [vocab.get(w) or vocab['<unkn>'] for w in tokens]
                sentence_as_ids[index] = target_id
                target_embedding = sess.run(predicted_context_embs, {x: [sentence_as_ids]})
                synset2context_embds[synset_id].append(target_embedding[0])


synset2avg_embedding = dict()
for synset, embeddings in synset2context_embds.items():
    average = sum(embeddings) / len(embeddings)
    synset2avg_embedding[synset] = average

with open(args.output_path, 'wb') as outfile:
    pickle.dump(synset2avg_embedding, outfile)
