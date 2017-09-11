import numpy as np
import tensorflow as tf
from collections import defaultdict 
import argparse
import pickle
from datetime import datetime
from itertools import islice


parser = argparse.ArgumentParser(description='Trains meaning embeddings based on precomputed LSTM model')
parser.add_argument('-m', dest='model_path', required=True, help='path to model trained LSTM model')
# model_path = '/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/lstm-wsd-small'
parser.add_argument('-v', dest='vocab_path', required=True, help='path to LSTM vocabulary')
#vocab_path = '/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/gigaword.1m-sents-lstm-wsd.index.pkl'
parser.add_argument('-i', dest='input_path', required=True, help='input path with sense annotated sentences')
parser.add_argument('-o',dest='output_path', required=True, help='path where sense embeddings will be stored')
parser.add_argument('-b', dest='batch_size', required=True, help='batch size')
parser.add_argument('-t', dest='max_lines', required=True, help='maximum number of lines you want to train on')
parser.add_argument('-s', dest='setting', required=True, help='sensekey | synset | hdn')
args = parser.parse_args()

print('loaded arguments for training meaning embeddings')

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
print('loaded vocab')

synset2context_embds = defaultdict(list)
meaning_freqs = defaultdict(int)
batch_size = int(args.batch_size)


with tf.Session() as sess:  # your session object
    saver = tf.train.import_meta_graph(args.model_path + '.meta', clear_devices=True)
    saver.restore(sess, args.model_path)
    x = sess.graph.get_tensor_by_name('Model_1/x:0')
    predicted_context_embs = sess.graph.get_tensor_by_name('Model_1/predicted_context_embs:0')
    lens = sess.graph.get_tensor_by_name('Model_1/lens:0')

    identifiers = [] # list of sy_ids
    annotated_sentences = []
    sentence_lens = [] # list of ints

    with open(args.input_path) as infile:
        with open(path, 'rb') as f:
            for n_lines in iter(lambda: tuple(islice(f, batch_size)), ()):
            

            if counter >= int(args.max_lines):
                break

            if counter % 1000 == 0:
                print(counter, datetime.now())


            sentence = line.strip()
            tokens, annotation_indices = ctx_embd_input(sentence)

            for index, synset_id in annotation_indices:

                if args.setting == 'hdn':
                    base_synset, synset_id = synset_id.split('_')

                sentence_as_ids = [vocab.get(w) or vocab['<unkn>'] for w in tokens]
                target_id = vocab['<target>']
                sentence_as_ids[index] = target_id

                meaning_freqs[synset_id] += 1

                # update batch information
                identifiers.append(synset_id)
                annotated_sentences.append(sentence_as_ids)
                sentence_lens.append(len(sentence_as_ids))

                if len(annotated_sentences) == batch_size:
                    max_length  = max([len(_list) for _list in annotated_sentences])
                    for _list in annotated_sentences:
                        length_diff = max_length - len(_list)
                        [_list.append(vocab['<unkn>']) for _ in range(length_diff)]

                    target_embeddings = sess.run(predicted_context_embs, {x: annotated_sentences,
                                                                          lens: sentence_lens})

                    for synset_id, target_embedding in zip(identifiers, target_embeddings):
                        synset2context_embds[synset_id].append(target_embedding)

                    identifiers = []
                    annotated_sentences = []
                    sentence_lens = []


synset2avg_embedding = dict()
for synset, embeddings in synset2context_embds.items():
    average = sum(embeddings) / len(embeddings)
    synset2avg_embedding[synset] = average

with open(args.output_path, 'wb') as outfile:
    pickle.dump(synset2avg_embedding, outfile)

with open(args.output_path + '.freq', 'wb') as outfile:
    pickle.dump(meaning_freqs, outfile)
