import numpy as np
import tensorflow as tf
from collections import defaultdict 

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
    
model_path = '/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/lstm-wsd-small'
vocab = np.load('/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/gigaword.1m-sents-lstm-wsd.index.pkl')
synset2context_embds = defaultdict(list)
  
with tf.Session() as sess:  # your session object
    saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
    saver.restore(sess, model_path)
    predicted_context_embs = sess.graph.get_tensor_by_name('Model/predicted_context_embs:0')
    x = sess.graph.get_tensor_by_name('Model/x:0')

    sentence = 'Have you permitted it to become a giveaway---eng-30-00032613-n program---eng-30-05616786-n'
    tokens, annotation_indices = ctx_embd_input(sentence)
    for index, synset_id in annotation_indices:
        target_id = vocab['<target>']
        sentence_as_ids = [vocab.get(w) or vocab['<unkn>'] for w in tokens]
        sentence_as_ids[index] = target_id
        target_embedding = sess.run(predicted_context_embs, {x: [sentence_as_ids]})
        synset2context_embds[synset_id].append(target_embedding[0])


for synset, embeddings in synset2context_embds.items():
    average = sum(embeddings) / len(embeddings)
    print(synset, average)
    break
