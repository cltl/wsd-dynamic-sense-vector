import numpy as np
from sklearn.semi_supervised import LabelSpreading
from time import time
from scipy.spatial.distance import pdist
from collections import Counter
from scipy.sparse.csr import csr_matrix
import tensorflow as tf
import os
import sys

class LabelPropagation(object):
    
    def __init__(self, sess, vocab_path, model_path, batch_size):
        self.sess = sess
        self.batch_size = batch_size
        self.vocab = np.load(vocab_path)
        saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
        start_sec = time()
        sys.stdout.write('Loading... ')
        saver.restore(sess, model_path)
        sys.stdout.write('Done (%.0f sec).\n' %(time()-start_sec))
#         self.predicted_context_embs = sess.graph.get_tensor_by_name('Model/predicted_context_embs:0')
#         self.x = sess.graph.get_tensor_by_name('Model/x:0')
        self.x = sess.graph.get_tensor_by_name('Model_1/x:0')
        self.predicted_context_embs = sess.graph.get_tensor_by_name('Model_1/predicted_context_embs:0')
        self.lens = sess.graph.get_tensor_by_name('Model_1/lens:0')
        self.similarity_threshold = 0.95
        self.minimum_vertex_degree = 10
        
    def _convert_sense_ids(self, data):
        str2id = {}
        ids = []
        for lemma in data:
            for i in range(len(data[lemma])):
                sense_id, sentence_tokens, target_index = data[lemma][i]
                if sense_id is None:
                    sense_id = -1
                else:
                    if sense_id not in str2id:
                        str2id[sense_id] = len(str2id)
                        ids.append(sense_id)
                    sense_id = str2id[sense_id]
                data[lemma][i] = (sense_id, sentence_tokens, target_index)
        return ids
        
    def predict(self, data):
        '''
        input data format: dict(lemma -> list((sense_id[str], sentence_tokens, target_index)))
        set sense_id to None for unlabeled instances 

        batch_size: number of sentences in a batch to be used as input for LSTM
        
        output format: dict(lemma -> list(sense_id)), the order in each list corresponds to the input
        '''
        start_sec = time()
        adding_edges_elapsed_sec = 0
        num_low_degree_vertices = 0

        lstm_input = []
        target_id = self.vocab['<target>']
        pad_id = self.vocab['<pad>']
        sense_ids = self._convert_sense_ids(data)
        for lemma in data:
            for _, sentence_tokens, target_index in data[lemma]:
                sentence_as_ids = [self.vocab.get(w) or self.vocab['<unkn>'] 
                                   for w in sentence_tokens]
                sentence_as_ids[target_index] = target_id
                lstm_input.append(sentence_as_ids)
        lens = [len(s) for s in lstm_input]
        max_len = max(lens)
        for s in lstm_input:
            while len(s) < max_len:
                s.append(pad_id)
        lens = np.array(lens)
        lstm_input = np.array(lstm_input)
        lstm_output = []
        for batch_start in range(0, len(lstm_input), self.batch_size):
            batch_end = min(len(lstm_input), batch_start+self.batch_size)
            lstm_output.append(self.sess.run(self.predicted_context_embs, 
                                             {self.x: lstm_input[batch_start:batch_end], 
                                              self.lens: lens[batch_start:batch_end]}))
        lstm_output = np.vstack(lstm_output)
        
        output = {}
        start_index = 0
        for lemma in data:
            stop_index = start_index + len(data[lemma])
            contexts = lstm_output[start_index:stop_index]
            start_index = stop_index
            labels = [sense for sense, _, _ in data[lemma]]
            # choose edges
            num_examples = len(contexts)
            distances = pdist(contexts)
            rows = np.array([i for i in range(num_examples-1) for j in range(i+1,num_examples)])
            cols = np.array([j for i in range(num_examples-1) for j in range(i+1,num_examples)])
            sorted_indices = np.argsort(distances)
            most_similar_pairs = sorted_indices[0:int(len(distances)*(1-self.similarity_threshold))]
            # add edges to low-connectivity vertices
            adding_edges_start_sec = time()
            degree = Counter()
            degree.update(rows[most_similar_pairs])
            degree.update(cols[most_similar_pairs])
            additional_pairs = set()
            for v in range(num_examples):
                i = len(most_similar_pairs) # those are already used
                if degree[v] < self.minimum_vertex_degree: 
                    num_low_degree_vertices += 1
                while degree[v] < self.minimum_vertex_degree and i < len(sorted_indices):
                    idx = sorted_indices[i]
                    if idx not in additional_pairs and (rows[idx] == v or cols[idx] == v):
                        additional_pairs.add(idx)
                        degree[rows[idx]] += 1
                        degree[cols[idx]] += 1
                    i += 1
            adding_edges_elapsed_sec += (time() - adding_edges_start_sec) 
            # make the matrix
            p = np.concatenate([most_similar_pairs, list(additional_pairs)])
            affinity = csr_matrix((distances[p], (rows[p], cols[p])), 
                                  shape=(num_examples,num_examples))
            # predict
            label_prop_model = LabelSpreading(kernel=lambda a, b: affinity)
            label_prop_model.fit(contexts, labels)
            output[lemma] = [sense_ids[index] for index in label_prop_model.transduction_]
            
        elapsed_sec = (time()-start_sec)
        print('Elapsed time: %.2f min' %(elapsed_sec/60.0))
        print('Time for adding edges: %.2f min (%.2f%% of total time)' 
              %(adding_edges_elapsed_sec/60.0, 
                adding_edges_elapsed_sec*100.0/elapsed_sec))
        print('Number of vertices with low connectivity: %d (%.2f%% of all vertices)' 
              %(num_low_degree_vertices, num_low_degree_vertices*100.0/len(lstm_output)))
        return output
    
if __name__ == '__main__':
    vocab_path = '/home/minhle/scratch/wsd-with-marten/preprocessed-data/40e2c1f/gigaword-for-lstm-wsd.index.pkl'
    model_path = '/home/minhle/scratch/wsd-with-marten/output/lstm-wsd-google_trained_on_gigaword_10pc'
    assert os.path.exists(vocab_path) and os.path.exists(model_path + '.meta'), 'Please update the paths hard-coded in this file (for testing only)'
    with tf.Session() as sess:
        lp = LabelPropagation(sess, vocab_path, model_path, 2)
        senses = lp.predict({'dog': [('dog.01', 'The dog runs through the yard'.split(), 1),
                                     ('dog.02', 'He ate a hot dog'.split(), 4),
                                     (None, 'Dogs are friends of human'.split(), 0)],
                             'horse': [('horse.01', 'She enjoys watching horse races', 3),
                                       ('horse.01', 'He plays with only one horse against two bishops', 5),
                                       [None, 'Horses, horses, horses', 0]]})
        print(senses)
        