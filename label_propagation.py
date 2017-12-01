import numpy as np
from time import time
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from scipy.sparse.csr import csr_matrix
import tensorflow as tf
import os
import sys
from sklearn import semi_supervised
import math

class RBF(object):
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, u, v):
        return math.exp(-self.gamma*np.sum((u-v)**2))
    
def expander(u, v): 
    ''' The similarity function used by Ravi and Diao (2015, pp 524) 
    Ravi, S., & Diao, Q. (2015). Large Scale Distributed Semi-Supervised 
    Learning Using Streaming Approximation. arXiv:1512.01752 [Cs], 51.
    '''
    return np.dot(u,v)

class LabelPropagation(object):
    
    def __init__(self, sess, vocab_path, model_path, batch_size, sim_func=expander):
        self.sess = sess
        self.batch_size = batch_size
        self.sim_func = sim_func
        self.vocab = np.load(vocab_path)
        saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
        start_sec = time()
        sys.stdout.write('Loading model from %s... ' %model_path)
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
        data2 = dict((lemma, []) for lemma in data)
        str2id = {}
        ids = []
        for lemma in data:
            for sense_id, sentence_tokens, target_index in data[lemma]:
                if sense_id is None:
                    sense_id = -1
                else:
                    if sense_id not in str2id:
                        str2id[sense_id] = len(str2id)
                        ids.append(sense_id)
                    sense_id = str2id[sense_id]
                data2[lemma].append((sense_id, sentence_tokens, target_index))
        return data2, ids
        
    def _apply_label_propagation_model(self, contexts, labels, affinity):
        label_prop_model = semi_supervised.LabelPropagation(kernel=lambda a, b: affinity)
        label_prop_model.fit(contexts, labels)
        return label_prop_model.transduction_
            
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
        num_added_edges = 0
        num_total_edges = 0
        sense_counts = {}

        print('Running LSTM...')
        lstm_input = []
        target_id = self.vocab['<target>']
        pad_id = self.vocab['<pad>']
        converted_data, sense_ids = self._convert_sense_ids(data)
        for lemma_no, lemma in enumerate(converted_data):
            sense_counts[lemma] = Counter()
            sense_counts[lemma].subtract(sense for sense, _, _ in data[lemma] 
                                         if sense is not None)
#             if lemma_no >= 100: break # for debugging
            for _, sentence_tokens, target_index in converted_data[lemma]:
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
        for batch_no, batch_start in enumerate(range(0, len(lstm_input), self.batch_size)):
            batch_end = min(len(lstm_input), batch_start+self.batch_size)
            lstm_output.append(self.sess.run(self.predicted_context_embs, 
                                             {self.x: lstm_input[batch_start:batch_end], 
                                              self.lens: lens[batch_start:batch_end]}))
            if (batch_no+1) % 100 == 0:
                print('Batch #%d...' %(batch_no+1))
        lstm_output = np.vstack(lstm_output)
        print('Running LSTM... Done.')
        
        output = {}
        start_index = 0
        for lemma_no, lemma in enumerate(converted_data):
#             if lemma_no >= 100: break # for debugging
            print("Lemma #%d of %d: %s" %(lemma_no, len(converted_data), lemma))
            stop_index = start_index + len(converted_data[lemma])
            contexts = lstm_output[start_index:stop_index]
            start_index = stop_index
            labels = [sense for sense, _, _ in converted_data[lemma]]
            # choose edges
            num_examples = len(contexts)
            distances = euclidean_distances(contexts)
            sorted_indices = np.dstack(np.unravel_index(np.argsort(distances.ravel()), 
                                                        distances.shape))[0]
            sorted_indices = [(u, v) for u, v in sorted_indices if u < v] # keep only one of two equivalent pairs
            num_most_similar_pairs = int(num_examples*(num_examples-1)*(1-self.similarity_threshold))
            selected_pairs = set(pair for pair in sorted_indices[:num_most_similar_pairs])
            sorted_within_row_indices = np.argsort(distances)
            # add edges to low-connectivity vertices
            degree = Counter()
            degree.update(u for u, _ in selected_pairs)
            degree.update(v for _, v in selected_pairs)
            adding_edges_start_sec = time()
            for v in range(num_examples):
                if degree[v] < self.minimum_vertex_degree: 
                    num_low_degree_vertices += 1
                    for idx in sorted_within_row_indices[v]:
                        if degree[v] >= self.minimum_vertex_degree: break
                        if (v, idx) not in selected_pairs and (idx, v) not in selected_pairs:
                            selected_pairs.add((v, idx))
                            degree[v] += 1
                            degree[idx] += 1
                            num_added_edges += 1
            adding_edges_elapsed_sec += (time() - adding_edges_start_sec)
            num_total_edges += len(selected_pairs) 
            # make the matrix
            for u, v in selected_pairs:
                print('\t%d, %d --> %f' %(u, v, self.sim_func(contexts[u], contexts[v])))
            sims, rows, cols = zip(*[(self.sim_func(contexts[u], contexts[v]), u,v) 
                                     for u,v in selected_pairs] +
                                    [(self.sim_func(contexts[v], contexts[u]), v,u) 
                                     for u,v in selected_pairs])
            affinity = csr_matrix((sims, (rows, cols)), shape=(num_examples,num_examples))
            # predict
            output[lemma] = [sense_ids[index] for index in 
                             self._apply_label_propagation_model(contexts, labels, affinity)]
            sense_counts[lemma].update(output[lemma])
            print(sense_counts[lemma].most_common())
            
        elapsed_sec = (time()-start_sec)
        print('Elapsed time: %.2f min' %(elapsed_sec/60.0))
        print('Time for adding edges: %.2f min (%.2f%% of total time)' 
              %(adding_edges_elapsed_sec/60.0, 
                adding_edges_elapsed_sec*100.0/elapsed_sec))
        print('Number of vertices with low connectivity: %d (%.2f%% of all vertices)' 
              %(num_low_degree_vertices, num_low_degree_vertices*100.0/len(lstm_output)))
        return output
    
class LabelSpreading(LabelPropagation):
    ''' Allowing the label properation algorithm to change the underlying gold
    annotations. It's designed to work with noisy data but we since we have 
    a very skewed distribution, we might end up with infrequent senses being
    overridden. '''
    
    def _apply_label_propagation_model(self, contexts, labels, affinity):
        label_prop_model = semi_supervised.LabelSpreading(kernel=lambda a, b: affinity)
        label_prop_model.fit(contexts, labels)
        return label_prop_model.transduction_    
    
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
        