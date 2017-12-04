import numpy as np
from time import time
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from scipy.sparse.csr import csr_matrix
import os
import sys
from sklearn import semi_supervised
import math

class RBF(object):
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, X, Y):
        distances = euclidean_distances(X, Y)
        return math.exp(-self.gamma*distances*distances)
    
def expander(X, Y): 
    ''' The similarity function used by Ravi and Diao (2015, pp 524) 
    Ravi, S., & Diao, Q. (2015). Large Scale Distributed Semi-Supervised 
    Learning Using Streaming Approximation. arXiv:1512.01752 [Cs], 51.
    '''
    return X.dot(Y.T)

class LabelPropagation(object):
    
    def __init__(self, sess, vocab_path, model_path, batch_size, sim_func=expander):
        self.sess = sess
        self.batch_size = batch_size
        self.sim_func = sim_func
        self.vocab = np.load(vocab_path)
        import tensorflow as tf
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
        self.predicting_elapsed_sec = 0
        self.adding_edges_elapsed_sec = 0
        self.num_low_degree_vertices = 0
        self.num_all_vertices = 0
        self.num_added_edges = 0
        self.num_total_edges = 0
        self.debugging = False
        
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
        
    def _apply_label_propagation_model(self, contexts, labels):
        label_prop_model = semi_supervised.LabelPropagation(kernel=self.affinity_func)
        label_prop_model.fit(contexts, labels)
        return label_prop_model.transduction_
            
    def affinity_func(self, X1, X2):
        assert X1 is X2, "Unsupported case: two different sets of vectors"
        contexts = X1
        num_examples = len(contexts)
        sims = self.sim_func(contexts)
        sorted_indices = np.dstack(np.unravel_index(np.argsort(-sims.ravel()), 
                                                    sims.shape))[0]
        sorted_indices = [(u, v) for u, v in sorted_indices if u < v] # keep only one of two equivalent pairs
        num_most_similar_pairs = int(num_examples*(num_examples-1)*(1-self.similarity_threshold))
        selected_pairs = set(pair for pair in sorted_indices[:num_most_similar_pairs])
        sorted_within_row_indices = np.argsort(-sims)
        # add edges to low-connectivity vertices
        degree = Counter()
        degree.update(u for u, _ in selected_pairs)
        degree.update(v for _, v in selected_pairs)
        adding_edges_start_sec = time()
        for v in range(num_examples):
            if degree[v] < self.minimum_vertex_degree: 
                self.num_low_degree_vertices += 1
                for idx in sorted_within_row_indices[v]:
                    if degree[v] >= self.minimum_vertex_degree: break
                    if (v, idx) not in selected_pairs and (idx, v) not in selected_pairs:
                        selected_pairs.add((v, idx))
                        degree[v] += 1
                        degree[idx] += 1
                        self.num_added_edges += 1
        self.adding_edges_elapsed_sec += (time() - adding_edges_start_sec)
        self.num_total_edges += len(selected_pairs) 
        # make the matrix
        for u, v in selected_pairs:
            print('\t%d, %d --> %f' %(u, v, self.sim_func(contexts[u], contexts[v])))
        sims, rows, cols = zip(*[(sims[u,v], u,v) for u,v in selected_pairs] +
                                [(sims[v,u], v,u) for u,v in selected_pairs])
        return csr_matrix((sims, (rows, cols)), shape=(num_examples,num_examples))
        
    def _pad(self, list_of_lists, pad_id):
        max_len = max(len(s) for s in list_of_lists)
        for s in list_of_lists:
            while len(s) < max_len:
                s.append(pad_id)
        return list_of_lists
        
    def _run_lstm(self, converted_data):
        print('Running LSTM...')
        # create one big matrix of LSTM input
        lstm_input = []
        target_id = self.vocab['<target>']
        pad_id = self.vocab['<pad>']
        for lemma in converted_data:
            for _, sentence_tokens, target_index in converted_data[lemma]:
                sentence_as_ids = [self.vocab.get(w) or self.vocab['<unkn>'] 
                                   for w in sentence_tokens]
                sentence_as_ids[target_index] = target_id
                lstm_input.append(sentence_as_ids)
        lens = np.array([len(s) for s in lstm_input])
        self._pad(lstm_input, pad_id)
        lstm_input = np.array(lstm_input)
        # process the input in batches
        lstm_output = []
        for batch_no, batch_start in enumerate(range(0, len(lstm_input), self.batch_size)):
            batch_end = min(len(lstm_input), batch_start+self.batch_size)
            lstm_output.append(self.sess.run(self.predicted_context_embs, 
                                             {self.x: lstm_input[batch_start:batch_end], 
                                              self.lens: lens[batch_start:batch_end]}))
            if (batch_no+1) % 100 == 0:
                print('Batch #%d...' %(batch_no+1))
        lstm_output = np.vstack(lstm_output)
        # unpack the output into a mapping {lemma --> contexts}
        lemma2contexts = {}
        start = 0
        for lemma in converted_data:
            lemma2contexts[lemma] = lstm_output[start:start+len(converted_data[lemma])]
        assert start == lstm_output.shape[0]
        print('Running LSTM... Done.')
        return lemma2contexts
        
    def predict(self, data):
        '''
        input data format: dict(lemma -> list((sense_id[str], sentence_tokens, target_index)))
        set sense_id to None for unlabeled instances 

        batch_size: number of sentences in a batch to be used as input for LSTM
        
        output format: dict(lemma -> list(sense_id)), the order in each list corresponds to the input
        '''
        start_sec = time()
        sense_counts = {}
        for lemma in data:
            sense_counts[lemma] = Counter()
            sense_counts[lemma].subtract(sense for sense, _, _ in data[lemma] 
                                         if sense is not None)
            self.num_all_vertices += len(data[lemma])
        converted_data, sense_ids = self._convert_sense_ids(data)
        lemma2context = self._run_lstm(converted_data)
        
        output = {}
        for lemma_no, (lemma, contexts) in enumerate(lemma2context.items()):
            if self.debugging and lemma_no >= 100: break # for debugging
            print("Lemma #%d of %d: %s" %(lemma_no, len(converted_data), lemma))
            labels = [sense for sense, _, _ in converted_data[lemma]]
            predicted_indices = self._apply_label_propagation_model(contexts, labels)
            output[lemma] = [sense_ids[index] for index in predicted_indices]
            sense_counts[lemma].update(output[lemma])
            if self.debugging: print(sense_counts[lemma].most_common())
            
        self.predicting_elapsed_sec += (time()-start_sec)
        return output
    
    def print_stats(self):
        print('Predicting time: %.2f min' %(self.predicting_elapsed_sec/60.0))
        print('Time for adding edges: %.2f min (%.2f%% of total time)' 
              %(self.adding_edges_elapsed_sec/60.0, 
                self.adding_edges_elapsed_sec*100.0/self.predicting_elapsed_sec))
        print('Number of vertices with low connectivity: %d (%.2f%% of all vertices)' 
              %(self.num_low_degree_vertices, self.num_low_degree_vertices*100.0/self.num_all_vertices))
        
    
class LabelSpreading(LabelPropagation):
    ''' Allowing the label properation algorithm to change the underlying gold
    annotations. It's designed to work with noisy data but we since we have 
    a very skewed distribution, we might end up with infrequent senses being
    overridden. '''
    
    def _apply_label_propagation_model(self, contexts, labels, affinity):
        label_prop_model = semi_supervised.LabelSpreading(kernel=lambda a, b: affinity)
        label_prop_model.fit(contexts, labels)
        return label_prop_model.transduction_    
    
class NearestNeighbor(LabelPropagation):
    ''' This is a baseline to evaluate label propagation models '''
    
    def predict(self, data):
        '''
        input data format: dict(lemma -> list((sense_id[str], sentence_tokens, target_index)))
        set sense_id to None for unlabeled instances 

        batch_size: number of sentences in a batch to be used as input for LSTM
        
        output format: dict(lemma -> list(sense_id)), the order in each list corresponds to the input
        '''
        converted_data, sense_ids = self._convert_sense_ids(data)
        lemma2context = self._run_lstm(converted_data)
        
        output = {}
        for lemma_no, (lemma, contexts) in enumerate(lemma2context.items()):
            if self.debugging and lemma_no >= 100: break # for debugging
            print("Lemma #%d of %d: %s" %(lemma_no, len(converted_data), lemma))
            d = converted_data[lemma]
            labeled_indices = [i for i in range(len(d)) if d[i] >= 0]
            unlabeled_indices = [i for i in range(len(d)) if d[i] < 0]
            
            labeled_contexts = contexts[labeled_indices]
            unlabeled_contexts = contexts[unlabeled_indices]
            sims = self.sim_func(unlabeled_contexts, labeled_contexts)
            most_similar_labeled_contexts = np.argsort(-sims)[:,0]
            predicted_indices = d.copy()
            for i, j in zip(unlabeled_indices, most_similar_labeled_contexts):
                predicted_indices[i] = d[labeled_indices[j]]
            output[lemma] = [sense_ids[index] for index in predicted_indices]
        return output
    