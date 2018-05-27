import tensorflow as tf
import numpy as np
import generate_hdn_datasets
from block_timer.timer import Timer
import pandas as pd
from model import HDNModel, LSTMLanguageModel
from tqdm import tqdm
from version import version

model_path = 'output/model-h2048p512/lstm-wsd-gigaword-google'
vocab_path = generate_hdn_datasets.word_vocab_path
gigaword_path_pattern = generate_hdn_datasets.inp_pattern
hdn_path_pattern = 'output/gigaword-hdn-%s.2018-05-18-f48a06c.pkl'

def stratified_subsampling(indices, buffer, n=100):
    tqdm.pandas(desc="Extracting target words", unit="word")
    get_word = lambda row: buffer[row["sent_start"]
                                  :row["sent_stop"]][row["word_index"]]
    indices['word'] = indices.progress_apply(get_word, axis=1)
    tqdm.pandas(desc="Stratified subsampling", unit="group")
    return (indices.groupby('word', group_keys=False)
            .progress_apply(lambda x: x.sample(min(len(x), n))))

if __name__ == '__main__':
    out_path = 'output/monosemous-context-embeddings.%s.npz' %version
    
    with Timer('Read Gigaword training set from %s' %(gigaword_path_pattern %'train')):
        buffer_train = np.load(gigaword_path_pattern %'train')['buffer']
    with Timer('Read HDN training indices from %s' %(hdn_path_pattern %'train')):
        hdn_train = pd.read_pickle(hdn_path_pattern %'train')
    hdn_train = stratified_subsampling(hdn_train, buffer_train, n=100)
    # make later processing more efficient
    hdn_train = hdn_train.sort_values(by='sent_len', axis='index')
    vocab = np.load(vocab_path)
    
    mono_words, mono_hdn_lists, mono_embs = [], [], []
    with tf.Session() as sess:
        lm = LSTMLanguageModel(sess, model_path)
        batches = HDNModel.gen_batches((buffer_train, hdn_train), 32000, vocab)
        batches = tqdm(batches, desc="Generating context embeddings", unit="batch")
        for x, lens, candidates, _, y in batches:
            context_embeddings = lm.get_embeddings_batch(sess, x, lens)
            mono_words.append(y)
            mono_hdn_lists.append(candidates)
            mono_embs.append(context_embeddings)
    mono_words = np.concatenate(mono_words)
    mono_hdn_lists = np.concatenate(mono_hdn_lists)
    mono_embs = np.vstack(mono_embs)
    
    np.savez(out_path, 
             mono_words=mono_words, 
             mono_hdn_lists=mono_hdn_lists,
             mono_embs=mono_embs)
    print('Results written to %s' %out_path)