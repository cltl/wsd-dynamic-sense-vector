import tensorflow as tf
from evaluate import tensor_utils
import numpy as np
import generate_hdn_datasets
from block_timer.timer import Timer
import pandas as pd
from model import HDNModel
from tqdm import tqdm
from version import version

model_path = 'output/model-h2048p512/lstm-wsd-gigaword-google'
vocab_path = generate_hdn_datasets.word_vocab_path
gigaword_path_pattern = generate_hdn_datasets.inp_pattern
hdn_path_pattern = 'output/gigaword-hdn-%s.2018-05-18-f48a06c.pkl'

if __name__ == '__main__':
    out_path = 'output/monosemous-context-embeddings.%s.npz' %version
    
    with Timer('Read Gigaword training set from %s' %(gigaword_path_pattern %'train')):
        buffer_train = np.load(gigaword_path_pattern %'train')['buffer']
    with Timer('Read HDN training indices from %s' %(hdn_path_pattern %'train')):
        hdn_train = pd.read_pickle(hdn_path_pattern %'train')
    vocab = np.load(vocab_path)
    
    mono_words, mono_embs = [], []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
        saver.restore(sess, model_path)
        x, predicted_context_embs, lens = tensor_utils.load_tensors(sess)

        batches = HDNModel.gen_batches((buffer_train, hdn_train), 32000, vocab)
        batches = tqdm(batches, desc="Generating context embeddings", unit="batch")
        for x_val, lens_val, _, _, y in batches:
            feed_dict = {x: x_val, lens: lens_val}
            context_embeddings = sess.run(predicted_context_embs, feed_dict)
            mono_words.append(y)
            mono_embs.append(context_embeddings)
    mono_words = np.vstack(mono_words)
    mono_embs = np.vstack(mono_embs)
    
    np.savez(out_path, mono_words=mono_words, mono_embs=mono_embs)