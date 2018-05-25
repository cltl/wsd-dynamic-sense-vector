import tensorflow as tf
from evaluate import tensor_utils
import numpy as np
import generate_hdn_datasets
from block_timer.timer import Timer
import pandas as pd
from model import HDNModel
from tqdm import tqdm

if __name__ == '__main__':
    model_path = 'output/model-h2048p512/lstm-wsd-gigaword-google'
    vocab_path = 'output/model-h2048p512/gigaword-lstm-wsd.index.pkl'
    gigaword_path_pattern = generate_hdn_datasets.inp_pattern
    hdn_path_pattern = 'output/gigaword-hdn-%s.2018-05-18-f48a06c.pkl'
    with Timer('Read Gigaword training set from %s' %(gigaword_path_pattern %'train')):
        buffer_train = np.load(gigaword_path_pattern %'train')['buffer']
    with Timer('Read HDN training indices from %s' %(hdn_path_pattern %'train')):
        hdn_train = pd.read_pickle(hdn_path_pattern %'train')

    vocab = np.load(vocab_path)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
        saver.restore(sess, model_path)
        x, predicted_context_embs, lens = tensor_utils.load_tensors(sess)

        sentence_tokens = 'I studied computer science'.lower().split()
        sentence_as_ids = [vocab.get(w) or vocab['<unkn>'] for w in sentence_tokens]

        batches = HDNModel.gen_batches((buffer_train, hdn_train), 32000)
        batches = tqdm(batches, desc="Generating context embeddings", unit="batch")
        for x, lens, _, _ in batches:
            feed_dict = {x: [sentence_as_ids], lens: [len(sentence_as_ids)]}
            target_embeddings = sess.run(predicted_context_embs, feed_dict)
            