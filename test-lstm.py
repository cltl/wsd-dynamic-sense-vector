import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    sentences = ['The latest from Congress , where Jeff Sessions denied Russia collusion in his Senate testimony'.split(),
                 'In my 29 years of being a firefighter , I have never , ever seen anything of this scale'.split()]
    
    model_path = 'output/lstm-wsd-small'
    vocab = np.load('output/gigaword.1m-sents.index.pkl')
    target_id = vocab['<target>']
    sentences_as_ids = [[vocab[w] for w in s] for s in sentences]
    sentences_as_ids[0][3] = sentences_as_ids[1][3] = target_id
    
    with tf.Session() as sess:  # your session object
        saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
        saver.restore(sess, model_path)
        predicted_context_embs = sess.graph.get_tensor_by_name('Model/predicted_context_embs:0')
        x = sess.graph.get_tensor_by_name('Model/x:0')
        
        print(sess.run(predicted_context_embs, {x: [sentences_as_ids[0]]}))
        print(sess.run(predicted_context_embs, {x: [sentences_as_ids[1]]}))
        