import numpy as np

def load_tensors(sess):
    x = sess.graph.get_tensor_by_name('Model_1/x:0')
    predicted_context_embs = sess.graph.get_tensor_by_name('Model_1/predicted_context_embs:0')
    lens = sess.graph.get_tensor_by_name('Model_1/lens:0')
    
    return x, predicted_context_embs, lens
            
def pad(sents, max_len, pad_id, eos_id):
    if eos_id is not None: 
        max_len += 1
    arr = np.empty((len(sents), max_len), dtype=np.int32)
    arr.fill(pad_id)
    for i, s in enumerate(sents):
        arr[i, :len(s)] = s
        if eos_id is not None:
            arr[i, len(s)] = eos_id
    return arr
