

def load_tensors(sess):
    x = sess.graph.get_tensor_by_name('Model_1/x:0')
    predicted_context_embs = sess.graph.get_tensor_by_name('Model_1/predicted_context_embs:0')
    lens = sess.graph.get_tensor_by_name('Model_1/lens:0')
    
    return x, predicted_context_embs, lens
