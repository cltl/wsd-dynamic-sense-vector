import gensim
import sys
import logging 
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import re
import itertools
import os

class IterableChain(object):
    def __init__(self, iterables):
        self.iterables = iterables
    def __iter__(self):
        return itertools.chain(*self.iterables)

if __name__ == '__main__':
    pretrained_model_path = sys.argv[1]
    in_path = sys.argv[2]
    if os.path.isfile(in_path):
        sentences = LineSentence(in_path)
    else:
        iters = []
        for root, dirs, fnames in os.walk(in_path):
            for fname in fnames:
                if re.search(r'\.txt\.gz$', fname):
                    child_path = os.path.join(root, fname)
                    iters.append(LineSentence(child_path))
        sentences = IterableChain(iters)
    out_path = sys.argv[3]

    # so that gensim will print something nice to the standard output
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec.load(pretrained_model_path)
#     model.workers = 1 # for debugging
    model.min_count = 1
    model.build_vocab(sentences, sense_delimiter='---', update=True)
    model.train(sentences, sense_delimiter='---', 
                total_examples=model.corpus_count, epochs=10)
    model.save(out_path)
