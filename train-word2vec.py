import gensim
import sys

class MySentences(object):
    def __init__(self, path):
        self.path = path
 
    def __iter__(self):
        for line in open(self.path):
            yield line.split()

if __name__ == '__main__':
    sentences = MySentences(sys.argv[1]) # a memory-friendly iterator
    out_path = sys.argv[2]
    model = gensim.models.Word2Vec(sentences, workers=32)
    model.save(out_path)
    
    import logging # so that gensim will print something nice to the standard output
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model.wv.accuracy('data/questions-words.txt')
    model.wv.evaluate_word_pairs('data/wordsim353_combined.tab')
