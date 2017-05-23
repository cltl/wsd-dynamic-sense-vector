import gensim
import sys
import logging 
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec

if __name__ == '__main__':
    sentences = LineSentence(sys.argv[1]) # a memory-friendly iterator
    out_path = sys.argv[2]
    
    # so that gensim will print something nice to the standard output
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(sentences, workers=32)
    model.save(out_path)
    model.wv.accuracy('data/questions-words.txt')
    model.wv.evaluate_word_pairs('data/wordsim353_combined.tab')
