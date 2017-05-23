import gensim
import sys
import logging 
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec

if __name__ == '__main__':
    pretrained_model_path = sys.argv[1]
    sentences = LineSentence(sys.argv[2]) 
    out_path = sys.argv[3]

    # so that gensim will print something nice to the standard output
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec.load(pretrained_model_path)
    model.min_count = 1
    model.build_vocab(sentences, sense_delimiter='---', update=True)
    model.train(sentences, sense_delimiter='---', 
                total_examples=model.corpus_count, epochs=10)
    model.save(out_path)
    model.wv.accuracy('data/questions-words.txt')
    model.wv.evaluate_word_pairs('data/wordsim353_combined.tab')
