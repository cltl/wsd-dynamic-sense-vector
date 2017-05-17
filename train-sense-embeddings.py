import gensim
import sys
import logging 
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    pretrained_model_path = sys.argv[1]
    sentences = LineSentence(sys.argv[2]) # a memory-friendly iterator
    out_path = sys.argv[3]

    # so that gensim will print something nice to the standard output
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec()
    model.load(pretrained_model_path)
    model.build_vocab(sentences, sense_delimiter='---')
    model.train(sentences, workers=32, sense_delimiter='---')
    model.save(out_path)
    model.wv.accuracy('data/questions-words.txt')
    model.wv.evaluate_word_pairs('data/wordsim353_combined.tab')
