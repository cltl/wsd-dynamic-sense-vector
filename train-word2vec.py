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
    model = gensim.models.Word2Vec(sentences, workers=32)
    model.accuracy('data/questions-words.txt')
    model.save('output/gigaword.pkl')

