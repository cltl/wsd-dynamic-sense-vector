from gensim.models import Word2Vec
import sys

data_path = 'output/sense_embeddings'
    
def examine_synset(s):
    print('Nearest neighbors of %s:' %s)
    for word, score in model.wv.most_similar(s):
        print('\t%s\t%.3f' %(word, score))
    print('todo: look up lemma in WordNet')
    print()

if __name__ == '__main__':
    sys.stdout.write('Loading from %s... ' %data_path)
    sys.stdout.flush()
    model = Word2Vec.load(data_path)
    print('Done!')
    try:
        while True:
            print('Enter a WordNet synset (Enter for default, q for quit)')
            sys.stdout.write('>> ')
            sys.stdout.flush()
            cmd = sys.stdin.readline().strip()
            if cmd == 'q':
                break
            elif cmd == '':
                for s in ('party.03', 'socialist.01', 'meaning.02'):
                    examine_synset(s)
            else:
                examine_synset(cmd)
    except KeyboardInterrupt:
        print()
    print('Goodbye!')