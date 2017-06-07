from gensim.models import Word2Vec
import sys
import re
from nltk.corpus import wordnet as wn

def examine_synset(s):
    m = re.search(r'^(?:eng-30-)?(\d+)-(\w)', s)
    if m:
        ss = wn._synset_from_pos_and_offset(m.group(2), int(m.group(1)))
        print(ss)
        if not re.search(r'^eng-30-', s):
            s = 'eng-30-' + s
    if s in model.wv.vocab:
        print('Nearest neighbors:')
        for word, score in model.wv.most_similar(s):
            print('\t%s\t%.3f' %(word, score))
    else:
        print('Not found in the vocabulary.')
    print()

if __name__ == '__main__':
    data_path = sys.argv[1]
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
                for s in ('00001930-n', '00033020-n', '00023100-n',
                          '00028270-n', '06613686-n', '09044862-n'):
                    print('--- Example: %s ---' %s)
                    examine_synset(s)
            else:
                examine_synset(cmd)
    except KeyboardInterrupt:
        print()
    print('Goodbye!')