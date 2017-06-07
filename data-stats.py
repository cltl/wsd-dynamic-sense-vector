import os
import re
import gzip
from collections import Counter


if __name__ == '__main__':
    disambiguated_wikipedia_dirs = ('data/disambiguated-wikipedia-wordnet/hdn',
                                    'data/disambiguated-wikipedia-wordnet/synset')
    for dir in disambiguated_wikipedia_dirs:
        c = Counter()
        paths = (os.path.join(root, fname)
                 for root, subdirs, fnames in os.walk(dir)
                 for fname in fnames
                 if re.search(r'\.txt.gz', fname))
        i = 0
        for path in paths:
            with gzip.open(path, 'rt') as f:
                for line in f:
                    for wn_offset in re.findall(r'eng-30-(\d+-\w)', line):
                        c[wn_offset] += 1
                    i += 1
                    if i % 1000000 == 0:
                        print(i)
        with open('output/synset-count-%s.csv' %os.path.basename(dir), 'wt') as f2:
            for synset in c:
                f2.write('%d\n' %c[synset])
        print(c.most_common(100))