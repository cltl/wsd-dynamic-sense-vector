import os
import gzip
from bs4 import BeautifulSoup
import spacy
nlp = spacy.load('en_default')
import sys

def iter_paragraphs(paths):
    for path in paths:
        with gzip.open(path) as f:
            content = f.read()
        soup = BeautifulSoup(content,'html.parser')
        paras = soup.find_all('p')
        for p in paras: yield p.text.strip()


def iter_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            if '.gz' in fname:
                yield os.path.join(root, fname)

def iter_sents(paragraphs):
    for i, doc in enumerate(nlp.pipe(paragraphs, batch_size=10000, n_threads=32)):
        assert isinstance(doc, spacy.tokens.doc.Doc) and doc.is_parsed
        for sent in doc.sents:
            yield [str(tok).strip() for tok in sent]
        if (i+1) % 10000 == 0:
            sys.stderr.write('%10d' %(i+1))
        if (i+1) % 100000 == 0:
            sys.stderr.write('\n')

gigaword_path = 'data/gigaword'
example_file = 'data/gigaword/gigaword_eng_5_d1/data/afp_eng/afp_eng_200112.gz'

if __name__ == '__main__':
    for sent in iter_sents(iter_paragraphs(iter_files(gigaword_path))):
        for tok in sent:
            sys.stdout.write(tok)
            sys.stdout.write(' ')
        sys.stdout.write('\n')
