import os
import gzip
from bs4 import BeautifulSoup
import spacy
from configs import gigaword_path, preprocessed_gigaword_path
import codecs
from utils import progress
from version import version
import sys

def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser)

nlp = spacy.load('en_default', create_pipeline=custom_pipeline)

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
    for doc in nlp.pipe(paragraphs, batch_size=10000):
        for sent in doc.sents:
            yield [str(tok).strip() for tok in sent]


# example_file = 'data/gigaword/gigaword_eng_5_d1/data/afp_eng/afp_eng_200112.gz'

if __name__ == '__main__':
    dir_ = os.path.join('preprocessed-data', version)
    os.makedirs(dir_, exist_ok=True)
    preprocessed_gigaword_path = os.path.join(dir_, 'gigaword.txt')
    sys.stderr.write('Writing to %s\n' %preprocessed_gigaword_path)
    with codecs.open(preprocessed_gigaword_path, 'w', 'utf-8') as f:
        paths = list(iter_files(gigaword_path))
        paths.sort() # remove difference between machines
        paths = progress(paths, ticks=1, label='files', max_=len(paths))
        for sent in iter_sents(iter_paragraphs(paths)):
            for tok in sent:
                f.write(tok)
                f.write(' ')
            f.write('\n')
