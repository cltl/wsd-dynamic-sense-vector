'''
This program takes Gigaword and turns it into a collection of sentences in a
big gzip'd file. Each sentence is placed in one line, tokens are separated by
a space.
'''

import os
import gzip
from bs4 import BeautifulSoup
from tqdm import tqdm
import spacy
from configs import gigaword_path
from version import version

def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser)

nlp = spacy.load('en_default', create_pipeline=custom_pipeline)


def iter_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if '.gz' in fname:
                yield os.path.join(root, fname)


def iter_paragraphs(paths):
    for path in paths:
        with gzip.open(path) as f:
            content = f.read()
        soup = BeautifulSoup(content,'html.parser')
        paras = soup.find_all('p')
        for p in paras: yield p.text.strip()


def iter_sents(paragraphs):
    for doc in nlp.pipe(paragraphs, batch_size=10000):
        for sent in doc.sents:
            yield [str(tok).strip() for tok in sent]


def run():
    out_path = os.path.join('output', 'gigaword.%s.txt.gz' %version)
    if os.path.exists(out_path):
        print('Found the output at %s, skipped.' %out_path)
    else:
        temp_path = out_path + '.tmp'
        print('Writing to %s' %temp_path)
        with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
            # sort to remove difference between machines
            paths = sorted(iter_files(gigaword_path))
            paths = tqdm(paths, desc='Preprocessing GigaWord', unit='file')
            for sent in iter_sents(iter_paragraphs(paths)):
                for tok in sent:
                    f.write(tok)
                    f.write(' ')
                f.write('\n')
        os.rename(temp_path, out_path)
        print('Successful! Result saved to %s' %out_path)
        

if __name__ == '__main__':
    run()