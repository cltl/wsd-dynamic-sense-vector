from configs import preprocessed_gigaword_path, preprocessed_data_dir
from collections import Counter
from nltk.stem import WordNetLemmatizer
import codecs
import os

if __name__ == '__main__':
    cutoff_freq = 20
    token_count = Counter()
    lemma_count = Counter()
    wordnet_lemmatizer = WordNetLemmatizer()
    with codecs.open(preprocessed_gigaword_path, 'r', 'utf-8') as f:
        for line_no, line in enumerate(f):
            for tok in line.split():
                tok = tok.strip()
                token_count[tok] += 1
                lemma = wordnet_lemmatizer.lemmatize(tok.lower())
                lemma_count[lemma] += 1
            if (line_no+1) % 100000 == 0:
                print(line_no+1)
#             if line_no >= 1000: break # for debugging
    with codecs.open(os.path.join(preprocessed_data_dir, 'token.lst'), 'w', 'utf-8') as f:
        for tok in token_count:
            if token_count[tok] >= cutoff_freq:
                f.write('%s\t%d\n' %(tok, token_count[tok]))
    with codecs.open(os.path.join(preprocessed_data_dir, 'lemma.lst'), 'w', 'utf-8') as f:
        for lemma in lemma_count:
            if lemma_count[lemma] >= cutoff_freq:
                f.write('%s\t%d\n' %(lemma, lemma_count[lemma]))
                