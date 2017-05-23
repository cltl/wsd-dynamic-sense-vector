
## Setup 

```
pip3 install --user beautifulsoup4 sklearn testfixtures unittest2 pyemd morfessor
pip3 install --user spacy 
pip3 install --user https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.1/en_core_web_md-1.2.1.tar.gz
python3 -m spacy link en_core_web_md en_default
```

Put a link to your copy of Gigaword 5th edition

```
ln -s /path/to/gigaword/5ed data/gigaword
```

Install the modified version of gensim in order to train sense embeddings.

```
./install-gensim-modified.sh
```

Make sure you have some Java JDK installed.

### Specific instructions for DAS-5

If you get error `No module named 'pip'` while importing spacy, you might want to log in to one of the compute nodes and install pip.
For example:

```
ssh node057
easy_install-3.4 --user pip
python3 -c 'import pip'
exit
```

## Stesps

1. Create a word2vec model from Gigaword corpus: `sbatch word2vec_gigaword.job`
2. Extract BabelNet-to-WordNet mappings: `./extract-bn-wn-mappings.sh`
3. Convert disambiguated Wikipedia?
4. Train sense embeddings: `./train-sense-embeddings.sh`