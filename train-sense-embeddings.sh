python3 -u train-sense-embeddings.py output/gigaword data/sample-disambiguated-text.txt output/sample-sense-embeddings
python3 -u train-sense-embeddings.py output/gigaword data/disambiguated-wikipedia-wordnet/hdn output/hdn
python3 -u train-sense-embeddings.py output/gigaword data/disambiguated-wikipedia-wordnet/synset output/synset