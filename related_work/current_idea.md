## Motivation

**low performance on LFS** [insert references]

**fine-grained senses**: related work [supersense embeddings, insert references] suggest that WSD system perform better on coarse-grained senses than fine-grained, because:
* there is more training data on a more coarse-grained level
* fine-grained is some much considered too fine-grained or not understandable

**lack of training data for supervised approaches**: although supervised approaches [insert references] usually beat unsupervised ones, they do often suffer from lack of training data.

## Terminology

**asymmetrical entailment**
Since the hyponymy relation is expressed as *asymmetrical entailment*, the hypernym entails the hyponym, e.g. *an apple entails that it's a fruit**.

**sense groups**: several approaches have attempted to make wordnet less fine-grained [insert references].

**highest non least common subsumer**: It is the highest synset under a least common subsumer (the most specific common ancestor of two concepts found in a given ontology). Hence, these are all the synsets that entail from a hyponym that are not shared across synsets.

## Methodology

### input
(automatically) sense-annotated data, i.e sentences in which expressions have been tagged with (WordNet | BabelNet) synsets.

### sense groups
The is-a relation in WordNet groups synsets hierarchically.

### training data enrichment
for each sense-annotated expression in a sentence,
we duplicate the instance with its  **highest non least common subsumer**.

### embedding training
using word2vec, we train both word and sense embeddings in the same space (hence also for the **highest non least common subsumer**).

### sense assignment
The chosen synset is the synset of the which the **highest non least common subsumer** has the highest cosine similarity with the sentence embedding.

possibly, we can experiment with the S2C setting of chen2014unified
