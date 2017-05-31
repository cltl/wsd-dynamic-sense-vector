# Conversion [Babelfied Wikipedia](http://lcl.uniroma1.it/babelfied-wikipedia/)


### corpus
As explained in the [README](http://lcl.uniroma1.it/babelfied-wikipedia/files/README.txt), there are three types of annotations:
* **BABELFY** annotation provided by Babelfy
* **MCS** annotation provided by the most common sense (MCS) backoff strategy
* **HL**: hyperlink inside the Wikipedia page


### conversion
For each annotation, we first look for a:
* **HL** annotation
* [if not there] a **BABELFY** annotation

For each found annotation, we map it to
* **WordNet 3.0 identifier**: from BabelNet 3.7
* **Highest Discriminating Node**

In order to find the wordnet lemma, we also make use of the **MCS** annotation.

### Output

* folder **synset**: all training instances relevant for sem2013-aw
* folder **hdn**: all training instances relevant for sem2013-aw
