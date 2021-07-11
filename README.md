## A Deep Dive into Word Sense Disambiguation with LSTM

This package contains the code replicate the experiments from:

```xml
@InProceedings{C18-1030,
  author = 	"Le, Minh
		and Postma, Marten
		and Urbani, Jacopo
		and Vossen, Piek",
  title = 	"A Deep Dive into Word Sense Disambiguation with LSTM",
  booktitle = 	"Proceedings of the 27th International Conference on Computational Linguistics",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"354--365",
  location = 	"Santa Fe, New Mexico, USA",
  url = 	"http://aclweb.org/anthology/C18-1030"
}
```
## Demo
For a demo, we refer to [here](https://github.com/cltl/LSTM-WSD)


## Setup

    pip3 install --user beautifulsoup4 sklearn testfixtures unittest2 pyemd morfessor
    pip3 install --user pandas==0.20.3 seaborn==0.8.1
    pip3 install --user spacy
    pip3 install --user https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.1/en_core_web_md-1.2.1.tar.gz
    python3 -m spacy link en_core_web_md en_default
    pip3 install --user nltk
    python3 -m nltk.downloader wordnet
    pip3 install --user tensorflow-gpu
    pip3 install --user docopt

Put a link to your copy of Gigaword 5th edition

    ln -s /path/to/gigaword/5ed data/gigaword

Install the modified version of gensim in order to train sense embeddings.

    ./install-gensim-modified.sh

Make sure you have a Java JDK installed.

## Training models

The scripts you will find in the repo are meant to capture exactly what how we carried out our experiments, i.e., like a lab log. These experiments are very computing intensive -- we have run different parts of them in 3 different high-performance clusters -- so anyone trying to reproduce them likely needs to arrange their own supercomputer. As we don't have control over what computing environment people will use, we can't guarantee that the scripts can run as-is.

Requirements: You'll need at least 64 GB of RAM to run the preparation script.

You don't need access to Dutch DAS-5 or Cartesius to run these steps.
The `*.job` files are bash script that you could run on any Unix machine.

### Reproduce variation/stability experiments

These experiments measure how much the performance is affected by the randomness
in training. Basically, we train smaller models many times, each time with
a different (but fixed) random seed.

1. Pre-process GigaWord into plain text: `git checkout 694cb4d && sbatch process-gigaword.job`
2. More preprocessing to make binary files: `git checkout a453bc1 && sbatch cartesius/prepare-lstm-wsd.job`
0. `git checkout ce8a024`. Run at the same time: `sbatch cartesius/exp-variation1.job` and `cartesius/sbatch exp-variation2.job`
0. `git checkout a74bda6`. Preprocess to make binary files (the format is slightly different from the previous version): `sbatch cartesius/prepare-lstm-wsd.job`
3. `git checkout e93fdb2`. Run `cartesius/exp-h256p64.sh` (which calls `sbatch`)
2. When everything finishes, do `git checkout 42bc700` and run `sbatch cartesius/exp-variation-score.job`

### Reproduce (training speed) optimization experiment

1. Pre-process GigaWord into plain text (if you haven't done so): `git checkout a74bda6 && sbatch process-gigaword.job`
0. `git checkout a74bda6`
2. More preprocessing to make binary files: `sbatch cartesius/prepare-lstm-wsd.job`
3. `git checkout e93fdb2`
4. Run in parallel: `sbatch cartesius/exp-optimization{i}.job` where i=1,2,3,4

### Data size experiment

1. Pre-process GigaWord into plain text (if you haven't done so): `git checkout 694cb4d && sbatch process-gigaword.job`
0. `git checkout a74bda6`
2. More preprocessing to make binary files: `sbatch cartesius/prepare-lstm-wsd.job`
3. `git checkout 4e4a04a`
4. Run `sbatch cartesius/exp-data-size.job {i}` with i="01",10,25,50,75

### Model size experiements

Notice that there was uncertainty about the real version that produce h2048p512
and h512p128, see `difference-edited.txt` for a comparison with a recent version.

1. h=2048, p=512: `git checkout 354acc1cfdd542142490afe40447cb6f40d2fd7c && ./train-lstm-wsd-full-data-google-model.job`
2. h=512, p=128: `git checkout 354acc1cfdd542142490afe40447cb6f40d2fd7c && ./train-lstm-wsd-full-data-large-model.job`
3. h=512, p=64: see `exp-h256p64.sh` in "stability" section
4. h=100, p=10: see `exp-variation*.job` in "stability" section

### Hyperparameter tuning for label propagation

2. `git checkout 0448586`
3. `das5/exp-hyperp-label-propagation.sh` (which calls `sbatch`)

## Evaluating models

See [evaluate/README.md](evaluate/README.md).

## Specific instructions for DAS-5

If you get error `No module named 'pip'` while importing spacy, you might want
to log in to one of the compute nodes and install pip.
For example:

    ssh node057
    easy_install-3.4 --user pip
    python3 -c 'import pip'
    exit

If you get `No module named 'UserString'` while loading a Word2vec model from
disk, probably it is caused by the difference between compute node and login
node. You might ssh to one of the compute nodes to continue your work.

## Known issues

1. The reported results were produced using a model that didn't use `<eos>`
(end of sentence) tokens, different from Yuan et al. We added `<eos>`
in a later version.
2. The models were trained on sentences that were accidentally prepended
with their length (e.g. "24 Under the settlements , including Georgia 's ,
Liggett agreed to put the warning ' ' smoking is addictive '' on its packs ."),
this likely decreases the performance a bit.
3. On line 110 of the file [evaluate/test-lstm_v2.py](https://github.com/cltl/wsd-dynamic-sense-vector/blob/c2ee1d90aa06b4bd854cdf421f3f7f235cb45157/evaluate/test-lstm_v2.py#L110), **<unkn>** should have been **<pad>**. We tried to rerun for one experiment with this change applied and found no difference in the results.
