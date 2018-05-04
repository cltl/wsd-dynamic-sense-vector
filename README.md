
## Setup

    pip3 install -r requirements.txt
    python3 -m spacy link en_core_web_md en_default
    printf 'import nltk; nltk.download("wordnet")' | python3

Put a link to your copy of Gigaword 5th edition

    ln -s /path/to/gigaword/5ed data/gigaword

Make sure you have a Java JDK installed.

### Specific instructions for DAS-5

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

## Steps

### Preprocessing

Run `das5/preprocess.job`.

### Reproduction paper

Requirements: You'll need at least 64 GB of RAM to run the preparation script. 

#### Convert annotated corpora into input format LSTM

0. cd script
1. bash convert_for_paper.sh 
The bash file calls the python script sense_annotations2lstm_format.py, which converts the sense annotations into the format needed to train the embeddings at the preferred granularity level (sensekey or synset).

#### label propagation

After running the `sense_annotations2lstm_format.py` with OMSTI as corpus, the output of that run needs to be provided as an argument to `pwgc_to_ulm.py`.
This creates a development set for the label propagation:
a) annotated corpus: pwgc
b) unannotated corpus: omsti

#### Model size experiements

Notice that there was uncertainty about the real version that produce h2048p512
and h512p128, see `difference-edited.txt` for a comparison with a recent version.

1. h=2048, p=512: `git checkout 354acc1cfdd542142490afe40447cb6f40d2fd7c && ./train-lstm-wsd-full-data-google-model.job`
2. h=512, p=128: `git checkout 354acc1cfdd542142490afe40447cb6f40d2fd7c && ./train-lstm-wsd-full-data-large-model.job`
3. h=512, p=64: see `exp-h256p64.sh` in "stability" section
4. h=100, p=10: see `exp-variation*.job` in "stability" section

#### Reproduce variation/stability experiments

These experiments measure how much the performance is affected by the randomness
in training. Basically, we train smaller models many times, each time with 
a different (but fixed) random seed.

1. Pre-process GigaWord into plain text: `git checkout 694cb4d && sbatch process-gigaword.job`
2. More preprocessing to make binary files: `git checkout a453bc1 && sbatch cartesius/prepare-lstm-wsd.job`
0. `git checkout ce8a024`
1. Run at the same time: `sbatch cartesius/exp-variation1.job` and `cartesius/sbatch exp-variation2.job`
0. `git checkout a74bda6`
2. Preprocess to make binary files (the format is slightly different from the previous version): `sbatch cartesius/prepare-lstm-wsd.job`
3. `git checkout e93fdb2`
4. Run `cartesius/exp-h256p64.sh` (which calls `sbatch`)
2. When everything finishes, do `git checkout 42bc700` 
3. Run `sbatch cartesius/exp-variation-score.job`

#### Reproduce (training speed) optimization experiment

1. Pre-process GigaWord into plain text (if you haven't done so): `git checkout a74bda6 && sbatch process-gigaword.job`
0. `git checkout a74bda6`
2. More preprocessing to make binary files: `sbatch cartesius/prepare-lstm-wsd.job`
3. `git checkout e93fdb2`
4. Run in parallel: `sbatch cartesius/exp-optimization{i}.job` where i=1,2,3,4

#### Data size experiment

1. Pre-process GigaWord into plain text (if you haven't done so): `git checkout 694cb4d && sbatch process-gigaword.job`
0. `git checkout a74bda6`
2. More preprocessing to make binary files: `sbatch cartesius/prepare-lstm-wsd.job`
3. `git checkout 4e4a04a`
4. Run `sbatch cartesius/exp-data-size.job {i}` with i="01",10,25,50,75

#### Hyperparameter tuning for label propagation

1. TODO @Marten: how to create debug.lp and debug.lp.gold
2. `git checkout 0448586`
3. `das5/exp-hyperp-label-propagation.sh` (which calls `sbatch`)
