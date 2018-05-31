## Setup

    pip3 install --user beautifulsoup4 sklearn testfixtures unittest2 pyemd morfessor
    pip3 install --user pandas==0.20.3
    pip3 install --user spacy
    pip3 install --user lxml
    pip3 install --user numpy
    pip3 install --user scipy
    pip3 install --user https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.1/en_core_web_md-1.2.1.tar.gz
    python3 -m spacy link en_core_web_md en_default
    pip3 install --user nltk
    printf 'import nltk; nltk.download("wordnet")' | python3
    pip3 install --user tensorflow-gpu
    pip3 install --user docopt

## Requirements
Using slurm version slurm 17.02.2, we ran our experiments using `cuda80` and Python version 3.5.2.

## git checkout to correct commit
 
Please run `git checkout 50b898e`
 
## Downloading resources

Please run `bash install.sh`

## Input data LSTM

Please run `bash convert_all_sense_annotations.sh`

## Running experiments

Please run `bash all_lp_jobs.sh`
Please run `bash evaluate_in_parallel.sh` with the correct arguments.
Please run the bash script to see what these arguments are.

## Result tables
Please run `python official_results.py`
The tables can be found in a folder called `paper_tables`.


