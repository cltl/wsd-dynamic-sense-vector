## Setup

    pip3 install --user beautifulsoup4 sklearn testfixtures unittest2 pyemd morfessor
    pip3 install --user pandas==0.20.3
    pip3 install --user spacy
    pip3 install --user https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.1/en_core_web_md-1.2.1.tar.gz
    python3 -m spacy link en_core_web_md en_default
    pip3 install --user nltk
    printf 'import nltk; nltk.download("wordnet")' | python3
    pip3 install --user tensorflow-gpu
    pip3 install --user docopt


## Downloading resources

Please run `bash install.sh`

## Input data LSTM

Please run `bash convert_all_sense_annotations.sh`

## Running experiments

Please run `bash table_2.sh`

This will create in two folders:
* experiments_mfs_fallback
* experiments_no_mfs_fallback

Each folder wil have six folder, each containing information about one experiment. Each folder contains a file called result.json with the results.
