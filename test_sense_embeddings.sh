#!/bin/bash

module load cuda80/toolkit
module load cuda80/blas
module load cuda80
module load cuDNN

model_path=/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/lstm-wsd-small
vocab_path=/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/gigaword.1m-sents-lstm-wsd.index.pkl
input_path=/var/scratch/mcpostma/wsd-dynamic-sense-vector/scripts/synset-semcor.txt
output_path=/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/sense_embeddings.bin
python3 /var/scratch/mcpostma/wsd-dynamic-sense-vector/test-lstm.py -m $model_path -v $vocab_path -i $input_path -o $output_path -t 100
