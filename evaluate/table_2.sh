#!/usr/bin/env bash

model=resources/model-h2048p512/lstm-wsd-gigaword-google
vocab=gigaword-lstm-wsd.index.pkl

sbatch --time=10:00:00 evaluate_mfs_backoff.job $model $vocab experiments_mfs_fallback
sbatch --time=10:00:00 evaluate_no_mfs_backoff.job $model $vocab experiments_no_mfs_fallback
