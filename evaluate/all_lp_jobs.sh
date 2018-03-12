#!/usr/bin/env bash


model=resources/model-h2048p512/lstm-wsd-gigaword-google
vocab=resources/model-h2048p512/gigaword-lstm-wsd.index.pkl
system=higher_level_annotations/se2-aw-framework-synset-30_semcor_mun.lp

algo="propagate"
sim="expander"
#sbatch --time=04:00:00 run_lp.job $system $model $vocab $algo $sim

algo="spread"
sim="expander"
sbatch --time=04:00:00 run_lp.job $system $model $vocab $algo $sim

algo="nearest"
sim="expander"
#sbatch --time=04:00:00 run_lp.job $system $model $vocab $algo $sim


system=higher_level_annotations/se13-aw-framework-synset-30_semcor_mun.lp
sbatch --time=04:00:00 run_lp.job $system $model $vocab $algo $sim
