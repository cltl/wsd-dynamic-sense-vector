#!/usr/bin/env bash
#BATCH --time=12:00:00
#SBATCH -C TitanX
#SBATCH --gres=gpu:1


#model=resources/model-h2048p512/lstm-wsd-gigaword-google
#vocab=resources/model-h2048p512/gigaword-lstm-wsd.index.pkl
#out=debug

if [ $# -eq 0 ];
then
    echo
    echo "Usage:                    : $0 model_path vocab_path out_dir mfs_fallback"
    echo
    echo
    echo "model_path                : path to model"
    echo "vocab_path                : path to .pkl file"
    echo "out_dir                   : output directory"
    echo "mfs_fallback              : mfs_fallback"
    echo ""
    echo "example of call for reproduction paper:"
    echo "bash evaluate_in_parallel.sh resources/model-h2048p512/lstm-wsd-gigaword-google resources/model-h2048p512/gigaword-lstm-wsd.index.pkl debug True"
    exit -1;
fi

out=$3
rm -rf $out && mkdir $out
sleep 10

use_case_strategy=False
use_number_strategy=False
mfs_fallback=True
batch_size=400
emb_setting=synset
eval_setting=synset
model=$1
vocab=$2
mfs_fallback=$4

#sbatch one_experiment.job $model $vocab "$out/synset-se13-semcor"  higher_level_annotations/synset-30_semcor synset False $mfs_fallback
#sbatch one_experiment.job $model $vocab "$out/synset-se13-semcor_omsti" higher_level_annotations/synset-30_semcor_mun synset False $mfs_fallback
#sbatch one_experiment.job $model $vocab "$out/synset-se13-semcor_omsti-lp" higher_level_annotations/synset-30_semcor_mun synset True
#sbatch one_experiment.job $model $vocab "$out/synset-se13-omsti"        higher_level_annotations/synset-30_mun synset False $mfs_fallback

#sbatch one_experiment.job $model $vocab "$out/sensekey-se13-semcor"       higher_level_annotations/sensekey-30_semcor sensekey False
#sbatch one_experiment.job $model $vocab "$out/sensekey-se13-semcor_omsti" higher_level_annotations/sensekey-30_semcor_mun sensekey False
#sbatch one_experiment.job $model $vocab "$out/sensekey-se13-omsti"        higher_level_annotations/sensekey-30_mun sensekey False

#sbatch one_experiment.job $model $vocab "$out/synset-se2-semcor"       higher_level_annotations/synset-171_semcor synset False $mfs_fallback
#sbatch one_experiment.job $model $vocab "$out/synset-se2-semcor_omsti" higher_level_annotations/synset-171_semcor_mun synset False $mfs_fallback
#sbatch one_experiment.job $model $vocab "$out/synset-se2-semcor_omsti-lp" higher_level_annotations/synset-171_semcor_mun synset True $mfs_fallback
#sbatch one_experiment.job $model $vocab "$out/synset-se2-omsti"        higher_level_annotations/synset-171_mun synset False $mfs_fallback

#sbatch one_experiment.job $model $vocab "$out/sensekey-se2-semcor"       higher_level_annotations/sensekey-171_semcor sensekey False
#sbatch one_experiment.job $model $vocab "$out/sensekey-se2-semcor_omsti" higher_level_annotations/sensekey-171_semcor_mun sensekey False
#sbatch one_experiment.job $model $vocab "$out/sensekey-se2-omsti"        higher_level_annotations/sensekey-171_mun sensekey False


## wsd framework experiments
command='sbatch' #change to bash when you are on a specific node
$command one_experiment.job $model $vocab "$out/synset-se2-framework-semcor" higher_level_annotations/se2-aw-framework-synset-30_semcor synset False $mfs_fallback
$command one_experiment.job $model $vocab "$out/synset-se2-framework-omsti"  higher_level_annotations/se2-aw-framework-synset-30_mun synset False $mfs_fallback
$command one_experiment.job $model $vocab "$out/synset-se2-framework-semcor_mun" higher_level_annotations/se2-aw-framework-synset-30_semcor_mun synset  False $mfs_fallback
$command one_experiment.job $model $vocab "$out/synset-se2-framework-semcor_mun-lp" higher_level_annotations/se2-aw-framework-synset-30_semcor_mun synset True $mfs_fallback

$command one_experiment.job $model $vocab "$out/sensekey-se2-framework-semcor" higher_level_annotations/se2-aw-framework-sensekey-30_semcor sensekey False $mfs_fallback
$command one_experiment.job $model $vocab "$out/sensekey-se2-framework-omsti"  higher_level_annotations/se2-aw-framework-sensekey-30_mun sensekey False $mfs_fallback
$command one_experiment.job $model $vocab "$out/sensekey-se2-framework-semcor_mun" higher_level_annotations/se2-aw-framework-sensekey-30_semcor_mun sensekey  False $mfs_fallback

$command one_experiment.job $model $vocab "$out/synset-se13-framework-semcor" higher_level_annotations/se13-aw-framework-synset-30_semcor synset False $mfs_fallback
$command one_experiment.job $model $vocab "$out/synset-se13-framework-omsti"  higher_level_annotations/se13-aw-framework-synset-30_mun synset False $mfs_fallback
$command one_experiment.job $model $vocab "$out/synset-se13-framework-semcor_mun" higher_level_annotations/se13-aw-framework-synset-30_semcor_mun synset  False $mfs_fallback
$command one_experiment.job $model $vocab "$out/synset-se13-framework-semcor_mun-lp" higher_level_annotations/se13-aw-framework-synset-30_semcor_mun synset True $mfs_fallback


$command one_experiment.job $model $vocab "$out/sensekey-se13-framework-semcor" higher_level_annotations/se13-aw-framework-sensekey-30_semcor sensekey False $mfs_fallback
$command one_experiment.job $model $vocab "$out/sensekey-se13-framework-omsti"  higher_level_annotations/se13-aw-framework-sensekey-30_mun sensekey False $mfs_fallback
$command one_experiment.job $model $vocab "$out/sensekey-se13-framework-semcor_mun" higher_level_annotations/se13-aw-framework-sensekey-30_semcor_mun sensekey False $mfs_fallback
