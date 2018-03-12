#!/usr/bin/env bash

out=submission

cd ..

#create submission folder
rm -rf $out
mkdir $out

#install.sh file
cp evaluate/install.sh $out
cp evaluate/README.md $out

#python scripts to run conversion to lstm input format
cp evaluate/sense_annotations2lstm_format.py $out
cp evaluate/mapping_utils.py $out
cp evaluate/wn_utils.py $out
cp evaluate/wsd_datasets_classes.py $out

#bash scripts to run conversion
cp evaluate/convert_all_sense_annotations.sh $out
cp evaluate/convert_sense_annotations_se2_framework.job $out
cp evaluate/convert_sense_annotations_se13_framework.job $out

#python scripts to run train sense embeddings and perform wsd
cp evaluate/test-lstm_v2.py $out
cp evaluate/tensor_utils.py $out
cp evaluate/perform_wsd.py $out

cp evaluate/morpho_utils.py $out
cp evaluate/score_utils.py $out
cp evaluate/tsne_utils.py $out
cp evaluate/official_scorer.py $out

#bash scripts to run the experiments
cp evaluate/evaluate_in_parallel.sh $out
cp evaluate/one_experiment.job $out
cp evaluate/one_full_experiment_v2.sh $out

#result tables
cp evaluate/official_results.py $out
cp evaluate/result_tables.py $out

# label propagation
cp evaluate/run_lp.job $out
cp evaluate/all_lp_jobs.sh $out
cp evaluate/debug_lp.py $out
cp evaluate/label_propagation.py $out
