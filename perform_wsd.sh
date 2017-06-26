
#!/bin/bash

#module load cuda80/toolkit
#module load cuda80/blas
#module load cuda80
#module load cuDNN

# add dataframes to folder
# add wsd_datasets_classes to folder

model_path=/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/lstm-wsd-small
vocab_path=/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/gigaword.1m-sents-lstm-wsd.index.pkl
wsd_df_path=/Users/marten/GoogleDrive/Spinoza/Research/SharedTask/Repositories/WSD-Gold_standard-Analyst/dataframes/sem2013-aw.p
sense_embeddings_path=/var/scratch/mcpostma/wsd-dynamic-sense-vector/output/sense_embeddings.bin
wsd_df_output_path=/Users/marten/GoogleDrive/Spinoza/Research/SharedTask/Repositories/WSD-Gold_standard-Analyst/dataframes/sem2013-aw.p.wsd

#python3 /var/scratch/mcpostma/wsd-dynamic-sense-vector/perform_wsd.py -m $model_path -v $vocab_path -c $wsd_df_path -s $sense_embeddings_path -o $wsd_df_output_path
python3 perform_wsd.py -m $model_path -v $vocab_path -c $wsd_df_path -s $sense_embeddings_path -o $wsd_df_output_path



