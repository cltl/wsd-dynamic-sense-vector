# todo
# official evaluation

#check if enough arguments are passed, else print usage information
if [ $# -eq 0 ];
then
    echo
    echo "Usage:                    : $0 name_experiment model_path vocab_path sense_annotations_path output_sense_embeddings wsd_df_in wsd_df_out"
    echo
    echo "NOTE: this script assumes you have a folder called experiments in the same dir as this script \
    in which one folder will be created for each experiment"
    echo
    echo "name_experiment           : name of experiment"
    echo "model_path                : path to trained LSTM model (without .meta)"
    echo "vocab_path                : endswith .index.pkl"
    echo "sense_annotations_path    : path to file with sense annotations"
    echo "wsd_df_in                 : path to dataframe with wsd competition info"
    exit -1;
fi

#assign command line arguments to variables and obtain basename
cwd=/${PWD#*/}
name_experiment=$1
model_path=$2
vocab_path=$3
sense_annotations_path=$4
wsd_df_in=$5

#create output folder
out_folder=$cwd/experiments/$name_experiment
log_file=$out_folder/log.txt
rm -rf $out_folder && mkdir $out_folder

#set output paths
embeddings_path=$out_folder/meaning_embeddings.bin
wsd_df_out=$out_folder/wsd_output.bin
results=$out_folder/results.txt


# log experiments settings
echo -n 'Started: ' && date >> $log_file

echo "storing results in $out_folder"
echo "storing results in $out_folder" >> $log_file

echo "name experiment: $name_experiment" >> $log_file
echo "model path: $model_path" >> $log_file
echo "vocab path: $vocab_path" >> $log_file
echo "sense_annotations_path: $sense_annotations_path" >> $log_file
echo "embeddings_path: $embeddings_path" >> $log_file
echo "wsd_df_in: $wsd_df_in" >> $log_file
echo "wsd_df_out: $wsd_df_out" >> $log_file
echo "results: $results" >> $log_file

# create embeddings
echo 'started training meaning embeddings at:' >> $log_file
date >> $log_file
echo "python3 /var/scratch/mcpostma/wsd-dynamic-sense-vector/test-lstm.py -m $model_path -v $vocab_path -i $sense_annotations_path -o $embeddings_path -t 2000"
python3 /var/scratch/mcpostma/wsd-dynamic-sense-vector/test-lstm.py -m $model_path -v $vocab_path -i $sense_annotations_path -o $embeddings_path -t 2000
echo 'finished training meaning embeddings at:' >> $log_file
date >> $log_file

# evaluate embeddings
echo 'started evaluation' >> $log_file
date >> $log_file
echo "python3 /var/scratch/mcpostma/wsd-dynamic-sense-vector/perform_wsd.py -m $model_path -v $vocab_path -c $wsd_df_in -s $embeddings_path -o $wsd_df_out -r $results" >> $log_file
python3 /var/scratch/mcpostma/wsd-dynamic-sense-vector/perform_wsd.py -m $model_path -v $vocab_path -c $wsd_df_in -s $embeddings_path -o $wsd_df_out -r $results
echo 'ended evaluation' >> $log_file
date >> $log_file

echo -n 'End: ' && date >> $log_file
