# todo
# official evaluation

#check if enough arguments are passed, else print usage information
if [ $# -eq 0 ];
then
    echo
    echo "Usage:                    : $0 out_dir batch_size emb_setting eval_setting model_path vocab_path sense_annotations_path output_sense_embeddings wsd_df_in wsd_df_out"
    echo
    echo "out_dir                   : out_dir"
    echo "batch_size                : batch size"
    echo "emb_setting               : granularity for embeddings"
    echo "eval_setting              : granularity for evaluation"  
    echo "setting                   : sensekey | synset | hdn"
    echo "model_path                : path to trained LSTM model (without .meta)"
    echo "vocab_path                : endswith .index.pkl"
    echo "sense_annotations_path    : path to file with sense annotations"
    echo "wsd_df_in                 : path to dataframe with wsd competition info"
    exit -1;
fi

#assign command line arguments to variables and obtain basename
cwd=/${PWD#*/}
out_folder=$1
batch_size=$2
emb_setting=$3
eval_setting=$4
model_path=$5
vocab_path=$6
sense_annotations_path=$7
wsd_df_in=$8
mfs_fallback=$9
path_case_freq="${10}"
use_case_strategy="${11}"
path_plural_freq="${12}"
use_number_strategy="${13}"
path_lp="${14}"
use_lp="${15}"

echo $path_case_freq

#create output folder
log_file=$out_folder/log.txt
rm -rf $out_folder && mkdir $out_folder

#set output paths
embeddings_path=$out_folder/meaning_embeddings.bin
wsd_df_out=$out_folder/wsd_output.bin
results=$out_folder/results.txt
settings_path=$out_folder/settings.json


# log experiments settings
echo -n 'Started: ' && date >> $log_file

echo "storing results in $out_folder"
echo "storing results in $out_folder" >> $log_file

echo "name experiment: $name_experiment" >> $log_file
echo "batch_size" $batch_size >> $log_file
echo "emb_setting" $emb_setting >> $log_file
echo "mfs_fallback" $mfs_fallback >> $log_file
echo "eval_setting" $eval_setting >> $log_file
echo "model path: $model_path" >> $log_file
echo "vocab path: $vocab_path" >> $log_file
echo "sense_annotations_path: $sense_annotations_path" >> $log_file
echo "embeddings_path: $embeddings_path" >> $log_file
echo "wsd_df_in: $wsd_df_in" >> $log_file
echo "wsd_df_out: $wsd_df_out" >> $log_file
echo "results: $results" >> $log_file
echo "path_case_freq:" $path_case_freq >> $log_file
echo "use_case_strategy:" $use_case_strategy >> $log_file
echo "path_plural_freq:" $path_plural_freq >> $log_file
echo "use_number_strategy:" $use_number_strategy >> $log_file
echo "path_lp" $path_lp >> $log_file
echo "use_lp" $use_lp >> $log_file

# create embeddings
echo 'started training meaning embeddings at:' >> $log_file
date >> $log_file
echo "python3 test-lstm_v2.py -m $model_path -v $vocab_path -i $sense_annotations_path -o $embeddings_path -b $batch_size -t 1000000 -s $emb_setting"
python3 test-lstm_v2.py -m $model_path -v $vocab_path -i $sense_annotations_path -o $embeddings_path -b $batch_size -t 1000000 -s $emb_setting
echo 'finished training meaning embeddings at:' >> $log_file
date >> $log_file

# evaluate embeddings
echo 'started evaluation' >> $log_file
date >> $log_file
echo "python3 perform_wsd.py -m $model_path -v $vocab_path -c $wsd_df_in -s $embeddings_path -o $wsd_df_out -r $results -g $eval_setting " >> $log_file
python3 perform_wsd.py -m $model_path -v $vocab_path -c $wsd_df_in -l $settings_path -s $embeddings_path -o $wsd_df_out -r $results -g $eval_setting -f $mfs_fallback -t $path_case_freq -a $use_case_strategy -p $path_plural_freq -b $use_number_strategy -y $path_lp -z $use_lp
echo 'ended evaluation' >> $log_file
date >> $log_file

echo -n 'End: ' && date >> $log_file
