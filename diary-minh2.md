The first file gets way too long so I moved to this new file.
           
## Fri 24 Nov

One experiment failed, probably because of memory overflow:

    /var/log/slurm/spool_slurmd//job3779738/slurm_script: line 12: 50729 Killed                  python3 -u measure-speedups.py --config sampled-softmax

Apparently, unoptimized batches and sampled softmax don't go well together.
To do sampled softmax, we need to create a copy of sentences with subvocab indices,
i.e. we'll need double the amount of RAM. Optimizing batches before applying 
sampled softmax doesn't work either because
(big #sentence) * (big vocabulary) = OOM. So I will only report the 
effect of applying both of them together.

Same-length experiment:

    squeue | grep minh
               3779739       gpu exp-opti   minhle  R 1-00:49:00      1 gcn15
    
    [minhle@int1 wsd-dynamic-sense-vector]$ tail -n 1 slurm-3779739.out
        finished 88000 of 157344 batches, sample batch cost: 0.0000002

So it would take: 10/(88000/157344) = 17.88 days to finish 10 epochs. I can
already report this number. I stopped the job so it won't consume too much of
my budget.

Cancelled job 3779919 (exp-h256p64.job) because it the dev set wasn't fixed. It
would cause bigger random variation.

Fix 1001 problems. Started 3 types of experiments on Cartesius. Data preparation
is still running (it's working on the 75% training set which isn't used by
any running experiments).

    [minhle@int1 wsd-dynamic-sense-vector]$ !sque
    squeue | grep minh
               3784737    normal prepare-   minhle  R    7:31:20      1 tcn1111
               3785722       gpu exp-opti   minhle  R       1:17      1 gcn9
               3785690       gpu exp-h256   minhle  R      35:56      1 gcn36
               3785691       gpu exp-h256   minhle  R      35:56      1 gcn58
               3785692       gpu exp-h256   minhle  R      35:56      1 gcn59
               3785688       gpu exp-h256   minhle  R      35:59      1 gcn30
               3785689       gpu exp-h256   minhle  R      35:59      1 gcn31
               3785679       gpu exp-data   minhle  R      38:40      1 gcn40
               3785673       gpu exp-data   minhle  R      44:24      1 gcn7

## Mon 27 Nov

    [minhle@int2 wsd-dynamic-sense-vector]$ grep Dev slurm-3785690.out | tail
    ...
        Dev cost: 4.041, hit@100: 0.8%
    [minhle@int2 wsd-dynamic-sense-vector]$ grep Dev slurm-3785691.out | tail
    ...
        Dev cost: 3.996, hit@100: 0.8%
    [minhle@int2 wsd-dynamic-sense-vector]$ grep Dev slurm-3785692.out | tail
    ...
        Dev cost: 4.006, hit@100: 0.8%
    [minhle@int2 wsd-dynamic-sense-vector]$ grep Dev slurm-3785688.out | tail
    ...
        Dev cost: 4.002, hit@100: 0.8%
    [minhle@int2 wsd-dynamic-sense-vector]$ grep Dev slurm-3785689.out | tail
    ...
        Dev cost: 3.999, hit@100: 0.8%

Two data-size experiments have finished:

    [minhle@int2 wsd-dynamic-sense-vector]$ cat output/2017-11-24-4e4a04a/lstm-wsd-gigaword_01-pc_large.results/mun/results.txt
    1009
    [minhle@int2 wsd-dynamic-sense-vector]$ cat output/2017-11-24-4e4a04a/lstm-wsd-gigaword_10-pc_large.results/mun/results.txt
    1034
    
    [minhle@int2 wsd-dynamic-sense-vector]$ cat output/2017-11-24-4e4a04a/lstm-wsd-gigaword_01-pc_large.results/semcor/results.txt
    971
    [minhle@int2 wsd-dynamic-sense-vector]$ cat output/2017-11-24-4e4a04a/lstm-wsd-gigaword_10-pc_large.results/semcor/results.txt
    1031

Started new ones:

    [minhle@int2 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-data-size.job 25
    Submitted batch job 3788824
    [minhle@int2 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-data-size.job 50
    Submitted batch job 3792013
    [minhle@int2 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-data-size.job 75
    Submitted batch job 3792014

Optimization experiment: finished in 10 hours (compared to 17 or 19 days)

    [minhle@int2 wsd-dynamic-sense-vector]$ head slurm-3785722.out
    Started: Fri Nov 24 23:33:32 CET 2017
    ...
    [minhle@int2 wsd-dynamic-sense-vector]$ tail slurm-3785722.out
    ...
    Finished: Sat Nov 25 09:59:55 CET 2017

Made a backup of all job outputs in `~/wsd-dynamic-sense-vector.bak`.

## Wed 29 Nov

3 of 5 h256p64 experiments finished.

## Thu 30 Nov

4 of 5 h256p64 experiments finished successfully, the other was terminated due
to time limit. Resumed it because it's very close to finish.

    [minhle@int2 wsd-dynamic-sense-vector]$ git checkout e93fdb2
    ...
    [minhle@int2 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-h256p64.job 71
    Submitted batch job 3802297

The results are quite stable, I'm happy :D

    [minhle@int2 wsd-dynamic-sense-vector]$ tail -vn +1 output/2017-11-24-e93fdb2/lstm-wsd-gigaword-h256p64-seed_*.results/mun/results.txt
    ==> output/2017-11-24-e93fdb2/lstm-wsd-gigaword-h256p64-seed_12.results/mun/results.txt <==
    1044
    ==> output/2017-11-24-e93fdb2/lstm-wsd-gigaword-h256p64-seed_398.results/mun/results.txt <==
    1055
    ==> output/2017-11-24-e93fdb2/lstm-wsd-gigaword-h256p64-seed_493.results/mun/results.txt <==
    1054
    ==> output/2017-11-24-e93fdb2/lstm-wsd-gigaword-h256p64-seed_814.results/mun/results.txt <==
    1054
    
    [minhle@int2 wsd-dynamic-sense-vector]$ tail -vn +1 output/2017-11-24-e93fdb2/lstm-wsd-gigaword-h256p64-seed_*.results/semcor/results.txt
    ==> output/2017-11-24-e93fdb2/lstm-wsd-gigaword-h256p64-seed_12.results/semcor/results.txt <==
    1039
    ==> output/2017-11-24-e93fdb2/lstm-wsd-gigaword-h256p64-seed_398.results/semcor/results.txt <==
    1052
    ==> output/2017-11-24-e93fdb2/lstm-wsd-gigaword-h256p64-seed_493.results/semcor/results.txt <==
    1053
    ==> output/2017-11-24-e93fdb2/lstm-wsd-gigaword-h256p64-seed_814.results/semcor/results.txt <==
    1038


## Fri 1 Dec 2017

Fixed bug: mistake distance for affinity (similarity) in label propagation.

Started some hyperparameter testing on DAS-5:

    [minhle@fs0 wsd-dynamic-sense-vector]$ das5/exp-hyperp-label-propagation.sh
    Submitted batch job 1694124
    Submitted batch job 1694125
    Submitted batch job 1694126
    Submitted batch job 1694127
    Submitted batch job 1694128

## Tue 5 Dec 2017

Fixed 101 bugs in label propagation experiments. Got some results: label 
propagation (with current implementation) is worse than simple nearest neighbor
but label spreading is the best:

    [minhle@fs0 wsd-dynamic-sense-vector]$ head -n 3 slurm-1697466.out && tail -n 2 slurm-1697466.out
    Started: Mon Dec  4 15:36:59 CET 2017
    Log is written to output/2017-12-04-working/exp-hyperp-label-propagation_algo-propagate_sim-rbf_gamma-0.01.log
    total 0.35854777446258185
    Finished: Mon Dec  4 16:15:32 CET 2017
    [minhle@fs0 wsd-dynamic-sense-vector]$ head -n 3 slurm-1697467.out && tail -n 2 slurm-1697467.out
    Started: Mon Dec  4 15:36:59 CET 2017
    Log is written to output/2017-12-04-working/exp-hyperp-label-propagation_algo-spread_sim-expander_gamma-.log
    total 0.4909980550096231
    Finished: Mon Dec  4 16:09:39 CET 2017
    [minhle@fs0 wsd-dynamic-sense-vector]$ head -n 3 slurm-1697468.out && tail -n 2 slurm-1697468.out
    Started: Mon Dec  4 15:36:59 CET 2017
    Log is written to output/2017-12-04-working/exp-hyperp-label-propagation_algo-nearest_sim-expander_gamma-.log
    total 0.4798983717070091
    Finished: Mon Dec  4 15:53:56 CET 2017

The similarity function is also very important because I tried label spreading
before with Euclidean distance and RBF but got terrible results.

I'm curious about nearest neighbor of the averaged context vector.

Ran a big batch of experiments to test everything. I hope I could sort it out
today. Otherwise it'll sit there forever because I need to finish the paper and
from January I'll have very limited time.
    
    [minhle@fs0 wsd-dynamic-sense-vector]$ tail -n 1 output/2017-12-05-dcd006e/*.log
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-average_sim-expander_gamma-.log <==
    total 0.22927465097096772
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-average_sim-rbf_gamma-1.log <==
    total 0.2948646144132952
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-nearest_sim-expander_gamma-.log <==
    total 0.4798983717070091
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-nearest_sim-rbf_gamma-1.log <==
    total 0.35843575930998667
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-propagate_sim-expander_gamma-.log <==
    total 0.35850958747874256
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-propagate_sim-rbf_gamma-0.01.log <==
    total 0.35854777446258185
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-propagate_sim-rbf_gamma-0.1.log <==
    total 0.35855032026150446
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-propagate_sim-rbf_gamma-0.5.log <==
    total 0.35841030132076046
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-propagate_sim-rbf_gamma-1.log <==
    total 0.35841030132076046
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-spread_sim-expander_gamma-.log <==
    total 0.4909980550096231
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-spread_sim-rbf_gamma-0.01.log <==
    total 0.47848036170711095
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-spread_sim-rbf_gamma-0.1.log <==
    total 0.46497744422154563
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-spread_sim-rbf_gamma-0.5.log <==
    total 0.3595915520208552
    
    ==> output/2017-12-05-dcd006e/exp-hyperp-label-propagation_algo-spread_sim-rbf_gamma-1.log <==
    total 0.35843321351106405

Cartesius was super busy for a while, I only got access to GPUs today. Seed 71 
experiment was ruined because when it started yesterday, the source code has
changed to a different version. So I checkout the right version again and resumed it:

    [minhle@int2 wsd-dynamic-sense-vector]$ git checkout e93fdb2
    ...
    [minhle@int2 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-h256p64.job 71
    Submitted batch job 3814222

I'm using 6.3 tetrabytes! :-O :-O :-O

    [minhle@int2 wsd-dynamic-sense-vector]$ du -sh .
    6.3T    .

Resumed data-size 25% as well because it was terminated due to some mysterious error
(`ResourceExhaustedError (see above for traceback): output/2017-11-24-fc78409/lstm-wsd-gigaword_25-pc_large/model.ckpt-1090253.data-00000-of-00001.tempstate5936053556082246253`). I guess
there's a hidden quota on the scratch folder of Cartesius.

    [minhle@int2 wsd-dynamic-sense-vector]$ python3 version.py
    2017-11-24-4e4a04a
    [minhle@int2 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-data-size.job 25
    Submitted batch job 3814289

Data-size 50% is gone because it happens to be run on a working version and I
have just purged all working output to avoid weird errors.

Adapted evaluation scripts to report more numbers.

## Wed 6 Dec 2017

For whatever reason, my evaluation script can't be submitted as a slurm job --
it always get stuck at CG (completing) status. I'm running it using interactive
job:

    [minhle@int2 wsd-dynamic-sense-vector]$ srun -p gpu -t 5:00:00 --pty bash
    [minhle@gcn40 wsd-dynamic-sense-vector]$ python3 version.py
    2017-12-06-42bc700
    [minhle@gcn40 wsd-dynamic-sense-vector]$ nohup cartesius/exp-variation-score.job > output/`python3 version.py`/exp-variation-score.job.out 2>&1 &
    [1] 40999
    [minhle@gcn40 wsd-dynamic-sense-vector]$ tail -f output/`python3 version.py`/exp-variation-score.job.out
    ...

## Thu 7 Dec

Worked on the paper. Data size 25% experiment has finished. Tried to run the
newest evaluation script on it but no GPU machine is available yet.

       42bc700..0a0d02b  master     -> origin/master
    First, rewinding head to replay your work on top of it...
    Fast-forwarded master to 0a0d02b4538dcf7322742e32e367a90ec1055899.
    [minhle@int2 wsd-dynamic-sense-vector]$ sbatch cartesius/eval-data-size.job
    Submitted batch job 3820439

## Fri 19 Dec

Meeting with Jacopo+Marten. Jacopo would like to retrain everything with <eos>
token. Checked everything again. There doesn't seem to be big difference (that
I don't know of) between the version that produced current reported results
and a more recent version. Let's try.

Added `<eos>` to the preparation script.

    >>> from collections import Counter
    >>> c = Counter()
    >>> with open('preprocessed-data/694cb4d/gigaword.txt') as f:
          for sent in f:
            c[sent.strip().split()[-1]] += 1
    >>> c.most_common(10)
    [('.', 141537114), ("''", 7066432), ('"', 7015844), (')', 2214057), ('_', 1964897), (':', 1605763), ('?', 1486728), ('--', 774285), ("'", 648803), ('...', 434971)]
    >>> total = sum(c.values())
    >>> [(tok, cnt/total) for tok, cnt in c.most_common(10)]
    [('.', 0.8052320716307731), ("''", 0.04020230113211145), ('"', 0.039914496196088396), (')', 0.012596199360251295), ('_', 0.01117867983270516), (':', 0.00913549690604858), ('?', 0.008458283721904037), ('--', 0.004405057422483782), ("'", 0.00369116600590189), ('...', 0.002474634316970099)]

Started a job to prepare datasets with `<eos>`:

    [minhle@fs0 wsd-dynamic-sense-vector]$ sbatch das5/prepare-lstm-wsd.job
    Submitted batch job 1713648

I'll also need to add it to the evaluation scripts.

... OK, did it. It's 12:30 am now... 

## Sat 20 Jan

Fixed some problems with training and evaluation scripts and tried to train
a large model on 1% data. We need to limit the target index to the length
of the sentence (excluding <eos>) otherwise the task wouldn't make sense
and the performance on dev set is hurt.

When I look up my data on Cartesius, I found out that everything was wiped out.
They must have cleaned up their scratch folder at some point. Luckily, I 
backed up the output of jobs.

When comparing the current version with `2017-11-24-4e4a04a`, the new version
works better (i.e. the NLL on dev set decreases faster). But that's only for 
the first 20 epochs. It's 0:24 Sunday now, let's see how it goes tomorrow morning. 

## Sun 21 Jan 

Until epoch 39, it's still running good but then it crashes because there's no
more disk space. I should stop saving everything :-(

It took me about 15 minutes to clean up my scratch folder. Now I'm using 
a healthy amount of 293G over the 400G quota.

## Fri 16 Feb

Meet Jacopo and Marten again. Need to retrain models to rerun the analysis with Marten's 
debugged code. My models on Cartesius were wiped out long ago so I need to retrain
some of them.

## Sun 18 Feb

Started preprocessing again. The README wasn't entirely accurate so it took me
a while to figure it out again.

## Mon 19 Feb

Preprocessing finished.

Downloaded the models I put on Kyoto:

    model-h100p10
    model-h2048p512
    model-h256p64
    model-h512p128

Now I need to re-run the evaluation script...

## Wed 28 Feb

Sent a tar file to Jacopo for model training. 

I need to run this command:

    ./evaluate.job ../output/model-h512p128/lstm-wsd-gigaword-large ../preprocessed-data/2018-01-19-36b6246/gigaword-for-lstm-wsd.index.pkl ../output/model-h512p128.results

But before doing so I need to install Marten's evaluation framework first. One
script stubbornly refuses to finish. Nohup'd it so I can go to bed >"<  

    nohup ./convert_sense_annotations_171.job mun synset &

## Thu 1 Mar

For some reasons, the script still stopped in the middle. I rerun Marten's 
script again

    [minhle@fs0 evaluate]$ bash convert_all_sense_annotations.sh
    Submitted batch job 1784349
    Submitted batch job 1784350
    Submitted batch job 1784351
    Submitted batch job 1784352
    Submitted batch job 1784353
    Submitted batch job 1784354
    [minhle@fs0 evaluate]$ squeue
    ...
           1784349      defq convert_   minhle  R       0:05      1 node062
           1784350      defq convert_   minhle  R       0:05      1 node065
           1784351      defq convert_   minhle  R       0:05      1 node066
           1784352      defq convert_   minhle  R       0:05      1 node067
           1784353      defq convert_   minhle  R       0:05      1 node068
           1784354      defq convert_   minhle  R       0:05      1 node030

## Fri 2 Mar

Looks like the scripts didn't fail in Wednesday night. It just didn't say that
it finished so I thought it was terminated. 

`evaluate.job` doesn't work. Dug up Marten's instructions (it was 11 days ago!
LREC and work made it impossible for me to work on this).

    [minhle@fs0 evaluate]$ ./evaluate_in_parallel.sh ../output/model-h512p128/lstm-wsd-gigaword-large ../preprocessed-data/2018-01-19-36b6246/gigaword-for-lstm-wsd.index.pkl ../output/model-h512p128.results
    Submitted batch job 1784361
    Submitted batch job 1784362
    Submitted batch job 1784363
    Submitted batch job 1784364
    Submitted batch job 1784365
    Submitted batch job 1784366

Looks like Marten's script is working.

    ./evaluate_in_parallel.sh ../output/model-h2048p512/lstm-wsd-gigaword-google ../output/model-h2048p512/gigaword-lstm-wsd.index.pkl ../output/model-h2048p512-mfs-false.results False &
    ./evaluate_in_parallel.sh ../output/model-h2048p512/lstm-wsd-gigaword-google ../output/model-h2048p512/gigaword-lstm-wsd.index.pkl ../output/model-h2048p512-mfs-true.results True &

This is the results:

    [minhle@fs0 wsd-dynamic-sense-vector]$ python3 compile_results.py
    *** Senseval2 ***
                             model +MFS      P      R     F1
    4          Our LSTM (T: OMSTI)   No  0.682  0.447  0.540
    9          Our LSTM (T: OMSTI)  Yes  0.665  0.665  0.665
    2         Our LSTM (T: SemCor)   No  0.706  0.656  0.680
    11        Our LSTM (T: SemCor)  Yes  0.694  0.694  0.694
    5   Our LSTM (T: SemCor+OMSTI)   No  0.683  0.635  0.658
    10  Our LSTM (T: SemCor+OMSTI)  Yes  0.673  0.673  0.673
    *** SemEval13 ***
                            model +MFS      P      R     F1
    3         Our LSTM (T: OMSTI)   No  0.676  0.501  0.575
    6         Our LSTM (T: OMSTI)  Yes  0.651  0.651  0.651
    0        Our LSTM (T: SemCor)   No  0.654  0.638  0.646
    7        Our LSTM (T: SemCor)  Yes  0.648  0.648  0.648
    1  Our LSTM (T: SemCor+OMSTI)   No  0.660  0.644  0.651
    8  Our LSTM (T: SemCor+OMSTI)  Yes  0.653  0.653  0.653

Everything is lower than before... Emailed Marten. I have spent the whole morning
on this project. I should be working on the coreference paper now.


## Fri 9 Mar

Get more numbers...

    ./evaluate_in_parallel.sh ../output/model-h100p10/lstm-wsd-gigaword-google ../output/model-h100p10/gigaword-lstm-wsd.index.pkl ../output/model-h100p10-mfs-true.results True &
    ./evaluate_in_parallel.sh ../output/model-h256p64/lstm-wsd-gigaword-google ../output/model-h256p64/gigaword-lstm-wsd.index.pkl ../output/model-h256p64-mfs-true.results True &
    ./evaluate_in_parallel.sh ../output/model-h512p128/lstm-wsd-gigaword-google ../output/model-h512p128/gigaword-lstm-wsd.index.pkl ../output/model-h512p128-mfs-true.results True &

Marten changed the evaluation script. Everything needs to be redone.

## Sun 11 Mar

Marten's script was running quite slowly... I will check the results tomorrow.

    [minhle@node029 wsd-dynamic-sense-vector]$ cat evaluate-coling.sh
    cd evaluate
    ./evaluate_in_parallel.sh ../output/model-h2048p512/lstm-wsd-gigaword-google ../output/model-h2048p512/gigaword-lstm-wsd.index.pkl ../output/model-h2048p512-mfs-false.results False
    ./evaluate_in_parallel.sh ../output/model-h2048p512/lstm-wsd-gigaword-google ../output/model-h2048p512/gigaword-lstm-wsd.index.pkl ../output/model-h2048p512-mfs-true.results True
    ./evaluate_in_parallel.sh ../output/model-h100p10/lstm-wsd-gigaword-small_seed-124-best-model ../output/model-h100p10/gigaword-for-lstm-wsd.index.pkl ../output/model-h100p10-mfs-true.results True
    ./evaluate_in_parallel.sh ../output/model-h256p64/lstm-wsd-gigaword-h256p64-seed_12-best-model ../output/model-h256p64/gigaword-for-lstm-wsd.index.pkl ../output/model-h256p64-mfs-true.results True
    ./evaluate_in_parallel.sh ../output/model-h512p128/lstm-wsd-gigaword-large ../output/model-h512p128/gigaword-lstm-wsd.index.pkl ../output/model-h512p128-mfs-true.results True
    
## Mon 12 Mar

Woke up to a bug in the previous commands. Google models were evaluate
successfully but others have failed. Actually I saw some hints last night but
didn't understand.

Finished the evaluation before the working hours so Piek and Jacopo could have
a look at the paper.

## Tue 13 Mar

Evaluate data-size-experiment models:

    cd evaluate
    ./evaluate_in_parallel.sh ../output/data-sizes/1/lstm-wsd-gigaword_01-pc_large-best-model ../preprocessed-data/2017-11-24-a74bda6/gigaword-for-lstm-wsd.index.pkl ../output/data-sizes/1.results True
    ./evaluate_in_parallel.sh ../output/data-sizes/10/lstm-wsd-gigaword_10-pc_large-best-model ../preprocessed-data/2017-11-24-a74bda6/gigaword-for-lstm-wsd.index.pkl ../output/data-sizes/10.results True
    ./evaluate_in_parallel.sh ../output/data-sizes/25/lstm-wsd-gigaword_25-pc_large-best-model ../preprocessed-data/2017-11-24-a74bda6/gigaword-for-lstm-wsd.index.pkl ../output/data-sizes/25.results True

Added to paper

TODO: get real num. of params. Hopefully I calculated them correctly. 
see https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model

## Fri 27 Apr 2018

Attempt to build a model to predict HDNs (highest disambiguating nodes).

Plan: go through Gigaword and replace monosemous lemma by an HDN (of other 
lemmas) that dominates it in WordNet hierarchy. Train an LSTM to predict this
HDN (against competing HDNs).

I built a simple script to generate a dataset. The output is not versioned yet,
if the script finishes successfully, I'll rerun the script to give it a proper
version.

    [minhle@fs0 wsd-dynamic-sense-vector]$ sbatch das5/preprocess-hdn.job
    Submitted batch job 1798960
    [minhle@fs0 wsd-dynamic-sense-vector]$ tail -f slurm-1798960.out
    Monosemous multi-word-expressions: 42820
    All monosemous words: 101162
    All lemmas: 119034
    Proportion monosemous/all: 0.8498580237579179
     61%|██████    | 106537103/175771829 [08:40<05:38, 204585.90it/s]

## Sat 29 Apr

Tried to make batches out of the transformed dataset. Failed.

    /cm/local/apps/slurm/var/spool/job1799463/slurm_script: line 4: 30937 Killed                  python3 -u preprocess_hdn.py

I'll try to limit the size of data tomorrow.

## Fri 4 May

Limited training set size to 1.5e8 sentences.

    [minhle@fs0 wsd-dynamic-sense-vector]$ !sbatch
    sbatch das5/preprocess-hdn.job
    Submitted batch job 1802365
    [minhle@fs0 wsd-dynamic-sense-vector]$ tail -f slurm-1802365.out
    

