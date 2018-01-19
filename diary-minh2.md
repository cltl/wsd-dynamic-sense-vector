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

I'll also need to add it to the evaluation scripts. 

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


TODO: docker image
        
5. [x] Try out scikit implementation 
4. [x] Implement custom kernel ("We found that the graph has about the right density for common senses when ... between 85 to 98.")
3. [x] measure running time
3. [x] Build graph and run Scikit label propagation
4. [x] for 25 Oct: label prop. Semcor + OMSTI
5. [x] for 25 Oct: list of all experiments for the reproduction paper
6. [x] save models of every epoch (instead of only the best one)
6. [x] Read more about label propagation (Zhou et al. 2004)
7. [x] Hyperparameter tuning of label propagation
8. [ ] Training creates a lot of models, how to reduce it?
9. [ ] Send code+data to Jacopo to run
10. [x] Polish the arxiv paper
11. [x] Use the same dev set for different sizes of the data.