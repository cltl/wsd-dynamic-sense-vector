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

                          
5. [x] Try out scikit implementation 
4. [x] Implement custom kernel ("We found that the graph has about the right density for common senses when ... between 85 to 98.")
3. [x] measure running time
2. [ ] Query UkWaC??? for sentences that contain a certain lemma
3. [x] Build graph and run Scikit label propagation
4. [x] for 25 Oct: label prop. Semcor + OMSTI
5. [x] for 25 Oct: list of all experiments for the reproduction paper
6. [x] save models of every epoch (instead of only the best one)
6. [x] Read more about label propagation (Zhou et al. 2004)
7. [ ] Hyperparameter tuning of label propagation
8. [ ] Training creates a lot of models, how to reduce it?
9. [ ] Send code+data to Jacopo to run
10. [ ] Polish the paper
11. [x] Use the same dev set for different sizes of the data.