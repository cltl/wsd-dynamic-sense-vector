## Fri 4 May 2018

I got fed up with the ugliness of the code so I decided to tidy it up. It's
apparently not the right time as I now have a job and a PhD to finish but I
can't help it.

First thing is to standardize the file naming convention. Everything is now
written to `output/` with a version number.

    [minhle@fs0 wsd-dynamic-sense-vector]$ sbatch das5/preprocess.job
    Submitted batch job 1802408
    [minhle@fs0 wsd-dynamic-sense-vector]$ tail -f slurm-1802408.out
    Writing to output/gigaword.2018-05-04-149cde7.txt.gz.tmp
    Preprocessing GigaWord:   0%|          | 0/737 [00:00<?, ?file/s]

Before introducing the versioning system, I favored writing big scripts to 
avoid writing intermediate files. This habit is carried over to much later
version of the code. Now, I think I should write smaller scripts that write
to versioned intermediate files. Besides, I will convert GigaWord to numerical 
format sooner to speed up preprocessing.

## Thu 10 May 

(Linguistic) preprocessing of Gigaword finished:

    [minhle@fs0 wsd-dynamic-sense-vector]$ cat slurm-1802408.out
    Writing to output/gigaword.2018-05-04-149cde7.txt.gz.tmp
    Preprocessing GigaWord: 100%|██████████| 737/737 [43:22:42<00:00, 211.89s/file]
    Successful! Result saved to output/gigaword.2018-05-04-149cde7.txt.gz

Now I'll start deduplication...

Deduplication is done and vectorization too:

    [minhle@fs0 wsd-dynamic-sense-vector]$ python3 -u vectorize_gigaword.py
    Building vocabulary: 100%|█████████████████████████████████████████████████████████████████| 109174684/109174684 [17:57<00:00, 101291.82line/s]
    Total unique words: 3819567
    Retained 936831 words
    Vocabulary written to output/vocab.2018-05-10-7d764e7.pkl
    Vectorizing "train": 100%|██████████████████████████████████████████████████████████████████| 109174684/109174684 [21:41<00:00, 83876.44line/s]
    Result saved to output/gigaword-train.2018-05-10-7d764e7.npz
    Vectorizing "dev": 100%|██████████████████████████████████████████████████████████████████████| 12130521/12130521 [02:06<00:00, 96134.14line/s]
    Result saved to output/gigaword-dev.2018-05-10-7d764e7.npz

## Tue 15 May

Solidify the conversion of monosemous words to HDNs. It is very satisfying to 
look at a neat source file. It's running now on DAS-5:

    [minhle@fs0 wsd-dynamic-sense-vector]$ python3 version.py
    2018-05-15-e200820
    [minhle@fs0 wsd-dynamic-sense-vector]$ nohup python3 -u convert_mono2hdn.py > output/convert_mono2hdn.2018-05-15-e200820.out 2>&1 &
    [1] 20356
    [minhle@fs0 wsd-dynamic-sense-vector]$ tail -f output/convert_mono2hdn.2018-05-15-e200820.out
    nohup: ignoring input
    Looking for HDNs: 100%|██████████| 119034/119034 [00:29<00:00, 4100.90lemma/s]
    Looking for monosemous nouns... Done.
    All noun lemmas: 119034
    Monosemous noun lemmas: 101168 (85.0%)
    Monosemous multi-word-expressions: 58344 (57.7% of monosemous)
        Samples: carissa_plum, Second_Earl_of_Guilford, rotary_converter, Christmas_holly, level_best
    Lemmas that are not associated with any HDN: entity, physical_entity, abstract_entity, freshener, otherworld, whacker
    Number of HDNs per lemma: mean=5.3, median=5.0, std=2.0)
    Counting lines in output/gigaword-train.2018-05-10-9fd479f.txt.gz... Done.
    Transforming "train":   0%|          | 110000/109174684 [00:13<3:43:07, 8146.95sentence/s]



