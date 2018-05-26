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

## Fri 18 May 2018

I finally decided how to encode HDN datasets. I use a DataFrame to store the
location of the sentence+target word, the HDN and its competing candidates.
It's better than bare Numpy array because you can keep track of what each column
means.

    [minhle@fs0 wsd-dynamic-sense-vector]$ python3 version.py
    2018-05-18-f48a06c
    [minhle@fs0 wsd-dynamic-sense-vector]$ nohup python3 -u generate_hdn_datasets.py > output/generate_hdn_datasets.`python3 version.py`.out 2>&1 &
    [1] 19151
    [minhle@fs0 wsd-dynamic-sense-vector]$
    [minhle@fs0 wsd-dynamic-sense-vector]$
    [minhle@fs0 wsd-dynamic-sense-vector]$ tail -f output/generate_hdn_datasets.2018-05-18-f48a06c.out
    nohup: ignoring input
    Looking for HDNs:  25%|██▌       | 30000/119034 [00:07<00:22, 3923.06lemma/s]
    ...
    Extracting monosemous words from "train": 100%|██████████| 109174684/109174684 [1:25:59<00:00, 21161.01it/s]
    Extracting examples from "train": 100%|██████████| 34967048/34967048 [09:15<00:00, 62949.74it/s]
    Transformed dataset "train" written to output/gigaword-hdn-train.2018-05-18-f48a06c.pkl
    Extracting monosemous words from "dev": 100%|██████████| 12130521/12130521 [09:42<00:00, 20808.11it/s]
    Extracting examples from "dev": 100%|██████████| 3884335/3884335 [01:00<00:00, 64312.73it/s]
    Transformed dataset "dev" written to output/gigaword-hdn-dev.2018-05-18-f48a06c.pkl
            
## Mon 21 May

Managed to train one epoch but the program broke at evaluation against devset.

    Epoch #1 finished:
        Train cost: 0.162
    Traceback (most recent call last):
    ...
      File "/var/scratch/minhle/wsd-dynamic-sense-vector/model.py", line 337, in measure_dev_cost
        precomputed_batches=batches)
      File "/var/scratch/minhle/wsd-dynamic-sense-vector/model.py", line 343, in _predict
        probs = np.empty((len(data[1]), len(self.hdn2id)), dtype=np.float32)
    MemoryError

Avoid allocating one big `probs` array but accumulate batch costs and accuracies.
I made my code cleaner and more modular.

## Fri 25 May

Use this model: `output/hdn-large.2018-05-21-b1d1867-best-model` produced by
`das5/train-lstm-hdn.job`.
Evaluated my HDN model, [it didn't work](https://github.com/cltl/LSTM-WSD/commit/96c43a923ac01d88882a4b66a3d2cc5db6c571a6).
The precision is around 40% which is terrible.

Found [a terrible bug](https://github.com/cltl/wsd-dynamic-sense-vector/blob/b1d18670b0c8ca59ad93a48768dea6724439c57d/model.py#L306). I knew this problem 
before but still, why Python allows a loop variable to persist outside of 
its loop??? >"<

Started training a HDN model again:

    [minhle@fs0 wsd-dynamic-sense-vector]$ python3 version.py
    2018-05-25-e069882
    [minhle@fs0 wsd-dynamic-sense-vector]$ sbatch das5/train-lstm-hdn.job
    Submitted batch job 1822549

Also extract embeddings:

    [minhle@fs0 wsd-dynamic-sense-vector]$ python3 version.py
    2018-05-25-3f0488c
    [minhle@fs0 wsd-dynamic-sense-vector]$ sbatch das5/extract-embeddings-monosemous.job
    Submitted batch job 1822550

We know that 50% of monosemous cases in Gigaword is "percent". It wouldn't make
sense to maintain that much redundancy so I subsampled to 100 example per word.


