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

Now I'll start deduplication