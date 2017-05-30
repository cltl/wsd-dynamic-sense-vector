## Tue 9 May

Wrote a script to convert Gigaword to bare sentences and tokenize them.

Wrote another script to train Word2vec on the resulting file.

I think copying the file to a SSD and run the training script from there would improve speed.
It did. Running on regular disk, the program can only use 200-300% CPU (i.e. the equivalent of
2-3 CPUs running 100%) but with this trick, it can reach 3000%.

```
  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
 1925 minhle    20   0 4061164 464940  11880 R  3051  0.7  43:50.84 python3
```

Started my script for running for 12 hours. If that's not enough, we need to ask for a dedicated server.

```
[minhle@fs0 wsd-with-marten]$ sbatch word2vec_gigaword.job
Submitted batch job 1394958
```

## Tue 10 May

The job failed. Error: `OSError: [Errno 122] Disk quota exceeded`.
I moved everything to `scratch` folder. Restarted.

```
[minhle@fs0 wsd-with-marten]$ sbatch word2vec_gigaword.job
Submitted batch job 1395268
```

## Mon 15 May

The Gigaword word2vec training finished somewhere last week. I ran it through wordsim-353 and it got:

- Pearson's r: 0.5095
- Spearman's rho: 0.5169

Comparable to [Collobert and Weston (2008)](https://www.aclweb.org/aclwiki/index.php?title=WordSimilarity-353_Test_Collection_(State_of_the_art)), 
not bad? At least we know that it captured something.

```
>>> import gensim
>>> model = gensim.models.Word2Vec.load('output/gigaword')
>>> model.wv.evaluate_word_pairs('data/wordsim353_combined.tab')
((0.5095200162518948, 1.0200792645277232e-24), SpearmanrResult(correlation=0.51687561224435263, pvalue=1.6647396296963623e-25), 0.0)
>>> a=model.accuracy('data/questions-words.txt')
2017-05-15 11:12:17,899 : INFO : precomputing L2-norms of word weight vectors
2017-05-15 11:12:20,394 : INFO : capital-common-countries: 73.7% (373/506)
2017-05-15 11:12:29,047 : INFO : capital-world: 67.2% (1703/2536)
2017-05-15 11:12:29,664 : INFO : currency: 23.0% (41/178)
2017-05-15 11:12:35,885 : INFO : city-in-state: 9.3% (170/1822)
2017-05-15 11:12:36,933 : INFO : family: 92.8% (284/306)
2017-05-15 11:12:40,109 : INFO : gram1-adjective-to-adverb: 19.5% (181/930)
2017-05-15 11:12:41,284 : INFO : gram2-opposite: 26.0% (89/342)
2017-05-15 11:12:45,860 : INFO : gram3-comparative: 76.7% (1021/1332)
2017-05-15 11:12:47,913 : INFO : gram4-superlative: 75.0% (450/600)
2017-05-15 11:12:50,492 : INFO : gram5-present-participle: 68.7% (519/756)
2017-05-15 11:12:55,416 : INFO : gram6-nationality-adjective: 80.4% (1162/1445)
2017-05-15 11:13:00,457 : INFO : gram7-past-tense: 57.9% (858/1482)
2017-05-15 11:13:03,889 : INFO : gram8-plural: 72.8% (722/992)
2017-05-15 11:13:06,107 : INFO : gram9-plural-verbs: 58.3% (379/650)
2017-05-15 11:13:06,108 : INFO : total: 57.3% (7952/13877)
```

Notice: this model is trained on a compute node so you need to enter one to load it back into memory.
Otherwise it might throw "ImportError: No module named 'UserString'".

## Tue 16 May

Today I'll adapt gensim to train sense vectors. Found this 
[super helpful thread](https://groups.google.com/d/msg/gensim/LTdrGBysMyw/joGz8F9uCAAJ)
where some guy tried to modify it to reproduce Omer Levy's paper. Let me paste
the instructions here for later reference:

1. Ensure your Python environment is using your working-copy of gensim for the python & the `_inner` compiled code (typically `.so` shared-libraries) – this might involve invoking setup.py from inside your project directory, or doing a 'pip' install using a local path
2. When your changes to the `.pyx` files seem ready, use `cython` to compile them to `.c` code. (You *might* need to do this from the root of the project, eg: `cython gensim/models/word2vec_inner.pyx`)
3. Use the command `python ./setup.py build_ext --inplace`, from the root of your gensim directory, to compile the `.c` to shared-libraries. (Depending on how well you did step (1), you might also need to do something like `python ./setup.py install` to also install the shared-libraries elsewhere.)
4. Run your tests, confirming especially that your changed code (and not some older or elsewhere-installed version) is being run from where you expect it. Debug & repeat (2)-(3) as necessary. 
 
Modified the vocabulary code to recognize and register senses.
 
Went out to attent Antske's talk.

## Wed 17 May

Finished, checked Python implementation of new CBOW. When provided with a 
`sense_delimiter`, it will only train sense embeddings, leaving word 
(and context) embeddings untouched.
It is rather hard to write automated tests to check these requirement... 

TODO:

- Implement the C version of CBOW
- Put some kind of warning in Skip-gram part
- What to do with `score_sentence_cbow`?
- How to write automated test?

## Tue 23 May

BabelNet-WordNet mappings.

Don't know how to deal with the C implementation... For now I use only
the Python implementation for sense embeddings. 
Will it be fast enough?? I'll need to run on full data to find out.

## Tue 30 May

Marten has created a version of disambiguated Wikipedia. I fixed some small 
bugs and ran sense embedding script on it. This is after 1.5 hour:

```
2017-05-30 14:59:13,067 : INFO : PROGRESS: at 0.40% examples, 515 words/s, in_qsize 64, out_qsize 0
2017-05-30 14:59:16,951 : INFO : PROGRESS: at 0.40% examples, 516 words/s, in_qsize 64, out_qsize 0
2017-05-30 14:59:28,178 : INFO : PROGRESS: at 0.40% examples, 516 words/s, in_qsize 63, out_qsize 0
2017-05-30 14:59:32,155 : INFO : PROGRESS: at 0.40% examples, 517 words/s, in_qsize 64, out_qsize 0
2017-05-30 14:59:44,045 : INFO : PROGRESS: at 0.40% examples, 517 words/s, in_qsize 63, out_qsize 0
```

Apparently I need to complete that C implementation.

Hurraaaaah! I can't believe I managed to do this in an afternoon. 
The difference between C and Python is drastic.  

```
2017-05-30 17:15:33,480 : INFO : PROGRESS: at 0.95% examples, 92117 words/s, in_qsize 0, out_qsize 0
2017-05-30 17:15:34,536 : INFO : PROGRESS: at 0.96% examples, 92076 words/s, in_qsize 0, out_qsize 0
2017-05-30 17:15:35,546 : INFO : PROGRESS: at 0.98% examples, 92182 words/s, in_qsize 0, out_qsize 0
2017-05-30 17:15:36,584 : INFO : PROGRESS: at 0.99% examples, 92152 words/s, in_qsize 0, out_qsize 0
2017-05-30 17:15:37,636 : INFO : PROGRESS: at 1.00% examples, 92142 words/s, in_qsize 0, out_qsize 0
```

For now it's good that it runs. I will need to check the resulting embeddings
next week. Hopefully they make sense.

```
[minhle@fs0 wsd-with-marten]$ tail slurm-1411884.out
2017-05-30 17:43:44,800 : INFO : PROGRESS: at sentence #3380000, processed 7838951 words, keeping 799 word types
2017-05-30 17:43:44,898 : INFO : PROGRESS: at sentence #3390000, processed 7862696 words, keeping 799 word types
...
```














