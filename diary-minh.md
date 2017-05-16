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
