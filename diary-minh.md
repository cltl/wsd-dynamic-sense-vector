## Mon 8 May

Meeting with Marten: 
[notes](https://docs.google.com/document/d/1Y3RW_D2C1rgL0Gid9WBUR9ExYtq_Jd9AGGYaU-QtSYA/edit#)

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

## Fri 2 Jun

Met Marten, Piek and Jacopo.

Priority decided: Marten will focus on this project.

## Tue 6 Jun

Talked to Marten: 
[notes](https://docs.google.com/document/d/13ra6QYof8l1hKD8391cT_jK-78Mz1gSrc1uw5kL4uqk/edit).

## Wed 7 Jun

Examining the synset embeddings, they don't make sense to me whilst word 
embeddings are still good. Could it be that the synset embeddings weren't
trained after all?

```
--- Example: 09044862-n ---
Synset('united_states.n.01')
Nearest neighbors:
    Tomango 0.454
    construes   0.444
    192-mile    0.432
    --``If  0.420
    Temping 0.416
    20:2020 0.413
    Rautbord    0.409
    implimenting    0.408
    rideable    0.405
    Zilar   0.400
>> US
Nearest neighbors:
    Canada  0.695
    America 0.694
    Japan   0.668
    Europe  0.657
    Germany 0.656
    U.S.    0.655
    France  0.654
    UK  0.625
    Australia   0.620
    California  0.601
```

The focus now is to get LSTM running so synset embeddings will need to wait. 

## Tue 13 Jun

Finished data processing code. Now it's time to write the LSTM.

```
Divided into 3959 batches (1000195 elements each, std=23020, except last batch of 1000008)... Done.
Added 4363547 elements as padding (0.11%).
```

Found super useful guide 
[here](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/).

```
In [38]: # Create input data
    ...: X = np.random.randn(2, 10, 8)
    ...:
    ...: # The second example is of length 6
    ...: X[1,6:] = 0
    ...: X_lengths = [10, 6]
    ...:
    ...: cell = tf.contrib.rnn.LSTMCell(num_units=64, state_is_tuple=True)
    ...:
    ...: outputs, last_states = tf.nn.dynamic_rnn(
    ...:     cell=cell,
    ...:     dtype=tf.float64,
    ...:     sequence_length=X_lengths,
    ...:     inputs=X)
    ...:
    ...: result = tf.contrib.learn.run_n(
    ...:     {"outputs": outputs, "last_states": last_states},
    ...:     n=1,
    ...:     feed_dict=None)
    ...:
    ...: assert result[0]["outputs"].shape == (2, 10, 64)
    ...:
    ...: # Outputs for the second example past length 6 should be 0
    ...: assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()
    ...:
```

## Wed 14 Jun

Managed to train the LSTM model (and load it back, which is non-trivial!)

## Wed 21 Jun

Implemented "sampled" softmax of Jean et al. (2015):

Before:

```
Epoch: 10
batch #100 cost: 7.1678638
batch #200 cost: 7.1870413
Epoch: 10 -> Train cost: 7.264, elapsed time: 6.4 minutes
```

After:

```
Epoch: 10
batch #100 cost: 7.0222702
batch #200 cost: 6.9646182
Epoch: 10 -> Train cost: 6.980, elapsed time: 2.9 minutes
```

So the the time was cut in more than half and NLL is reduced (because the 
vocabulary size is smaller).

Added resampling of training batches so that each token has equal chance to 
become target and batch order is erased.

## Thu 22 Jun

The batching and padding function runs terribly slow. The last 1m sentences
took almost 1 hour to complete.

```
processed 172000000 items, elapsed time: 1037.6 minutes...
processed 173000000 items, elapsed time: 1079.0 minutes...
processed 174000000 items, elapsed time: 1119.9 minutes...
```

Logged in the node and found my script using all RAM, a lot of Swap and only 
3% CPU. An obvious memory thrashing case. 

```
%Cpu(s):  0.0 us,  0.1 sy,  0.0 ni, 96.9 id,  2.9 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 99.5/65879032 [|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||]
KiB Swap: 73.3/33554428 [|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||                        ]

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
 4592 minhle    20   0 82.706g 0.057t    824 D   3.3 92.9 208:45.24 python3
  246 root      20   0       0      0      0 S   0.7  0.0  75:19.89 kswapd0
  247 root      20   0       0      0      0 S   0.7  0.0   6:33.39 kswapd1
```

Attempted a dry run and found out that just loaded all sentences into memory
already cause the memory to overflow to swap. On disk, the pickle file is
11G. After loading into memory, it becomes 100G. And that was on login node,
the compute node has even smaller RAM. Probably I need to sort on disk instead
and convert everything into Numpy arrays as soon as I load it into memory.

**References**

Jean, S., Cho, K., Memisevic, R., & Bengio, Y. (2015). On Using Very Large Target Vocabulary for Neural Machine Translation. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 1–10). Beijing, China: Association for Computational Linguistics. Retrieved from http://www.aclweb.org/anthology/P15-1001

## Sat 24 Jun

The script finished. It's surprisingly quick.

```
[minhle@fs0 wsd-with-marten]$ wc -l output/gigaword.txt
175771829 output/gigaword.txt
[minhle@fs0 wsd-with-marten]$ wc -l output/gigaword.txt.sorted
122811493 output/gigaword.txt.sorted

[minhle@fs0 wsd-with-marten]$ grep "Train co" slurm-1489807.out
Epoch: 1 -> Train cost: 7.144, elapsed time: 35.4 minutes
Epoch: 2 -> Train cost: 6.848, elapsed time: 70.9 minutes
Epoch: 3 -> Train cost: 6.323, elapsed time: 106.3 minutes
Epoch: 4 -> Train cost: 5.966, elapsed time: 141.8 minutes
Epoch: 5 -> Train cost: 5.740, elapsed time: 177.3 minutes
Epoch: 6 -> Train cost: 5.629, elapsed time: 212.7 minutes
Epoch: 7 -> Train cost: 5.549, elapsed time: 248.2 minutes
Epoch: 8 -> Train cost: 5.481, elapsed time: 283.7 minutes
Epoch: 9 -> Train cost: 5.455, elapsed time: 319.2 minutes
Epoch: 10 -> Train cost: 5.395, elapsed time: 354.6 minutes
```

## Mon 26 Jun

[Met Piek](https://docs.google.com/document/d/1yBZDocxE1TVDmC4AGlV-kjmMNho2WzOzNo8EKcgfMg4/edit)

## Tue 27 Jun

Monitor dev cost and use it as a stopping criteria.

Report timeline following [this guide](https://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow/37774470#37774470).
Look like the program spends most of its time on GPU indeed. I don't know
if I can make it any better.

Started a new experiment with big corpus, big model (more embedding dims, more
hidden nodes).

## Wed 28 Jun

[Met Jacopo](https://docs.google.com/document/d/1bvERpN0ayxY972qYitBDyAc42w9z7IxcHSaOJ0H4J84/edit)

## Tue 4 Jul

Does it use GPU? I measured it and it did. Perhaps when Jacopo ran `nvidia-smi`,
it showed only *his* usage. 

```
[minhle@node001 ~]$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 370.28                 Driver Version: 370.28                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  On   | 0000:03:00.0     Off |                  N/A |
| 22%   41C    P2   162W / 250W |  11663MiB / 12206MiB |     66%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0     19977    C   python3                                      11661MiB |
+-----------------------------------------------------------------------------+
```

Testing trivial model (#ce39e02a5d34705a069915e29d51ebd7888ea649). It runs 
much faster indeed.

```
Epoch: 10 finished, elapsed time: 0.3 minutes
    Train cost: 0.123
    Dev cost: 11.142, hit@100: 0.0%
```

Compared to the normal model below, the trivial model is 8.67 times faster, i.e.
CPU operations take roughly 11% running time. This can be improved of course.

```
Epoch: 10 finished, elapsed time: 2.6 minutes
    Train cost: 7.148
    Dev cost: 7.906, hit@100: 0.5%
```

How to load batches of different shapes into memory and use them in the same
variable? I tested `tf.assign()` to see if it copies memory around and 
apparently it does.

```
import tensorflow as tf
var = tf.Variable(0.9)
var2 = tf.Variable(0.0)
copy_first_variable = var2.assign(var)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print('Initial value: %f' %sess.run(var2))
sess.run(copy_first_variable)
print('After assigning: %f' %sess.run(var2))
add1 = var2.assign_add(1)
sess.run(var2)
print('After adding: %f' %sess.run(add1))
print('Value of the source: %f' %sess.run(var))
```

Output:

```
Initial value: 0.000000
After assigning: 0.900000
After adding: 1.900000
Value of the source: 0.900000
```

An alternative approach is to store everything in a flattened array + batch
indices + batch sizes. We can sample the batch info and use it to locate the data.

```
import tensorflow as tf
import numpy as np
data = tf.Variable(np.random.rand(10000))
batch_info = tf.Variable(np.array([[0, 100, 20], [2000, 30, 30], [2900, 50, 100], [7900, 30, 70]]), dtype=tf.int32)
i, = tf.train.slice_input_producer([batch_info])
batch = tf.reshape(data[i[0]:i[0]+i[1]*i[2]], (i[1], i[2]))
col = tf.random_uniform((1,), maxval=tf.shape(batch)[1], dtype=tf.int32)
y = batch[:,col[0]]
sess = tf.Session()
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(100):
    batch_val, y = sess.run([batch, y])
    print('%s\t%s' %(batch_val.shape, y.shape))
coord.request_stop()
coord.join(threads)
sess.close()
```

Output:

```
(30, 30)
(30, 70)
(100, 20)
(50, 100)
(50, 100)
...
```

A problem with this approach is I can't sample batches according to length 
anymore so the loss will be distorted (shorter sentences are over-represented).
Let's see how it goes...

Hit a wall: https://github.com/tensorflow/tensorflow/issues/9506

I tested different solution but int32 isn't allowed into GPU and float32
doesn't work with `tf.nn.embedding_lookup`. 

```
with tf.device("/gpu:0"):
    a = tf.Variable([0,1,2], dtype=tf.float32)
    E = tf.Variable(np.random.rand(1000, 50))
    embs = tf.nn.embedding_lookup(E, a, validate_indices=False)
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(a)
```

## Wed 5 Jun

Meeting with Jacopo

TODO: prepare a giant zip file, send to Jacopo to run on big GPU
TODO: run the preloading code anyway

TODO: try different values of parallel_iterations in tf.nn.dynamic_rnn

TODO: monitor performance, guide: https://stackoverflow.com/questions/37751739/tensorflow-code-optimization-strategy/37774430#37774430

