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

Met with Jacopo. Looks like computer science department can't solve our issue.
One solution is to write a short paper (or technical report) on reproducing
the Google paper. But first Jacopo would like to give it a shot.

Ran things on full data and the program start chunking out NaN. Turns out that
int32 is not big enough so I get negative indices.

```
>>> import numpy as np
>>> train=np.load('output/gigaword-lstm-wsd.train.npz')
>>> idx = train['indices']
>>> idx[-1]
array([-1123920531,         117,         101,   377887591,        3973, 3972], dtype=int32)
```

I converted everything into int64 and now the preparation script throws 
MemoryError :-( Now I need to take care of memory usage again.

Plan: use mem-map to perform linearization and serialization at the same time.
Write both sentences and vocabs into one vector (they are flattened anyway),
indices are written to an 2-D array.

## Thu 6 Jul

I should have realized that int64 is troublesome since I had seen [this 
discussion on Github](https://github.com/tensorflow/tensorflow/issues/9781) earlier. 
After solving all problems with saving and loading data, now I get this error:

> Input 'strides' of 'StridedSlice' Op has type int32 that does not match type int64 of argument 'begin'

This is how to reproduce it:

```
In [35]: import tensorflow as tf
    ...: a=tf.Variable([0,1,2], dtype=tf.int64)
    ...: i = tf.constant(1, dtype=tf.int64)
    ...: a[i]
    ...:
[...]
TypeError: Input 'strides' of 'StridedSlice' Op has type int32 that does not match type int64 of argument 'begin'.

In [36]: import tensorflow as tf
    ...: a=tf.Variable([0,1,2], dtype=tf.int32)
    ...: i = tf.constant(1, dtype=tf.int32)
    ...: a[i]
    ...:
Out[36]: <tf.Tensor 'strided_slice_22:0' shape=() dtype=int32>
```

Submitted an issue ([#11318](https://github.com/tensorflow/tensorflow/issues/11318)).

Rolled back everything related to preloading data. I hate it anyway since
the code is complex and awkward. Now I'll pack everything and send to Jacopo.

## Fri 7 Jul

The thing is running! Sent to Marten, Jacopo and Rutger. Next week I won't 
work on WSD to spend more time on coreference.

## Sat 8 Jul

The issue turns out to be solved 2 months ago. But I ran into 
[another issue](https://github.com/tensorflow/tensorflow/issues/11380). 

## Week of Mon 17 Jul

Got results from Marten. The "Google" model at ~10 epochs got 61% (WSD). This is
still lower than the baseline but better than the "large" model.

Jacopo suggested that we should write up a report.

I suggested that we train the "google" model on different sizes of the data
to make the case that we need 100B corpus to get Google's results.

## Tue 25 Jul

Fixed miscelleneous problems. Generalize the code. Fixed WSI.

## Wed 26 Jul

Submitted training on 1% and 10% of Gigaword. Scheduled to run for
maximally 2 weeks.

```
[minhle@fs0 wsd-with-marten]$ sbatch train-lstm-wsd-01pc-data-google-model.job
Submitted batch job 1529915
[minhle@fs0 wsd-with-marten]$ sbatch train-lstm-wsd-10pc-data-google-model.job
Submitted batch job 1529916
```

## Wed 2 Aug

Started training WSI on DAS-5 (large model). Scheduled to run for 2 weeks.

```
[minhle@fs0 wsd-with-marten]$ sbatch train-lstm-wsi.job
Submitted batch job 1532704
```

## Thu 3 Aug

All jobs died because of disk space :(( I should have used TensorFlow support 
for restarting training.

## Wed 16 Aug

Implemented managed session. Started training again. (I couldn't restore ) 

```
[minhle@fs0 wsd-with-marten]$ sbatch train-lstm-wsd-01pc-data-google-model.job
Submitted batch job 1538265
[minhle@fs0 wsd-with-marten]$ sbatch train-lstm-wsd-10pc-data-google-model.job
Submitted batch job 1538275
```

## Wed 6 Sep

Met Jacopo about my [WSI idea](https://www.overleaf.com/10943156fjxfxfpydpvn).
He liked it and, since me and Marten are both super busy, suggested that we 
can hire a student assistant or supervise a master thesis on this topic.

## Fri 8 Sep

Make an outline of the reimplementation paper. There are some numbers we need to
fill in.

## Mon 11 Sep

Help Marten to improve performance of his evaluation code.

Now I need to measure speedups.
 
Out of space. The prepared training data was removed a while ago (to get more
space apparently). I need to move `EvEn` to Cartesius and prepare the data
again... 

```
[minhle@fs0 wsd-with-marten]$ nohup python3 -u prepare-lstm-wsd.py output/gigaword.txt output/gigaword-lstm-wsd &
[1] 29077
$ zip EvEn.zip -r Even
```

Now just wait and wait and wait...

## Tue 12 Sep

Moved EvEn to Cartesius, got a little more space.

Strengthen reproducibility by versioning preprocessed data.

Started preprocessing data to measure speedups. I will need an unoptimized
version of the batches so it will take some time. 

```
[minhle@fs0 wsd-with-marten]$ ls -l preprocessed-data/40e2c1f
total 0
lrwxrwxrwx 1 minhle minhle 75 Sep 12 16:52 gigaword.txt -> /home/minhle/scratch/wsd-with-marten/preprocessed-data/694cb4d/gigaword.txt
lrwxrwxrwx 1 minhle minhle 82 Sep 12 16:52 gigaword.txt.sorted -> /home/minhle/scratch/wsd-with-marten/preprocessed-data/694cb4d/gigaword.txt.sorted
[minhle@fs0 wsd-with-marten]$ nohup python3 -u prepare-lstm-wsd.py > preprocessed-data/40e2c1f/prepare-lstm-wsd.py.out 2>&1 &
[1] 12220
[minhle@fs0 wsd-with-marten]$ tail -f preprocessed-data/40e2c1f/prepare-lstm-wsd.py.out
nohup: ignoring input
Building vocabulary...
processed 1000000 items, elapsed time: 0.3 minutes...
processed 2000000 items, elapsed time: 0.7 minutes...
processed 3000000 items, elapsed time: 1.0 minutes...
processed 4000000 items, elapsed time: 1.3 minutes...
```

Crashed due to disk space limit. Mailed Kees asking for more space.

## Wed 13 Sep

Kees increased my quota to 400G! Now I have enough to run my experiment.

Wasted another morning wrestling with TensorFlow. I had known that deleting
packages is a bad idea. Now my TensorFlow installation doesn't work anymore.
Emailed Kees asking about installing cuDNN v6.

Managed to run the preprocessing script anyway:

```
[minhle@fs0 wsd-with-marten]$ nohup python3 -u prepare-lstm-wsd.py > preprocessed-data/40e2c1f/prepare-lstm-wsd.py.out 2>&1 &
[1] 22530
[minhle@fs0 wsd-with-marten]$ tail -f preprocessed-data/40e2c1f/prepare-lstm-wsd.py.out
nohup: ignoring input
Reading vocabulary from preprocessed-data/40e2c1f/gigaword-for-lstm-wsd.index.pkl... Done.
Sentences are already sorted at preprocessed-data/40e2c1f/gigaword.txt.sorted
Dividing and padding...
processed 1000000 items, elapsed time: 0.2 minutes...
processed 2000000 items, elapsed time: 0.4 minutes...
processed 3000000 items, elapsed time: 0.6 minutes...
processed 4000000 items, elapsed time: 0.8 minutes...
processed 5000000 items, elapsed time: 1.0 minutes...
```

## Tue 17 Oct

Label propagation:

* http://scikit-learn.org/stable/modules/label_propagation.html
* https://github.com/yamaguchiyuto/label_propagation

Implemented a test script

## Thu 19 Oct

Turned the test script into a class for performing label propagation.

When I tested it with tiny data, one of labeled instance changes and the 
unlabeled instance gets its label. Isn't it strange?

    input: {'horse': array([0, 1, -1]), 'dog': array([0, 1, -1])}
    output: {'horse': array([1, 1, 0]), 'dog': array([1, 1, 0])}


TODO: try different values of parallel_iterations in tf.nn.dynamic_rnn

## Tue 21 Oct

Started working on experiments that would run on Cartesius: 
save all models, put results into versioned directories, vary the random seed.

## Wed 1 Nov

Took me a morning to setup Cartesius. Started the first step: preprocessing Gigaword.

    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/process-gigaword.job
    Submitted batch job 3718227
    [minhle@int1 wsd-dynamic-sense-vector]$ tail -f slurm-3718227.out

## Thu 2 Nov

Tried parallel training on Cartesius. It could only run one training script
at a time, all the rest failed and reported 4 different errors (!):
"Blas GEMM launch failed", "OOM when allocating tensor",
"failed initializing StreamExecutor for CUDA device", and
"Failed to create session".

Comparing the preprocessed version of Gigaword in DAS-5 and a (uncompleted)
preprocessed version of Gigaword in Cartesius and found that they are different.
 
Tried to parallelize BeautifulSoup, have negative effect: processing 10 documents
used to take 1.2 mins now increases to 1.6 mins. Moreover, processing the full 
737 files doesn't take up so much time anyway. The bottleneck turns out to be
spacy. Running its models The current version use threads instead of processes so the purported
parallelization actually does nothing. 

Why did I get bogged down to this optimization hell? The preprocessing runs for 
more than a day but so what? It also creates unstable output but I didn't find
any evidence of that leading to incorrect result. I wasted a day and now I wonder
if I created a new bug. Really, I shouldn't have gone down this road.

Submitted the job on Cartesius. It's pending...

and this is the reason:

> "This e-mail is to inform you that due to a vulnerability in the system software, the Cartesius system has been brought to an unscheduled maintenance mode. The batch system is being drained which means that your jobs, although they can still be committed to the queue, will not be executed. Please note that jobs that were already dispatched and being executed, have not been canceled.
> 
> We are currently in contact with our supplier for a solution. Our estimate is that the system will be up and running again by Tuesday morning, next week. Please find the actual system status on https://userinfo.surfsara.nl/systems/status."

## Sat 4 Nov

`prepare-lstm-wsd.job` finished. Now try `exp-variation.job`.

It started:

    [minhle@int2 wsd-dynamic-sense-vector]$ cat slurm-3725929.out
    Started: Sat Nov  4 00:37:02 CET 2017
    Running seed 124. Log is written to output/2017-11-04-a19751e/lstm-wsd-gigaword-small_seed-124.log

    [minhle@int2 wsd-dynamic-sense-vector]$ tail -f output/2017-11-04-a19751e/lstm-wsd-gigaword-small_seed-124.log
    ...
    Epoch #1:
        finished 1000 of 49551 batches, sample batch cost: 7.0678520

    [minhle@gcn42 ~]$ nvidia-smi
    Sat Nov  4 00:46:51 2017
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 375.26                 Driver Version: 375.26                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla K40m          On   | 0000:02:00.0     Off |                    0 |
    | N/A   45C    P0   119W / 235W |  10916MiB / 11439MiB |     83%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla K40m          On   | 0000:82:00.0     Off |                    0 |
    | N/A   36C    P0    62W / 235W |  10873MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID  Type  Process name                               Usage      |
    |=============================================================================|
    |    0     77394    C   python3                                      10912MiB |
    |    1     77394    C   python3                                      10869MiB |
    +-----------------------------------------------------------------------------+

## Wed 8 Nov

A big mistake: I didn't set `max_to_keep` (default to 5) so the saver removes 
my best model but keeps the latest one. 

Fixed it. This time I'll run a test
job before submitting. The last job took me 4 days and 1% of my budget.
 
Submitted `test.job`, waiting (Priority)... Done. It works. Now it's time to
restart `exp-variation.job`.

Submitted variation experiments. I divided the job file into two files because
in 5 days we can only maximally finish 5 seeds. 

    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-variation1.job
    Submitted batch job 3742501
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-variation2.job
    Submitted batch job 3742545
    [minhle@int1 wsd-dynamic-sense-vector]$ squeue | grep minh
               3742501       gpu exp-vari   minhle PD       0:00      1 (Priority)
               3742545       gpu exp-vari   minhle PD       0:00      1 (Priority)

Questions: 

- why does epoch number start at 1?
- how to use 2 GPUs?
- can the RAM fit 2 instances?

## Tue 14 Nov

5 days has ended. 8 experiments successfully completed, 2 were aborted.

    [minhle@int1 wsd-dynamic-sense-vector]$ ls output/2017-11-08-ce8a024/lstm-wsd-gigaword-small_seed*best-model.index
    output/2017-11-08-ce8a024/lstm-wsd-gigaword-small_seed-124-best-model.index
    output/2017-11-08-ce8a024/lstm-wsd-gigaword-small_seed-16651-best-model.index
    output/2017-11-08-ce8a024/lstm-wsd-gigaword-small_seed-261-best-model.index
    output/2017-11-08-ce8a024/lstm-wsd-gigaword-small_seed-293-best-model.index
    output/2017-11-08-ce8a024/lstm-wsd-gigaword-small_seed-651-best-model.index
    output/2017-11-08-ce8a024/lstm-wsd-gigaword-small_seed-91-best-model.index
    output/2017-11-08-ce8a024/lstm-wsd-gigaword-small_seed-9165-best-model.index
    output/2017-11-08-ce8a024/lstm-wsd-gigaword-small_seed-961-best-model.index

Now I'll evaluate them:

    sbatch cartesius/exp-variation-score.job
    
Error: `one_full_experiment_v2.sh: No such file or directory` --> mailed Marten.

Tried to measure speed with(out) some optimization tricks. Found out that the 
preprocessed data (for training) doesn't have length information. Need to rerun
preprocessing script again! (Maybe I should work around this... Changing the
script would mean more debugging and more waiting...)

## Wed 15 Nov

I looked at the 280 lines of code in `prepare-lstm-wsd.py` and didn't want to 
touch any of them. For the moment I change anything, I'll go down the rabbit 
hole and 1 day will pass very fast.

## Thu 16 Nov

Started a complex network of jobs on Cartesius. I hope that they all work...

    [minhle@int1 wsd-dynamic-sense-vector]$ git checkout a453bc1
    Previous HEAD position was a9618a6... make them runnable
    HEAD is now at a453bc1... run on Cartesius
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/prepare-lstm-wsd.job
    Submitted batch job 3764660
    [minhle@int1 wsd-dynamic-sense-vector]$ squeue | grep minh
               3764660    normal prepare-   minhle  R       0:11      1 tcn635
    [minhle@int1 wsd-dynamic-sense-vector]$ git checkout a9618a6
    Previous HEAD position was a453bc1... run on Cartesius
    HEAD is now at a9618a6... make them runnable
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/prepare-lstm-wsd.job
    Submitted batch job 3764663
    [minhle@int1 wsd-dynamic-sense-vector]$ squeue | grep minh
               3764663    normal prepare-   minhle  R       0:02      1 tcn721
               3764660    normal prepare-   minhle  R       1:25      1 tcn635
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch --dependency=afterok:3764660 cartesius/exp-variation-score.job
    Submitted batch job 3764664
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch --dependency=afterok:3764663 cartesius/exp-optimization1.job
    Submitted batch job 3764741
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch --dependency=afterok:3764663 cartesius/exp-optimization2.job
    Submitted batch job 3764742
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch --dependency=afterok:3764663 cartesius/exp-optimization3.job
    Submitted batch job 3764743
    [minhle@int1 wsd-dynamic-sense-vector]$ squeue | grep minh
               3764669       gpu exp-opti   minhle PD       0:00      1 (Dependency)
               3764741       gpu exp-opti   minhle PD       0:00      1 (Dependency)
               3764742       gpu exp-opti   minhle PD       0:00      1 (Dependency)
               3764743       gpu exp-opti   minhle PD       0:00      1 (Dependency)
               3764663    normal prepare-   minhle  R       7:54      1 tcn721
               3764660    normal prepare-   minhle  R       9:17      1 tcn635

Explanation:

- To run the scoring script for variation experiment, I need to rerun preparation script version a453bc1.
- To run optimization experiments, I need to run (for the first time) preparation script version a9618a6.
- I divided optimization experiments into smaller jobs to avoid hitting the 5-day limit,
all of them will start simultaneously.

## Mon 20 Nov

Somehow my jobs was killed early, actually on the same day I submitted them :-| 
I'll resubmit and contact Cartesius people. 

    [minhle@int1 wsd-dynamic-sense-vector]$ git checkout a453bc1
    Previous HEAD position was a9618a6... make them runnable
    HEAD is now at a453bc1... run on Cartesius
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/prepare-lstm-wsd.job
    Submitted batch job 3771725
    [minhle@int1 wsd-dynamic-sense-vector]$ squeue | grep minh
               3771725    normal prepare-   minhle  R       0:07      1 tcn618
    [minhle@int1 wsd-dynamic-sense-vector]$ git checkout a9618a6
    Checking out files: 100% (47/47), done.
    Previous HEAD position was a453bc1... run on Cartesius
    HEAD is now at a9618a6... make them runnable
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/prepare-lstm-wsd.job
    Submitted batch job 3771726
    [minhle@int1 wsd-dynamic-sense-vector]$ squeue | grep minh
               3771726    normal prepare-   minhle  R       0:04      1 tcn619
               3771725    normal prepare-   minhle  R       0:26      1 tcn618

## Tue 21 Nov

It was killed again, SURFsara hasn't replied.

Tested job files with `--time=1:00` and `--time 1:00` (without the equal sign).
They both seem to work properly and when they are canceled, there were a
message, for example:

    slurmstepd: *** JOB 3774760 ON tcn559 CANCELLED AT 2017-11-21T15:00:38 DUE TO TIME LIMIT ***

So my preprocessing script must have been killed for a different reason.


## Wed 22 Nov

Set for all NumPy arrays `dtype=np.int32`. Now the preparation script is running
without error while consuming 99% of RAM. I'll need to do memory map otherwise
I'll have no space for my model.

Marten and I fixed some bugs in Marten's evaluation script.

Tried loading shuffled dataset, it uses only around 40 GB (of 62.5 Gb) but I 
creating models resulted in a memory error. Luckily, GPU nodes have 96 GB so I
hope that's enough.

Forgot to disable subvocab creation for `sampled_softmax=False`. Did it now. 
Restarted experiment, only `exp-optimization1.job` for now.

    [minhle@int1 ~]$ squeue | grep minh
               3777791       gpu exp-opti   minhle  R      11:40      1 gcn39

## Thu 23 Nov

The training did work but super slowly. After 8h38m, it "finished 27000 of 157344 batches"
(17% of an epoch). At this speed it will take 19 days to finish 10 epochs. In
other words, it will consume all of my budget. I'll use this to estimate  

    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-optimization2.job
    Submitted batch job 3778073
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-optimization3.job
    Submitted batch job 3778074

Got these numbers for the variation experiment:

    [minhle@int1 output]$ tail -vn +1 results-seed-*/semcor/results.txt
    ==> results-seed-124/semcor/results.txt <==
    912
    ==> results-seed-16651/semcor/results.txt <==
    905
    ==> results-seed-261/semcor/results.txt <==
    914
    ==> results-seed-293/semcor/results.txt <==
    923
    ==> results-seed-651/semcor/results.txt <==
    946
    ==> results-seed-91/semcor/results.txt <==
    918
    ==> results-seed-9165/semcor/results.txt <==
    929
    ==> results-seed-961/semcor/results.txt <==
    911
    
    [minhle@int1 output]$ tail -vn +1 results-seed-*/mun/results.txt
    ==> results-seed-124/mun/results.txt <==
    989
    ==> results-seed-16651/mun/results.txt <==
    984
    ==> results-seed-261/mun/results.txt <==
    989
    ==> results-seed-293/mun/results.txt <==
    994
    ==> results-seed-651/mun/results.txt <==
    1009
    ==> results-seed-91/mun/results.txt <==
    985
    ==> results-seed-9165/mun/results.txt <==
    1004
    ==> results-seed-961/mun/results.txt <==
    990
 
An experiment failed because I forgot to copy one preprocessed data file to the
right folder. Restarted:

    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-optimization2.job
    Submitted batch job 3778716

Restarted yet again because subvocabs weren't prepared even for sampled softmax.

    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-h256p64.job
    Submitted batch job 3779735
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-optimization2.job
    Submitted batch job 3779738
    [minhle@int1 wsd-dynamic-sense-vector]$ sbatch cartesius/exp-optimization3.job
    Submitted batch job 3779739

5. [x] Try out scikit implementation 
4. [x] Implement custom kernel ("We found that the graph has about the right density for common senses when ... between 85 to 98.")1. [ ] Input and output for SemCor and OMSTI
3. [x] measure running time
2. [ ] Query UkWaC??? for sentences that contain a certain lemma
3. [x] Build graph and run Scikit label propagation
4. [x] for 25 Oct: label prop. Semcor + OMSTI
5. [x] for 25 Oct: list of all experiments for the reproduction paper
6. [x] save models of every epoch (instead of only the best one)
6. [x] Read more about label propagation (Zhou et al. 2004)
