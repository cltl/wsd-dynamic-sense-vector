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
