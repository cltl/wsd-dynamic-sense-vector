from time import time
import sys


def progress(it, ticks=1000000):
    start = time()
    for i, val in enumerate(it):
        yield(val)
        if (i+1) % ticks == 0:
            sys.stderr.write('processed %d items, elapsed time: %.1f minutes...\n' 
                             %(i+1, (time()-start)/60))
