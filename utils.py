from time import time
import sys


def progress(it, ticks=1000000, label='items', max_=None):
    start = time()
    for i, val in enumerate(it):
        yield(val)
        if (i+1) % ticks == 0:
            max_str = '' if max_ is None else ' of %d' %max_
            sys.stderr.write('processed %d%s %s, elapsed time: %.1f minutes...\n' 
                             %(i+1, max_str, label, (time()-start)/60))



def count_lines_fast(path, block_size=65536):
    '''
    Credit: glglgl (https://stackoverflow.com/a/9631635/217802)
    Might miss out the last line but it doesn't matter for a huge file such as Gigaword
    '''
    total_lines = 0
    with open(path, 'rb') as f:
        while True:
            bl = f.read(block_size)
            if not bl: break
            total_lines += bl.count(b"\n")
    return total_lines
