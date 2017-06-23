'''
Filtering and printing a text file based on sentence length.

Usage:
  sentlen.py --min=<min> --max=<max> [<input>...]
    
'''

import sys
from docopt import docopt
import fileinput

if __name__ == '__main__':
    arguments = docopt(__doc__, options_first=True)
    min_len = int(arguments['--min'])
    max_len = int(arguments['--max'])
    for line in fileinput.input(arguments['<input>']): 
        l = len(line.split())
        if l >= min_len and l <= max_len:
            sys.stdout.write("%d\t%s" %(l, line))