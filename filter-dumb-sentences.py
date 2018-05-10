'''
This program filter out "dumb" sentences such as those that are too short or 
contain only numbers. 
'''
import fileinput
import re

if __name__ == '__main__':
    try:
        for line in fileinput.input():
            if re.search(r'\w', line):
                words = line.strip().split()
                if len(words) > 5:
                    print(' '.join(words))
    except (BrokenPipeError, KeyboardInterrupt):
        pass