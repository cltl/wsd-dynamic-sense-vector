#!/bin/python3
'''
Created on 31 Oct 2017

Use this to get a version identifier for your experiment

@author: Minh Le
'''
import os
from datetime import date

# git diff --staged is faster than git diff (of everything) so let's do it first
changed = (os.popen('git diff --staged').read().strip() != '' or 
           os.popen('git diff').read().strip() != '')
if changed:
    version = date.today().strftime('%Y-%m-%d')
else:
    version = (os.popen('git show -s --format=%ci').read()[:10] +
               os.popen('git show -s --format=-%h').read().strip())
    
if __name__ == '__main__':
    print(version)