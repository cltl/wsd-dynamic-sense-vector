#!/bin/python3
'''
Created on 31 Oct 2017

Use this to get a version identifier for your experiment

@author: Minh Le
'''
from datetime import date
from subprocess import check_output

# git diff --staged is faster than git diff (of everything) so let's do it first
changed = (check_output('git diff', shell=True) != '')
if changed:
    version = date.today().strftime('%Y-%m-%d')
else:
    version = (check_output('git show -s --format=%ci', shell=True)[:10] +
               check_output('git show -s --format=-%h', shell=True).strip())
    
if __name__ == '__main__':
    print(version)