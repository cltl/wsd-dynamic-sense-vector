#!/bin/python3
'''
Created on 31 Oct 2017

Use this to get a version identifier for your experiment

@author: Minh Le
'''
import os
from datetime import date

changed = os.popen('git ls-files -m').read().strip() != ''
if changed:
    version = date.today().strftime('%Y-%m-%d') + '-' + 'working'
else:
    version = (os.popen('git show -s --format=%ci').read()[:10] +
           os.popen('git show -s --format=-%h').read().strip())
    
if __name__ == '__main__':
    print(version)