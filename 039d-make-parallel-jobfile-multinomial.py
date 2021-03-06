#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-parallel-jobfile-multinomial.py'
===============================================================================

This script generates the command-line calls to classify-multinomial.py. Each
call will train (with GridSearchCV) 1 classifier for 1 consonant (one-vs-rest)
with data from 1 subject.
"""
# @author: drmccloy
# Created on Thu Jan 11 15:08:36 PST 2018
# License: BSD (3-clause)

import yaml
import os.path as op

# load params
paramdir = 'params'
analysis_param_file = op.join(paramdir, 'current-analysis-settings.yaml')
with open(analysis_param_file, 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']

# BASIC FILE I/O
outfile = op.join('jobfiles', 'parallel-jobfile-multinomial.txt')

with open(outfile, 'w') as f:
    for subj_code in subjects:
        line = f'python 040c-classify-multinomial.py {subj_code}\n'
        f.write(line)
