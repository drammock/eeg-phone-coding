#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-parallel-jobfile.py'
===============================================================================

This script generates the command-line calls to classify.py. Each call will
train (with GridSearchCV) 1 classifier for 1 feature from 1 feature-system and
with data from 1 subject.
"""
# @author: drmccloy
# Created on Mon Aug 14 15:28:54 PDT 2017
# License: BSD (3-clause)

import yaml
import os.path as op

# BASIC FILE I/O
paramdir = 'params'
outfile = op.join(paramdir, 'parallel-jobfile.txt')
analysis_param_file = op.join(paramdir, 'current-analysis-settings.yaml')

# load params
with open(analysis_param_file, 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    features = analysis_params['features']

with open(outfile, 'w') as f:
    for subj_code in subjects:
        for feat in features:
            line = 'python 040-classify.py {} {}\n'.format(subj_code, feat)
            f.write(line)