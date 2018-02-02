#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-parallel-jobfile-OVR.py'
===============================================================================

This script generates the command-line calls to classify-OVR.py. Each call will
train (with GridSearchCV) 1 classifier for 1 consonant (one-vs-rest) with data
from 1 subject.
"""
# @author: drmccloy
# Created on Thu Jan 11 15:08:36 PST 2018
# License: BSD (3-clause)

import yaml
import os.path as op

# BASIC FILE I/O
paramdir = 'params'
outfile = op.join(paramdir, 'parallel-jobfile-OVR.txt')
analysis_param_file = op.join(paramdir, 'current-analysis-settings.yaml')

# load params
with open(analysis_param_file, 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    phones = analysis_params['canonical_phone_order']['eng']

with open(outfile, 'w') as f:
    for subj_code in subjects:
        for phone in phones:
            line = ('python 040c-classify-OVR.py {} {}\n'
                    .format(subj_code, phone))
            f.write(line)
