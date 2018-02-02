#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-parallel-jobfile-pairwise.py'
===============================================================================

This script generates the command-line calls to classify.py. Each call will
train (with GridSearchCV) 1 classifier for 1 feature from 1 feature-system and
with data from 1 subject.
"""
# @author: drmccloy
# Created on Wed Dec 20 12:27:14 PST 2017
# License: BSD (3-clause)

import yaml
import os.path as op


# BASIC FILE I/O
paramdir = 'params'
outfile = op.join(paramdir, 'parallel-jobfile-pairwise.txt')
analysis_param_file = op.join(paramdir, 'current-analysis-settings.yaml')

# load params
with open(analysis_param_file, 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    canonical_phone_order = analysis_params['canonical_phone_order']

with open(outfile, 'w') as f:
    for subj_code in subjects:
        phones = canonical_phone_order['eng'][:]  # make a copy
        while len(phones) > 1:
            phone_one = phones.pop()
            for phone_two in phones:
                line = ('python 040b-classify-pairwise.py {} {} {}\n'
                        .format(subj_code, phone_one, phone_two))
                f.write(line)
