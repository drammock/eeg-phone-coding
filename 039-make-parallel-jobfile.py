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
    feature_fnames = analysis_params['feature_fnames']
    feature_systems = analysis_params['feature_systems']

with open(outfile, 'w') as f:
    for subj_code in subjects:
        for feat_sys in feature_fnames:
            feat_list = feature_systems[feat_sys.split('_')[0]]
            for feat in feat_list:
                args = [subj_code, feat_sys, feat]
                line = 'python 040-classify.py {} {} {}\n'.format(*args)
                f.write(line)
