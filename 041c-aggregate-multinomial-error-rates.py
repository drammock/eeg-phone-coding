#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'aggregate-multinomial-error-rates.py'
===============================================================================

This script aggregates the error rates from 1 textfile per subject into a
single table.
"""
# @author: drmccloy
# Created on Wed Aug 23 10:55:02 PDT 2017
# License: BSD (3-clause)

import yaml
from glob import glob
import os.path as op
import pandas as pd

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    scheme = analysis_params['classification_scheme']
del analysis_params

if scheme != 'multinomial':
    raise RuntimeError('this script is only for multinomial classifiers')

# BASIC FILE I/O
datadir = 'processed-data-{}'.format(scheme)
fname = 'classifier-probabilities-eng-*.tsv'
files = glob(op.join(datadir, 'classifiers', '??', fname))
files.sort()
errors = pd.DataFrame()

for f in files:
    subj_code = f.split('.')[0].split('-')[-1]
    this_probs = pd.read_csv(f, sep='\t', index_col=0)
    this_probs.drop(['lang'], axis=1, inplace=True)
    this_probs[subj_code] = (this_probs['prediction'] == this_probs.index)
    accuracy = this_probs[subj_code].groupby(this_probs.index).mean()
    errors = pd.concat([errors, 1 - accuracy], axis=1)

errors.to_csv(op.join(datadir, 'error_rates.tsv'), sep='\t')
