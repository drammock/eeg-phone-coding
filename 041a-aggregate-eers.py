#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'aggregate-eers.py'
===============================================================================

This script aggregates the EER values from 1 textfile per subject/feature into
a single table.
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
    truncate = analysis_params['eeg']['truncate']
del analysis_params

# FILE NAMING VARIABLES
trunc = '-truncated' if truncate else ''

if scheme == 'multinomial':
    raise RuntimeError('no EERs with multinomial method')
varname = dict(OVR='consonant', pairwise='contrast').get(scheme, 'feature')

# BASIC FILE I/O
datadir = f'processed-data-{scheme}{trunc}'
eer_files = glob(op.join(datadir, 'classifiers', '??', 'eer-threshold-*.tsv'))
eer_files.sort()
eers = pd.DataFrame()
dtypes = dict(subj=str, threshold=float, eer=float)
dtypes[varname] = str

for f in eer_files:
    this_eer = pd.read_csv(f, sep='\t', dtype=dtypes)
    eers = pd.concat([eers, this_eer])

thresholds = eers.pivot(index=varname, columns='subj', values='threshold')
eers = eers.pivot(index=varname, columns='subj', values='eer')

eers.to_csv(op.join(datadir, 'eers.tsv'), sep='\t')
thresholds.to_csv(op.join(datadir, 'eer-thresholds.tsv'), sep='\t')
