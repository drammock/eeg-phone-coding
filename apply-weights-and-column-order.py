#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'apply-weights-and-column-order.py'
===============================================================================

This script combines a feature-based confusion matrix with weights from
EEG-trained classifiers.
"""
# @author: drmccloy
# Created on Mon Aug  1 14:04:26 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import yaml
import numpy as np
import os.path as op
from pandas import read_csv

# file I/O
paramdir = 'params'
outdir = 'processed-data'
analysis_params = 'current-analysis-settings.yaml'

# load analysis params
with open(op.join(paramdir, analysis_params), 'r') as paramfile:
    params = yaml.safe_load(paramfile)
clf_type = params['clf_type']
use_dss = params['dss']['use']
n_dss_channels_to_use = params['dss']['use_n_channels']
process_individual_subjs = params['process_individual_subjs']
fname_suffix = '-dss-{}'.format(n_dss_channels_to_use) if use_dss else ''
fname_id = '{}{}'.format(clf_type, fname_suffix)

# load phonesets (used to standardize row/column order)
phonesets = dict(np.load(op.join(paramdir, 'phonesets.npz')))
eng = read_csv(op.join(paramdir, 'eng-phones-superset.tsv'),
               encoding='utf-8', header=None)
eng = np.squeeze(eng.values).astype(unicode).tolist()
phonesets['eng'] = eng
# load list of languages, put English last
langs = np.load(op.join(paramdir, 'langs.npy'))

# load data
confmats = dict()
weightmats = dict()
weightedmats = dict()
for lang in langs:
    fid = '{}-{}'.format(lang, fname_id)
    # load feature-based confusion matrices
    fpath = op.join(outdir, 'features-confusion-matrix-{}.tsv').format(lang)
    confmat = read_csv(fpath, sep='\t', encoding='utf-8', index_col=0)
    # load eeg confusion matrices
    fpath = op.join(outdir, 'eeg-confusion-matrix-{}.tsv'.format(fid))
    weightmat = read_csv(fpath, sep='\t', index_col=0, encoding='utf-8')
    # make sure entries match
    assert set(confmat.index) == set(weightmat.index)
    assert set(confmat.columns) == set(weightmat.columns)
    # make sure orders match
    if not np.array_equal(confmat.index, weightmat.index):
        weightmat = weightmat.loc[confmat.index, :]
        assert np.array_equal(confmat.index, weightmat.index)
    if not np.array_equal(confmat.columns, weightmat.columns):
        weightmat = weightmat[confmat.columns]
        assert np.array_equal(confmat.columns, weightmat.columns)
    # combine
    weightedmat = (confmat / confmat.values.max() +
                   weightmat / weightmat.values.max()) / 2.
    # sort columns
    confmat = confmat[eng]
    weightmat = weightmat[eng]
    weightedmat = weightedmat[eng]
    # save
    outfile = 'features-confusion-matrix-{}.tsv'.format(lang)
    confmat.to_csv(op.join(outdir, outfile), sep='\t', encoding='utf-8')
    outfile = 'eeg-confusion-matrix-{}.tsv'.format(fid)
    weightmat.to_csv(op.join(outdir, outfile), sep='\t', encoding='utf-8')
    outfile = 'weighted-confusion-matrix-{}.tsv'.format(fid)
    weightedmat.to_csv(op.join(outdir, outfile), sep='\t', encoding='utf-8')
