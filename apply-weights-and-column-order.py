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
import numpy as np
import os.path as op
from pandas import read_csv

# file I/O
paramdir = 'params'
outdir = 'processed-data'

# load phonesets (used to standardize row/column order)
phonesets = np.load(op.join(paramdir, 'phonesets.npz'))
eng = phonesets['eng']
# load list of languages, put English last
foreign_langs = np.load(op.join(paramdir, 'foreign-langs.npy'))

# load data
confmats = dict()
weightmats = dict()
weightedmats = dict()
for lang in foreign_langs:
    # load feature-based confusion matrices
    fpath = op.join(outdir, 'features-confusion-matrix-{}.tsv').format(lang)
    confmat = read_csv(fpath, sep='\t', encoding='utf-8', index_col=0)
    # load eeg confusion matrices
    fpath = op.join(outdir, 'eeg-confusion-matrix-{}.tsv'.format(lang))
    weightmat = read_csv(fpath, sep='\t', index_col=0, encoding='utf-8')
    # make sure entries match
    assert set(confmat.index) == set(weightmat.index)
    assert set(confmat.columns) == set(weightmat.columns)
    # make sure orders match
    if not np.array_equal(confmat.index, weightmat.index):
        weightmat = weightmat.iloc[confmat.index, :]
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
    outfile = 'eeg-confusion-matrix-{}.tsv'.format(lang)
    weightmat.to_csv(op.join(outdir, outfile), sep='\t', encoding='utf-8')
    outfile = 'weighted-confusion-matrix-{}.tsv'.format(lang)
    weightedmat.to_csv(op.join(outdir, outfile), sep='\t', encoding='utf-8')
