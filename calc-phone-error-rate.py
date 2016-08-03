#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'calc-phone-error-rate-from-confusion-matrix.py'
===============================================================================

This script combines a feature-based confusion matrix with weights from
EEG-trained classifiers.
"""
# @author: drmccloy
# Created on Mon Aug  1 13:52:21 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import numpy as np
import os.path as op
from pandas import read_csv, DataFrame

# flags
ignore_vowels = True

# file I/O
paramdir = 'params'
outdir = 'processed-data'


def calc_err_prob(mat, ignore_vowels=True):
    if ignore_vowels:
        consonants = mat.index[np.in1d(mat.index.values, vowels, invert=True)]
        mat = mat.loc[consonants]
        eng_consonants = mat.columns[np.in1d(mat.columns.values, vowels,
                                             invert=True)]
        mat = mat[eng_consonants]
    mat = mat.div(mat.sum(axis=1), axis='index')  # normalize by row
    perfect_matches = mat.index[np.in1d(mat.index, mat.columns)
                                ].values.astype(unicode)
    err_prob = mat.apply(lambda x: 1 - x[x.name] if x.name in perfect_matches
                         else 1, axis=0, reduce=False)
    return err_prob


# load data
foreign_langs = np.load(op.join(paramdir, 'foreign-langs.npy'))
vowels = np.load(op.join(paramdir, 'vowels.npy'))
# loop over languages
err_probs = dict()
mean_err_prob = dict()
for lang in foreign_langs:
    # load feature-based confusion matrices
    fpath = op.join(outdir, 'features-confusion-matrix-{}.tsv').format(lang)
    confmat = read_csv(fpath, sep='\t', encoding='utf-8', index_col=0)
    # load eeg confusion matrices
    fpath = op.join(outdir, 'eeg-confusion-matrix-{}.tsv'.format(lang))
    weightmat = read_csv(fpath, sep='\t', index_col=0, encoding='utf-8')
    # calculate error probabilities
    feat_err_prob = calc_err_prob(confmat)
    weight_err_prob = calc_err_prob(weightmat)
    err_prob_df = DataFrame(dict(feature=feat_err_prob, eeg=weight_err_prob))
    # find mean EEG-minus-feature for each language
    mean_err_prob[lang] = (err_prob_df['eeg'] - err_prob_df['feature']).mean()
    err_probs[lang] = err_prob_df
out = DataFrame(mean_err_prob, columns=foreign_langs,
                index=['eeg_minus_feat']).T
# TODO: out.to_csv()