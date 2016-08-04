#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'classify-eeg.py'
===============================================================================

This script feeds epoched EEG data into a classifier.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)

# TODO: try averaging different # of tokens (2,4,5) before training classifier?

from __future__ import division, print_function
import numpy as np
from numpy.lib.recfunctions import merge_arrays
from os import mkdir
from os import path as op
from pandas import DataFrame, read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.svm import LinearSVC

rand = np.random.RandomState(seed=0)

# flags
align = 'v'  # whether epochs are aligned to consonant (c) or vowel (v) onset
have_dss = True
use_dss = True
n_dss_channels_to_use = 1
classify_individ_subjs = False

# file i/o
paramdir = 'params'
outdir = 'processed-data'
infile = 'merged-eeg-data.npz'

# load feature table
feat_ref = read_csv(op.join(paramdir, 'reference-feature-table-cons.tsv'),
                    sep='\t', index_col=0, encoding='utf-8')

# load merged EEG data & other params
invars = np.load(op.join(outdir, infile))
epochs = invars['epochs']
events = invars['events']
langs = invars['langs']
foreign_langs = invars['foreign_langs']
feats = invars['feats']
cons = invars['cons']
test_mask = invars['test_mask']
train_mask = invars['train_mask']
validation_mask = invars['validation_mask']
epochs_cat = epochs[:, :n_dss_channels_to_use, :].reshape(epochs.shape[0], -1)

# more containers
classifier_dict = dict()
validation = list()
language_dict = {lang: list() for lang in foreign_langs}

# do across-subject LDA
print('training classifiers:')
for fname in feat_ref.columns:
    print('  {}'.format(fname), end=': ')
    lda_classif = LDA(solver='svd')
    lda_trained = lda_classif.fit(X=epochs_cat[train_mask],
                                  y=feats[fname][train_mask])
    # handle class names and dtypes for structured array
    dtype_names = ['{}{}'.format(['+', '-'][val], fname)
                   for val in lda_trained.classes_]
    dtype_forms = [float] * len(lda_trained.classes_)
    dtype_dict = dict(names=dtype_names, formats=dtype_forms)
    # validate on new English talkers
    eng_prob = lda_trained.predict_proba(epochs_cat[validation_mask])
    eng_prob = np.array([tuple(x) for x in eng_prob], dtype=dtype_dict)
    validation.append(eng_prob)
    # foreign sounds: classification results
    for lang in foreign_langs:
        print(lang, end=' ')
        lang_mask = langs == lang
        test_data = epochs_cat[(test_mask & lang_mask)]
        foreign_prob = lda_trained.predict_proba(test_data)
        foreign_prob = np.array([tuple(x) for x in foreign_prob],
                                dtype=dtype_dict)
        language_dict[lang].append(foreign_prob)
    print()
    # save classifier objects
    classifier_dict[fname] = lda_trained
np.savez(op.join(outdir, 'classifiers.npz'), **classifier_dict)

# convert to DataFrames and save
validation_df = DataFrame(merge_arrays(validation, flatten=True),
                          index=cons[validation_mask])
validation_df.to_csv(op.join(outdir, 'classifier-probabilities-eng.tsv'),
                     sep='\t')

for lang in foreign_langs:
    lang_mask = langs == lang
    test_probs_df = DataFrame(merge_arrays(language_dict[lang], flatten=True),
                              index=cons[(test_mask & lang_mask)])
    test_probs_df.to_csv(op.join(outdir, 'classifier-probabilities-{}.tsv'
                                 ''.format(lang)), sep='\t')
