#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-feature-based-confusion-matrices.py'
===============================================================================

This script converts phonological feature values (output from a series of
binary feature classifiers) into segments bearing those feature values.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import json
import numpy as np
from os import path as op
from pandas import read_csv
from numpy import logical_not as negate

# file I/O
paramdir = 'params'
outdir = 'processed-data'

# load lists of languages and phones
langs = np.load(op.join(paramdir, 'langs.npy'))
lang_names = dict(hin='Hindi', swh='Swahili', hun='Hungarian', nld='Dutch',
                  eng='English')
with open(op.join(paramdir, 'phonesets.json'), 'r') as f:
    phonesets = json.load(f)
# load master features table
feat_tab = read_csv(op.join(paramdir, 'phoible-segments-features.tsv'),
                    sep='\t', encoding='utf-8', index_col=0)
sort_order = ['syllabic', 'consonantal', 'labial', 'coronal', 'dorsal',
              'continuant', 'sonorant', 'periodicGlottalSource', 'distributed',
              'strident']

# iterate over languages
for ix, lang in enumerate(langs):
    this_phones = phonesets[lang]
    # find which features are contrastive
    all_phones = list(set(this_phones + phonesets['eng']))
    all_feat_tab = feat_tab.loc[all_phones]
    vacuous = all_feat_tab.apply(lambda x: len(np.unique(x)) == 1).values
    privative = all_feat_tab.apply(lambda x: len(np.unique(x)) == 2 and
                                   '0' in np.unique(x)).values
    # create foreign & english subset tables
    this_feat_tab = feat_tab.loc[this_phones]
    this_feat_tab.index.name = lang
    assert this_feat_tab.shape[0] == len(this_phones)
    eng_feat_tab = feat_tab.loc[phonesets['eng']]
    eng_feat_tab.index.name = 'eng'
    assert eng_feat_tab.shape[0] == len(phonesets['eng'])
    # remove non-contrastive features
    this_feat_tab = this_feat_tab.iloc[:, negate(vacuous | privative)]
    eng_feat_tab = eng_feat_tab.iloc[:, negate(vacuous | privative)]
    # sort to group natural classes (makes confusion mattix look nicer)
    this_sort = [feat for feat in sort_order if feat in this_feat_tab.columns]
    eng_sort = [feat for feat in sort_order if feat in eng_feat_tab.columns]
    this_feat_tab = this_feat_tab.sort_values(by=this_sort, ascending=False)
    eng_feat_tab = eng_feat_tab.sort_values(by=eng_sort, ascending=False)
    """
    # make features binary (bad idea due to diphthong contour features)
    this_feat_tab = this_feat_tab.apply(lambda x: x == '+').astype(int)
    eng_feat_tab = eng_feat_tab.apply(lambda x: x == '+').astype(int)
    """
    # find feature distance
    mismatch = eng_feat_tab.apply(lambda x: np.sum(x.astype(str)[None, :] !=
                                                   this_feat_tab, axis=1),
                                  axis=1)
    # convert to confusion probability
    confusion_probability = np.exp(-1 * mismatch)
    # put English on horizontal axis
    confusion_probability = confusion_probability.T
    # save
    fpath = op.join(outdir, 'features-confusion-matrix-{}.tsv').format(lang)
    confusion_probability.to_csv(fpath, sep='\t', encoding='utf-8')
