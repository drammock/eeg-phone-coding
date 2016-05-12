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

# file I/O
paramdir = 'params'
outdir = 'processed-data'

# load list of languages
foreign_langs = np.load(op.join(paramdir, 'foreign-langs.npy'))

# load ASCII to IPA dictionary
with open(op.join(paramdir, 'ascii-to-ipa.json'), 'r') as ipafile:
    ipa = json.load(ipafile)

# the phone set needed for the probabilistic transcription system:
eng_phones = [u'aɪ', u'aʊ', u'b', u'd', u'eɪ', u'f', u'h', u'iː', u'j',
              u'k', u'kʰ', u'l', u'm', u'n', u'oʊ', u'p', u'pʰ', u's',
              u't', u'tʃ', u'tʰ', u'u', u'v', u'w', u'x', u'z', u'æ', u'ð',
              u'ŋ', u'ɑ', u'ɔ', u'ɔɪ', u'ə', u'ɛ', u'ɛə', u'ɟʝ', u'ɡ',
              u'ɪ', u'ɫ', u'ɻ', u'ʃ', u'ʊ', u'ʌ', u'ʒ', u'θ']

# load master features table
feat_tab = read_csv(op.join(paramdir, 'phoible-segments-features.tsv'),
                    sep='\t', encoding='utf-8', index_col=0)

# sort order
sort_order = ['syllabic', 'consonantal', 'labial', 'coronal', 'dorsal',
              'continuant', 'sonorant', 'periodicGlottalSource', 'distributed',
              'strident']

# iterate over languages
for ix, lang in enumerate(foreign_langs):
    try:
        # TODO: need to get updated foreign phone lists from MHJ / PJ
        this_phones = read_csv(op.join(paramdir, '{}-phones.tsv'.format(lang)),
                               sep='\t', encoding='utf-8')
        this_phones = np.squeeze(this_phones).values.astype(unicode).tolist()
    except IOError:
        this_phones = eng_phones
    # find which features are contrastive
    all_phones = list(set(this_phones + eng_phones))
    all_feat_tab = feat_tab.loc[all_phones]
    vacuous = all_feat_tab.apply(lambda x: len(np.unique(x)) == 1).values
    privative = all_feat_tab.apply(lambda x: len(np.unique(x)) == 2 and
                                   '0' in np.unique(x)).values
    # create foreign & english subset tables
    this_feat_tab = feat_tab.loc[this_phones]
    this_feat_tab.index.name = lang
    assert this_feat_tab.shape[0] == len(this_phones)
    eng_feat_tab = feat_tab.loc[eng_phones]
    eng_feat_tab.index.name = 'eng'
    assert eng_feat_tab.shape[0] == len(eng_phones)
    # remove non-contrastive features
    this_feat_tab = this_feat_tab.iloc[:, ~(vacuous | privative)]
    eng_feat_tab = eng_feat_tab.iloc[:, ~(vacuous | privative)]
    # sort to group natural classes (makes confusion mattix look nicer)
    this_feat_tab = this_feat_tab.sort_values(by=sort_order, ascending=False)
    eng_feat_tab = eng_feat_tab.sort_values(by=sort_order, ascending=False)
    """
    # make features binary (bad idea due to diphthong contour features)
    this_feat_tab = this_feat_tab.apply(lambda x: x == '+').astype(int)
    eng_feat_tab = eng_feat_tab.apply(lambda x: x == '+').astype(int)
    """
    # find feature distance
    match = eng_feat_tab.apply(lambda x: np.sum(x.astype(str)[np.newaxis, :] ==
                                                this_feat_tab, axis=1), axis=1)
    # save
    fpath = op.join(outdir, 'features-confusion-matrix-{}.tsv').format(lang)
    match.to_csv(fpath, sep='\t', encoding='utf-8')
