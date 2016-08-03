#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'merge-phonesets.py'
===============================================================================

Helper script to merge phonesets into a dict; also creates a set of all unique
phones.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import numpy as np
from os import path as op
from glob import glob
from pandas import read_csv

# file i/o
paramdir = 'params'

# these phone sets are determined by the output of the probabilistic
# transcription system; may not correspond to the languages' phonemes
fnames = glob(op.join(paramdir, '*-phones.tsv'))
langs = [op.split(x)[1][:3] for x in fnames]
phonesets = dict()
for lang in langs:
    # TODO: need to get updated phone list (for Hindi) from MHJ / PJ
    this_phones = read_csv(op.join(paramdir, '{}-phones.tsv'.format(lang)),
                           encoding='utf-8', header=None)
    this_phones = np.squeeze(this_phones.values).astype(unicode).tolist()
    phonesets[lang] = this_phones

# make unique set of all phones in these languages
all_phones = [phone for inventory in phonesets.values() for phone in inventory]
all_phones = list(set(all_phones))

# impose consistent ordering
sort_by = ['syllabic',  # to first group vowels together
           'consonantal', 'labial', 'coronal', 'dorsal', 'continuant',
           'sonorant', 'periodicGlottalSource', 'distributed', 'strident']
# read in PHOIBLE feature data
feat_tab = read_csv(op.join(paramdir, 'phoible-segments-features.tsv'),
                    sep='\t', encoding='utf-8', index_col=0)
# make sure we have features for all phones
synonyms = {u'gː': u'ɡː', u'ɑɻ': u'ɻ', u'tʃː': u't̠ʃː', u'ç': u'ç'}
for lang in langs:
    for old, new in synonyms.items():
        # update phonesets with synonyms
        if old in phonesets[lang]:
            phonesets[lang][phonesets[lang].index(old)] = new
# update all_phones with synonyms
indices = np.where(np.in1d(all_phones, feat_tab.index, invert=True))[0]
missing = np.array(all_phones, dtype=unicode)[indices]
for ix in indices:
    all_phones[ix] = synonyms[all_phones[ix]]
assert np.all(np.in1d(all_phones, feat_tab.index))
# subset feature table to all_phones, sort table, sort all_phones
feat_tab = feat_tab.loc[all_phones]
feat_tab = feat_tab.sort_values(by=sort_by, ascending=False)
all_phones = feat_tab.index
# apply sort order to phonesets
for lang in langs:
    indices = [np.where(feat_tab.index.values.astype(unicode) == x)[0][0]
               for x in phonesets[lang]]
    phonesets[lang] = np.array(phonesets[lang])[np.argsort(indices)].tolist()
# create separate list of just the vowels
vowels = feat_tab[(feat_tab['consonantal'] == '-') &
                  (feat_tab['syllabic'] == '+')].index.values.astype(unicode)
# save
np.savez(op.join(paramdir, 'phonesets.npz'), **phonesets)
np.save(op.join(paramdir, 'allphones.npy'), all_phones)
np.save(op.join(paramdir, 'vowels.npy'), vowels)