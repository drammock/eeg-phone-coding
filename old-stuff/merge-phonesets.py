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
import json
import numpy as np
from os import path as op
from pandas import read_csv

# file i/o
paramdir = 'params'

# load list of languages
langs = np.load(op.join(paramdir, 'langs.npy'))
# load ASCII to IPA dictionary
with open(op.join(paramdir, 'ascii-to-ipa.json'), 'r') as ipafile:
    ipa = json.load(ipafile)
# compute which phones were heard
tokens = read_csv(op.join(paramdir, 'cv-boundary-times.tsv'), sep='\t')
tokens['lang'] = tokens['talker'].apply(lambda x: x[:3])
tokens['cons'] = tokens['consonant'].apply(lambda x: x[:-2] if x[-1] in
                                           ['0', '1', '2', '3'] else x)
tokens['ipa'] = tokens['cons'].apply(lambda x: ipa[x.replace('-', '_')])
phonesets = dict()
for lang in langs:
    phonesets[lang] = list(set(tokens.loc[tokens['lang'] == lang, 'ipa']))
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
all_phones = feat_tab.index.tolist()
# apply sort order to phonesets
for lang in langs:
    indices = [np.where(feat_tab.index.values.astype(unicode) == x)[0][0]
               for x in phonesets[lang]]
    phonesets[lang] = np.array(phonesets[lang])[np.argsort(indices)].tolist()
# save
with open(op.join(paramdir, 'phonesets.json'), 'w') as f, \
         open(op.join(paramdir, 'allphones.json'), 'w') as g:
    json.dump(phonesets, f)
    json.dump(all_phones, g)
