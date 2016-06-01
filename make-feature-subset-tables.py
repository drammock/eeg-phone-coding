#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:54:59 2016

@author: drmccloy
"""

import json
import numpy as np
import os.path as op
from pandas import read_csv

# flags
make_feats_binary = True

# file i/o
paramdir = 'params'

# import ASCII to IPA dictionary
with open(op.join(paramdir, 'ascii-to-ipa.json'), 'r') as ipafile:
    ipa = json.load(ipafile)

# load language phone sets
foreign_langs = np.load(op.join(paramdir, 'foreign-langs.npy'))
all_phones = np.load(op.join(paramdir, 'allphones.npy')).tolist()

# load trial params
df = read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
              usecols=['talker', 'syll'], dtype=dict(talker=str, syll=str))
df['lang'] = df['talker'].apply(lambda x: x[:3])
df['cons'] = df['syll'].apply(lambda x: x[:-2].replace('-', '_')
                              if x.split('-')[-1] in ('0', '1', '2')
                              else x.replace('-', '_'))

# load feature table
feat_tab = read_csv(op.join(paramdir, 'phoible-segments-features.tsv'),
                    sep='\t', encoding='utf-8', index_col=0)
# combine phone sets with consonants from EEG stimulus transcripts
all_phones = list(set(ipa.values() + all_phones))
# make sure we have features for all phones
assert np.all(np.in1d(all_phones, feat_tab.index))
# reduce feature table to only the segments we need
feat_tab_all = feat_tab.iloc[np.in1d(feat_tab.index, all_phones)]
feat_tab_cons = feat_tab.iloc[np.in1d(feat_tab.index, ipa.values())]

# remove any features that are fully redundant within the training set
cons_eng = np.unique(df['cons'].loc[df['lang'] == 'eng']
                     .apply(str.replace, args=('-', '_'))).astype(unicode)
cons_eng = [ipa[x] for x in cons_eng]
feat_tab_cons_eng = feat_tab_cons.loc[cons_eng]
eng_vacuous = feat_tab_cons_eng.apply(lambda x: len(np.unique(x)) == 1).values
eng_privative = feat_tab_cons_eng.apply(lambda x: len(np.unique(x)) == 2 and
                                        '0' in np.unique(x)).values
# other redundant features (based on linguistic knowledge, not easy to infer);
# here we list all that *could* be excluded, but only exclude features
# [round, front, back] because they cause problems in later analysis
# (they lead to a whole column of NaNs in the confusion matrices)
eng_redundant = ['round',           # (labial) w vs j captured by 'labial'
                 'front', 'back'    # (dorsal) w vs j captured by 'labial'
                 # 'delayedRelease',  # (manner) no homorganic stop/affr. pairs
                 # 'nasal',           # (manner) {m,n} captured by +son. -cont.
                 # 'lateral',         # (manner) l vs r captured by distrib.
                 # 'labiodental',     # (labial) no vless w to contrast with f
                 # 'anterior',        # (coronal) all non-anter. are +distrib.
                 ]
eng_redundant = np.in1d(feat_tab.columns, eng_redundant)
nonredundant = feat_tab.columns[~(eng_vacuous | eng_privative | eng_redundant)]
feat_tab_all = feat_tab_all[nonredundant]
feat_tab_cons = feat_tab_cons[nonredundant]
feat_tab_cons_eng = feat_tab_cons_eng[nonredundant]
# convert features to binary (discards distinction between neg. & unvalued)
if make_feats_binary:
    feat_tab_all = feat_tab_all.apply(lambda x: x == '+').astype(int)
    feat_tab_cons = feat_tab_cons.apply(lambda x: x == '+').astype(int)
    feat_tab_cons_eng = feat_tab_cons_eng.apply(lambda x: x == '+').astype(int)
# save reference feature tables
fname = op.join(paramdir, 'reference-feature-table-english.tsv')
feat_tab_cons_eng.to_csv(fname, sep='\t', encoding='utf-8')
feat_tab_cons.to_csv(op.join(paramdir, 'reference-feature-table-cons.tsv'),
                     sep='\t', encoding='utf-8')
feat_tab_all.to_csv(op.join(paramdir, 'reference-feature-table-all.tsv'),
                    sep='\t', encoding='utf-8')
