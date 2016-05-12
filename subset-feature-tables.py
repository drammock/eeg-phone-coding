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

# load trial params
df = read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
              usecols=['subj', 'syll'], dtype=dict(subj=int, syll=str))

# load feature table
feat_tab = read_csv(op.join(paramdir, 'phoible-segments-features.tsv'),
                    sep='\t', encoding='utf-8', index_col=0)

# reduce feature table to only the segments we need
feat_tab = feat_tab.iloc[np.in1d(feat_tab.index, ipa.values())]
assert feat_tab.shape[0] == len(ipa)

# remove any features that are fully redundant within the training set
eng_cons = np.unique(df['cons'].loc[df['lang'] == 'eng']
                     .apply(str.replace, args=('-', '_'))).astype(unicode)
eng_cons = [ipa[x] for x in eng_cons]
eng_feat_tab = feat_tab.loc[eng_cons]
eng_vacuous = eng_feat_tab.apply(lambda x: len(np.unique(x)) == 1).values
eng_privative = eng_feat_tab.apply(lambda x: len(np.unique(x)) == 2 and
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
feat_tab = feat_tab[nonredundant]
eng_feat_tab = eng_feat_tab[nonredundant]
# convert features to binary (discards distinction between neg. & unvalued)
if make_feats_binary:
    feat_tab = feat_tab.apply(lambda x: x == '+').astype(int)
    eng_feat_tab = eng_feat_tab.apply(lambda x: x == '+').astype(int)
    rec_dtypes = [(str(f), int) for f in feat_tab.columns]
else:
    # determine dtype for later use in structured arrays
    dty = 'a{}'.format(max([len(x) for x in np.unique(feat_tab).astype(str)]))
    rec_dtypes = [(str(f), dty) for f in feat_tab.columns]
# save reference feature tables
feat_tab.to_csv(op.join(paramdir, 'reference-feature-table.tsv'), sep='\t',
                encoding='utf-8')
eng_feat_tab.to_csv(op.join(paramdir, 'english-reference-feature-table.tsv'),
                    sep='\t', encoding='utf-8')
