#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'learn-features-from-eeg.py'
===============================================================================

This script tries to learn optimal phonological features based on EEG data.
"""
# @author: drmccloy
# Created on Thu Aug  4 14:54:20 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import mne
import json
import numpy as np
from os import mkdir
from os import path as op
from pandas import read_csv
from numpy import logical_not as negate

# flags
align = 'v'  # whether epochs are aligned to consonant (c) or vowel (v) onset
have_dss = True
use_dss = False

# file i/o
eegdir = 'eeg-data-clean'
paramdir = 'params'
outdir = 'processed-data'
outfile = 'merged-dss-data.npz' if use_dss else 'merged-eeg-data.npz'
if not op.isdir(outdir):
    mkdir(outdir)

# load global params
subjects = np.load(op.join(paramdir, 'subjects.npz'))
df_cols = ['subj', 'talker', 'syll', 'train', 'wav_idx']
df_types = dict(subj=int, talker=str, syll=str, train=bool, wav_idx=int)
df = read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
              usecols=df_cols, dtype=df_types)
df['lang'] = df['talker'].apply(lambda x: x[:3])
df['valid'] = (df['lang'] == 'eng') & negate(df['train'])
df['test'] = negate(df['lang'] == 'eng')
foreign_langs = list(set(df['lang']) - set(['eng']))

# import ASCII to IPA dictionary
with open(op.join(paramdir, 'ascii-to-ipa.json'), 'r') as ipafile:
    ipa = json.load(ipafile)

# load feature table
feat_ref = read_csv(op.join(paramdir, 'reference-feature-table-cons.tsv'),
                    sep='\t', index_col=0, encoding='utf-8')

# determine dtype for later use in structured arrays
if isinstance(feat_ref.iloc[0, 0], int):  # feats are binary
    rec_dtypes = [(str(f), int) for f in feat_ref.columns]
else:
    dty = 'a{}'.format(max([len(x) for x in np.unique(feat_ref).astype(str)]))
    rec_dtypes = [(str(f), dty) for f in feat_ref.columns]

# make sure every stimulus is either training, validation, or testing
# and make sure training, validation, & testing don't overlap
assert df.shape[0] == df['train'].sum() + df['valid'].sum() + df['test'].sum()
assert np.all(df[['train', 'valid', 'test']].sum(axis=1) == 1)
training_dict = {idx: bool_ for idx, bool_ in zip(df['wav_idx'], df['train'])}
validate_dict = {idx: bool_ for idx, bool_ in zip(df['wav_idx'], df['valid'])}
testing_dict = {idx: bool_ for idx, bool_ in zip(df['wav_idx'], df['test'])}

# construct mapping from event_id to consonant and language (do as loop instead
# of dict comprehension to make sure everything is one-to-one)
df['cons'] = df['syll'].apply(lambda x: x[:-2].replace('-', '_')
                              if x.split('-')[-1] in ('0', '1', '2')
                              else x.replace('-', '_'))
cons_dict = dict()
lang_dict = dict()
for key, cons, lang in zip(df['wav_idx'], df['cons'], df['lang']):
    if key in cons_dict.keys() and cons_dict[key] != cons:
        raise RuntimeError
    if key in lang_dict.keys() and lang_dict[key] != lang:
        raise RuntimeError
    cons_dict[key] = cons.replace('-', '_')
    lang_dict[key] = lang

# init some global containers
epochs = list()
events = list()
subjs = list()
cons = list()
feats = list()
langs = list()
subj_feat_classifiers = dict()

# read in cleaned EEG data
print('reading data: subject', end=' ')
for subj_code, subj in subjects.items():
    print(str(subj), end=' ')
    basename = op.join(eegdir, '{0:03}-{1}-{2}-aligned-'
                       .format(int(subj), subj_code, align))
    if have_dss and use_dss:
        this_data = np.load(basename + 'dssdata.npy')
    else:
        this_epochs = mne.read_epochs(basename + 'epo.fif.gz', preload=True,
                                      proj=False, verbose=False)
        this_data = this_epochs.get_data()
        del this_epochs
        if use_dss:
            dss_mat = np.load(basename + 'dssmat.npy')
            this_data = np.einsum('ij,hjk->hik', dss_mat, this_data)
            del dss_mat
    # can't use mne.read_events with 0-valued event_ids
    this_events = np.loadtxt(basename + 'eve.txt', dtype=int)[:, -1]
    this_cons = np.array([cons_dict[e] for e in this_events])
    # convert phone labels to features (preserving trial order)
    this_feats = list()
    for con in this_cons:
        this_feat = [feat_ref[feat].loc[ipa[con]] for feat in feat_ref.columns]
        this_feats.append(this_feat)
    this_feats = np.array(this_feats, dtype=str)
    this_feats = np.array([tuple(x) for x in this_feats], dtype=rec_dtypes)
    # boolean masks for training / validation / testing
    this_train_mask = np.array([training_dict[e] for e in this_events])
    this_valid_mask = np.array([validate_dict[e] for e in this_events])
    this_test_mask = np.array([testing_dict[e] for e in this_events])
    this_lang = np.array([lang_dict[e] for e in this_events])
    # add current subj data to global container
    epochs.extend(this_data)
    events.extend(this_events)
    subjs.extend([subj] * this_data.shape[0])
    cons.extend(this_cons)
    feats.extend(this_feats)
    langs.extend(this_lang)
print()

# convert global containers to arrays
epochs = np.array(epochs)
events = np.array(events)
subjs = np.squeeze(subjs)
langs = np.array(langs)
cons = np.array(cons)
feats = np.array(feats, dtype=rec_dtypes)
train_mask = np.array([training_dict[e] for e in events])
validation_mask = np.array([validate_dict[e] for e in events])
test_mask = np.array([testing_dict[e] for e in events])

outvars = dict(epochs=epochs, events=events, subjs=subjs, langs=langs,
               cons=cons, feats=feats, train_mask=train_mask,
               validation_mask=validation_mask, test_mask=test_mask,
               foreign_langs=foreign_langs)

np.savez(op.join(outdir, outfile), **outvars)
