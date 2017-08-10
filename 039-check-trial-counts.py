#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'prep-data.py'
===============================================================================

This script tabulates retained epochs by stimulus type, to assess imbalance in
the data that could have arisen by chance when dropping epochs.
"""
# @author: drmccloy
# Created on Tue Aug  8 16:59:00 PDT 2017
# License: BSD (3-clause)

import yaml
import mne
import numpy as np
import pandas as pd
import os.path as op
from os import mkdir
from aux_functions import merge_features_into_df

np.set_printoptions(precision=6, linewidth=160)
pd.set_option('display.width', 160)

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = 'processed-data'
paramdir = 'params'

# LOAD PARAMS FROM YAML
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    align_on_cv = analysis_params['align_on_cv']
    feature_systems = analysis_params['feature_systems']
    skip = analysis_params['skip']

# choice doesn't matter here; just need something to pass to the merge function
feature_sys_fname = feature_systems['jfh_dense']

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
ft = '-'.join(feature_sys_fname.split('.')[0].split('-')[-2:])
fname_suffix = cv + nc + ft

# load the trial params
df_cols = ['subj', 'talker', 'syll', 'train', 'wav_idx']
df_types = dict(subj=int, talker=str, syll=str, train=bool, wav_idx=int)
df = pd.read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
                 usecols=df_cols, dtype=df_types)
df = merge_features_into_df(df, paramdir, feature_sys_fname)
df['subj_code'] = (df['subj'] + 1).map({v: k for k, v in subjects.items()})
df = df[(['subj_code'] + df_cols + ['lang', 'ascii', 'ipa'])]

retained_rows = np.empty((0,))

# iterate over subjects
for subj_code, subj in subjects.items():
    if subj_code in skip:
        continue
    basename = '{0:03}-{1}-{2}'.format(subj, subj_code, cv)
    # load the data
    epochs = mne.read_epochs(op.join(indir, 'epochs', basename + 'epo.fif.gz'),
                             verbose=False)
    # reduce to just this subject (NB: df['subj'] 0-indexed, subj dict is not)
    this_df = df.loc[df['subj'] == (subjects[subj_code] - 1)]
    # convert epochs selection indices to dataframe row indices
    row_indices = this_df.index.values[epochs.selection]
    retained_rows = np.r_[retained_rows, row_indices]

# remove dropped epochs (trials)
reduced_df = df.iloc[retained_rows, :]

# counts by phoneme & by stimulus identity across all trials
for column, fname in zip(['ipa', 'syll'], ['phone-counts', 'stim-counts']):
    table = reduced_df.groupby([column, 'subj_code']).count()['subj'].unstack()
    table['min'] = table.apply(np.nanmin, axis=1)
    table['max'] = table.apply(np.nanmax, axis=1)
    table.to_csv(op.join(outdir, '{}.tsv'.format(fname)), sep='\t',
                 na_rep='', float_format='%.0f')

# TODO: repeat above after separating train/valid/test?
raise RuntimeError

# split into training-validation-testing
train_mask = reduced_df['train']
valid_mask = np.logical_not(reduced_df['train']) & (reduced_df['lang'] == 'eng')
test_mask = reduced_df['lang'] != 'eng'
masks = [train_mask, valid_mask, test_mask]
prefixes = ['train-', 'valid-', 'test-']

# counts by phoneme & by stimulus identity in the train/valid/test subsets
for column, fname in zip(['ipa', 'syll'], ['phone-counts', 'stim-counts']):
    for mask, prefix in zip(masks, prefixes):
        table = reduced_df.loc[mask].groupby([column, 'subj_code']).count()['subj'].unstack()
        table['min'] = table.apply(np.nanmin, axis=1)
        table['max'] = table.apply(np.nanmax, axis=1)
        table.to_csv(op.join(outdir, '{}{}.tsv'.format(prefix, fname)),
                     sep='\t', na_rep='', float_format='%.0f')
