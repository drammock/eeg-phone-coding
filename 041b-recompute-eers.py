#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'get-eer.py'
===============================================================================

This script recomputes the equal error rate (and corresponding threshold) used
in the classification script, because they (foolishly) were not saved at that
step.
"""
# @author: drmccloy
# Created on Wed Aug 23 10:55:02 PDT 2017
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
import mne
from aux_functions import EER_threshold, merge_features_into_df

# basic file I/O
indir = 'eeg-data-clean'
outdir = 'processed-data'
paramdir = 'params'
feature_sys_fname = 'consonant-features-transposed-all-reduced.tsv'

# load params
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    align_on_cv = analysis_params['align_on_cv']
    features = analysis_params['features']
    skip = analysis_params['skip']
del analysis_params

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
datafile_suffix = 'redux-{}data.npy'.format(nc if do_dss else 'epoch-')

# load the trial params
df_cols = ['subj', 'talker', 'syll', 'train', 'wav_idx']
df_types = dict(subj=int, talker=str, syll=str, train=bool, wav_idx=int)
df = pd.read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
                 usecols=df_cols, dtype=df_types)
df = merge_features_into_df(df, paramdir, feature_sys_fname)
df = df[(df_cols + ['lang', 'ascii', 'ipa'] + features)]

# init containers
thresholds = {s: dict() for s in subjects}
eers = {s: dict() for s in subjects}

# loop over subjects
for subj_code, subj_num in subjects.items():
    if subj_code in skip:
        continue
    print(subj_code, end=' ')
    subj_outdir = op.join(outdir, subj_code)
    # load the epochs metadata
    basename = '{0:03}-{1}-{2}'.format(subj_num, subj_code, cv)
    epochs = mne.read_epochs(op.join(indir, 'epochs', basename + 'epo.fif.gz'),
                             verbose=False, preload=False)
    data = np.load(op.join(indir, 'time-domain-redux',
                           basename + datafile_suffix))
    event_ids = epochs.events[:, -1]
    # reduce to just this subj (NB: df['subj'] is 0-indexed, subj dict is not)
    this_df = df.loc[df['subj'] == (subj_num - 1)]
    # remove dropped epochs (trials). The `while` loop is skipped for most
    # subjects but should handle cases where the run was stopped and restarted,
    # by cutting out trials from the middle of `df` until
    # `df['wav_idx'].iloc[epochs.selection]` yields the stim IDs in `event_ids`
    match = this_df['wav_idx'].iloc[epochs.selection].values == event_ids
    unmatched = np.logical_not(np.all(match))
    while unmatched:
        first_bad_sel = np.where(np.logical_not(match))[0].min()
        first_bad = epochs.selection[first_bad_sel]
        mismatched_wavs = this_df['wav_idx'].iloc[first_bad:]
        mismatched_evs = event_ids[first_bad_sel:]
        new_start = np.where(mismatched_wavs == mismatched_evs[0])[0][0]
        this_df = pd.concat((this_df.iloc[:first_bad],
                             this_df.iloc[(first_bad + new_start):]))
        match = this_df['wav_idx'].iloc[epochs.selection].values == event_ids
        unmatched = np.logical_not(np.all(match))
    this_df = this_df.iloc[epochs.selection, :]
    assert np.array_equal(this_df['wav_idx'].values, event_ids)
    # make the data classifier-friendly
    train_mask = this_df['train']
    train_data = data[train_mask]
    for feat in features:
        print(feat, end=' ')
        # get labels
        train_labels = this_df.loc[train_mask, feat].values
        # handle sparse feature sets (that have NaN cells)
        valued = np.isfinite(train_labels)
        train_labels = train_labels[valued].astype(int)
        this_train_data = train_data[valued]
        # load the classifier
        clf_fname = 'classifier-{}-{}.npz'.format(cv + nc + feat, subj_code)
        clf = np.load(op.join(subj_outdir, clf_fname))
        clf = clf[clf.keys()[0]].tolist()
        # compute EER threshold (refits using best params from grid search)
        threshold, eer = EER_threshold(clf, X=this_train_data, y=train_labels,
                                       return_eer=True)
        thresholds[subj_code][feat] = threshold
        eers[subj_code][feat] = eer
    print()

eers = pd.DataFrame(eers)
thresholds = pd.DataFrame(thresholds)
eers.to_csv(op.join(outdir, 'eers.tsv'), sep='\t')
thresholds.to_csv(op.join(outdir, 'eer-thresholds.tsv'), sep='\t')
