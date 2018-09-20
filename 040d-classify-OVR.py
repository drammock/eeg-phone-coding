#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'classify-logistic.py'
===============================================================================

This script runs EEG data through a classifier, and stores the classifier
object as well as its classifications and (pseudo-)probabilities.
"""
# @author: drmccloy
# Created on Thu Aug  3 16:08:47 PDT 2017
# License: BSD (3-clause)

import sys
import yaml
import os.path as op
from os import makedirs
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import mne
from aux_functions import EER_score, EER_threshold, merge_features_into_df

rand = np.random.RandomState(seed=15485863)  # the one millionth prime

# command line args
subj_code = sys.argv[1]     # IJ, IQ, etc
this_consonant = sys.argv[2]  # consonant in IPA

# load params
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    align_on_cv = analysis_params['align_on_cv']
    n_jobs = analysis_params['n_jobs']
    truncate = analysis_params['eeg']['truncate']
pre_dispatch = '2*n_jobs'

# FILE NAMING VARIABLES
trunc = '-truncated' if truncate else ''
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
basename = '{0:03}-{1}-{2}'.format(subjects[subj_code], subj_code, cv)
datafile_suffix = 'redux-{}data.npy'.format(nc if do_dss else 'epoch-')
fname_suffix = cv + nc + this_consonant

# basic file I/O
indir = 'eeg-data-clean'
outdir = op.join(f'processed-data-OVR{trunc}', 'classifiers')
feature_sys_fname = 'all-features.tsv'
if not op.isdir(outdir):
    makedirs(outdir, exist_ok=True)

# initialize containers
subj_outdir = op.join(outdir, subj_code)
if not op.isdir(subj_outdir):
    makedirs(subj_outdir, exist_ok=True)
classifiers = dict()
probabilities = dict()
predictions = dict()

# load the data
epochs = mne.read_epochs(op.join(indir, f'epochs{trunc}',
                         basename + 'epo.fif.gz'), verbose=False)
data = np.load(op.join(indir, f'time-domain-redux{trunc}',
                       basename + datafile_suffix))
event_ids = epochs.events[:, -1]

# load the trial params
df_cols = ['subj', 'block', 'talker', 'syll', 'train', 'wav_idx']
df_types = dict(subj=int, block=int, talker=str, syll=str, train=bool,
                wav_idx=int)
df = pd.read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
                 usecols=df_cols, dtype=df_types)
df = merge_features_into_df(df, paramdir, feature_sys_fname)
df = df[(df_cols + ['lang', 'ascii', 'ipa'])]
df['label'] = (df['ipa'] == this_consonant)

# reduce to just this subject (NB: df['subj'] is 0-indexed, subj. dict is not)
df = df.loc[df['subj'] == (subjects[subj_code] - 1)]

# remove dropped epochs (trials). The `while` loop is skipped for most subjects
# but should handle cases where the run was stopped and restarted, by cutting
# out trials from the middle of `df` until
# `df['wav_idx'].iloc[epochs.selection]` yields the stim IDs in `event_ids`
match = df['wav_idx'].iloc[epochs.selection].values == event_ids
unmatched = np.logical_not(np.all(match))
# i = 0
while unmatched:
    # print('iteration {}; {} / {} matches'.format(i, match.sum(), len(match)))
    first_bad_sel = np.where(np.logical_not(match))[0].min()
    first_bad = epochs.selection[first_bad_sel]
    mismatched_wavs = df['wav_idx'].iloc[first_bad:]
    mismatched_evs = event_ids[first_bad_sel:]
    new_start = np.where(mismatched_wavs == mismatched_evs[0])[0][0]
    df = pd.concat((df.iloc[:first_bad],
                    df.iloc[(first_bad + new_start):]))
    match = df['wav_idx'].iloc[epochs.selection].values == event_ids
    unmatched = np.logical_not(np.all(match))
    # i += 1
df = df.iloc[epochs.selection, :]
assert np.array_equal(df['wav_idx'].values, event_ids)

# make the data classifier-friendly
train_mask = df['train']
train_data = data[train_mask]
train_labels = df.loc[train_mask, 'label'].values

# hyperparameter grid search setup
param_grid = [dict(C=(2. ** np.arange(-10, 16)))]
clf_kwargs = dict(tol=1e-4, solver='liblinear', random_state=rand)
gridsearch_kwargs = dict(scoring=EER_score, n_jobs=n_jobs, refit=True,
                         pre_dispatch=pre_dispatch, cv=5, verbose=3)

# run gridsearch
classifier = LogisticRegression(**clf_kwargs)
clf = GridSearchCV(classifier, param_grid=param_grid, **gridsearch_kwargs)
clf.fit(X=train_data, y=train_labels)
classifiers['{}-{}'.format(subj_code, this_consonant)] = clf

# compute EER threshold (refits using best params from grid search object)
threshold, eer = EER_threshold(clf, X=train_data, y=train_labels,
                               return_eer=True)
eer_fname = op.join(subj_outdir, 'eer-threshold-{}.tsv'.format(this_consonant))
with open(eer_fname, 'w') as f:
    f.write('{}\t{}\t{}\t{}\n'.format('subj', 'consonant', 'threshold', 'eer'))
    f.write('{}\t{}\t{}\t{}\n'.format(subj_code, this_consonant, threshold,
                                      eer))

# test on new English talkers & foreign talkers
for lang in set(df['lang']):
    mask = np.logical_and((df['lang'] == lang), np.logical_not(df['train']))
    test_data = data[mask]
    probabilities[lang] = clf.predict_proba(test_data)
    predictions[lang] = (probabilities[lang][:, 1] >= threshold).astype(int)
    # convert probabilities & predictions to DataFrames and save
    ipa = df.loc[mask, 'ipa']
    df_out = pd.DataFrame(probabilities[lang], index=ipa)
    df_out.columns = ['not_{}'.format(this_consonant), this_consonant]
    df_out['prediction'] = predictions[lang]
    df_out['lang'] = lang
    fname = 'classifier-probabilities-{}-{}-{}.tsv'.format(lang, fname_suffix,
                                                           subj_code)
    df_out.to_csv(op.join(subj_outdir, fname), sep='\t')

# save classifier objects
clf_fname = 'classifier-{}-{}.npz'.format(fname_suffix, subj_code)
np.savez(op.join(subj_outdir, clf_fname), **classifiers)
