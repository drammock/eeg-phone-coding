#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'classify.py'
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
from os import mkdir
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import mne
from aux_functions import EER_score, EER_threshold, merge_features_into_df

rand = np.random.RandomState(seed=15485863)  # the one millionth prime

# command line args
subj_code = sys.argv[1]     # IJ, IQ, etc
this_feature = sys.argv[2]  # vocalic, consonantal, etc

# basic file I/O
indir = 'eeg-data-clean'
outdir = 'processed-data'
paramdir = 'params'
feature_sys_fname = 'all-features.tsv'
if not op.isdir(outdir):
    mkdir(outdir)

# initialize containers
subj_outdir = op.join(outdir, subj_code)
if not op.isdir(subj_outdir):
    mkdir(subj_outdir)
classifiers = dict()
probabilities = dict()
predictions = dict()

# load params
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    align_on_cv = analysis_params['align_on_cv']
    n_jobs = analysis_params['n_jobs']
pre_dispatch = '2*n_jobs'

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
basename = '{0:03}-{1}-{2}'.format(subjects[subj_code], subj_code, cv)
datafile_suffix = 'redux-{}data.npy'.format(nc if do_dss else 'epoch-')
fname_suffix = cv + nc + this_feature

# load the data
epochs = mne.read_epochs(op.join(indir, 'epochs', basename + 'epo.fif.gz'),
                         verbose=False)
data = np.load(op.join(indir, 'time-domain-redux', basename + datafile_suffix))
event_ids = epochs.events[:, -1]

# load the trial params
df_cols = ['subj', 'block', 'talker', 'syll', 'train', 'wav_idx']
df_types = dict(subj=int, block=int, talker=str, syll=str, train=bool,
                wav_idx=int)
df = pd.read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
                 usecols=df_cols, dtype=df_types)
df = merge_features_into_df(df, paramdir, feature_sys_fname)
df = df[(df_cols + ['lang', 'ascii', 'ipa', this_feature])]

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
train_labels = df.loc[train_mask, this_feature].values

# handle sparse feature sets (that have NaN cells)
valued = np.isfinite(train_labels)
train_labels = train_labels[valued].astype(int)
train_data = train_data[valued]

# hyperparameter grid search setup
param_grid = [dict(C=(2. ** np.arange(-5, 16)),
                   gamma=(2. ** np.arange(-15, 4)))]
clf_kwargs = dict(probability=True, kernel='rbf',
                  decision_function_shape='ovr', random_state=rand)
gridsearch_kwargs = dict(scoring=EER_score, n_jobs=n_jobs, refit=True,
                         pre_dispatch=pre_dispatch, cv=5, verbose=3)

# run gridsearch
classifier = SVC(**clf_kwargs)
clf = GridSearchCV(classifier, param_grid=param_grid, **gridsearch_kwargs)
clf.fit(X=train_data, y=train_labels)
classifiers['{}-{}'.format(subj_code, this_feature)] = clf

# compute EER threshold (refits using best params from grid search object)
threshold, eer = EER_threshold(clf, X=train_data, y=train_labels,
                               return_eer=True)
eer_fname = op.join(subj_outdir, 'eer-threshold-{}.tsv'.format(this_feature))
with open(eer_fname, 'w') as f:
    f.write('{}\t{}\t{}\t{}\n'.format('subj', 'feature', 'threshold', 'eer'))
    f.write('{}\t{}\t{}\t{}\n'.format(subj_code, this_feature, threshold, eer))

# test on new English talkers & foreign talkers
for lang in set(df['lang']):
    mask = np.logical_and((df['lang'] == lang), np.logical_not(df['train']))
    test_data = data[mask]
    probabilities[lang] = clf.predict_proba(test_data)
    predictions[lang] = (probabilities[lang][:, 1] >= threshold).astype(int)
    # convert probabilities & predictions to DataFrames and save
    ipa = df.loc[mask, 'ipa']
    df_out = pd.DataFrame(probabilities[lang], index=ipa)
    df_out.columns = ['{}{}'.format(['-', '+'][val], this_feature)
                      for val in np.unique(train_labels)]
    df_out[this_feature] = predictions[lang]
    df_out['lang'] = lang
    fname = 'classifier-probabilities-{}-{}-{}.tsv'.format(lang, fname_suffix,
                                                           subj_code)
    df_out.to_csv(op.join(subj_outdir, fname), sep='\t')

# save classifier objects
clf_fname = 'classifier-{}-{}.npz'.format(fname_suffix, subj_code)
np.savez(op.join(subj_outdir, clf_fname), **classifiers)
