#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'classify-eeg.py'
===============================================================================

This script feeds epoched EEG data into a classifier.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)


from __future__ import division, print_function
import mne
import numpy as np
# from os import mkdir
from os import path as op
# from expyfun import binary_to_decimals
from pandas import read_csv
# from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

rand = np.random.RandomState(seed=0)

# file i/o
eegdir = 'eeg-data-clean'
paramdir = 'params'
# cvfile = 'cv-boundary-times.tsv'

# load global params
subjects = np.load(op.join(paramdir, 'subjects.npz'))
# params = np.load(op.join(paramdir, 'global-params.npz'))
# wav_names = params['wav_names']
# del params
df_cols = ['subj', 'talker', 'syll', 'train', 'wav_path', 'wav_idx']
df_types = dict(subj=int, talker=str, syll=str, train=bool, wav_path=str,
                wav_idx=int)
df = read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
              usecols=df_cols, dtype=df_types)
df['eng'] = df.talker.apply(lambda x: x[:3] == 'eng')
df['valid'] = df['eng'] & ~df['train']
df['test'] = ~df['eng']
# make sure training, validation, & testing data don't overlap
assert np.all(df[['train', 'valid', 'test']].sum(axis=1) == 1)
training_dict = {idx: bool_ for idx, bool_ in zip(df['wav_idx'], df['train'])}
validate_dict = {idx: bool_ for idx, bool_ in zip(df['wav_idx'], df['valid'])}
testing_dict = {idx: bool_ for idx, bool_ in zip(df['wav_idx'], df['test'])}
"""
# load C-V transition times
df = read_csv(cvfile, sep='\t')
df['key'] = df['talker'] + '/' + df['consonant'] + '.wav'
df['key'] = df['key'].apply(unicode)
assert set(wav_names) - set(df['key']) == set([])  # have a time for all keys?
"""

epochs = []
events = []
# read in cleaned EEG data
for subj_code, subj in subjects.items():
    basename = '{0:03}-{1}-'.format(int(subj), subj_code)
    this_epochs = mne.read_epochs(op.join(eegdir, basename + 'epo.fif.gz'),
                                  preload=True, proj=False)
    epochs.extend(this_epochs.get_data())
    events.extend(this_epochs.events[:, -1])
epochs = np.array(epochs)
events = np.array(events)
# compute masks for training, validation, and test
training_idx = np.array([training_dict[event_id] for event_id in events])
validate_idx = np.array([validate_dict[event_id] for event_id in events])
testing_idx = np.array([testing_dict[event_id] for event_id in events])

# LDA
lda_classifier = LDA(solver='svd', store_covariance=True)
lda_classifier.fit(epochs[training_idx], events[training_idx])
english_validation = lda_classifier.predict(epochs[validate_idx])
foreign_prediction = lda_classifier.predict(epochs[testing_idx])

"""
# SVM (won't work; requires 2-D input)
svm_classifier = LinearSVC(kernel='linear', class_weight='balanced',
                           decision_function_shape='ovr', random_state=rand)
svm_classifier.fit(epochs[training_idx], events[training_idx])
english_validation = svm_classifier.predict(epochs[validate_idx])
foreign_prediction = svm_classifier.predict(epochs[testing_idx])
"""
