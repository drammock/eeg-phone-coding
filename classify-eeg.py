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
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC

rand = np.random.RandomState(seed=0)

# flags
align = 'v'  # whether epochs are aligned to consonant (c) or vowel (v) onset
have_dss = True
use_dss = True
n_dss_channels = 1

# TODO: try averaging various numbers of tokens (2, 4, 5) prior to training
# classifier
# TODO: try decomposing in the time domain (PCA / whiten / time-domain DSS)

# file i/o
eegdir = 'eeg-data-clean'
paramdir = 'params'

# load global params
subjects = np.load(op.join(paramdir, 'subjects.npz'))
df_cols = ['subj', 'talker', 'syll', 'train', 'wav_idx']
df_types = dict(subj=int, talker=str, syll=str, train=bool, wav_idx=int)
df = read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
              usecols=df_cols, dtype=df_types)
df['lang'] = df['talker'].apply(lambda x: x[:3])
df['valid'] = (df['lang'] == 'eng') & ~df['train']
df['test'] = ~(df['lang'] == 'eng')
# make sure every stimulus is either training, validation, or testing
# and make sure training, validation, & testing don't overlap
assert df.shape[0] == df['train'].sum() + df['valid'].sum() + df['test'].sum()
assert np.all(df[['train', 'valid', 'test']].sum(axis=1) == 1)
training_dict = {idx: bool_ for idx, bool_ in zip(df['wav_idx'], df['train'])}
validate_dict = {idx: bool_ for idx, bool_ in zip(df['wav_idx'], df['valid'])}
testing_dict = {idx: bool_ for idx, bool_ in zip(df['wav_idx'], df['test'])}

# construct mapping from event_id to consonant and language (do as loop instead
# of dict comprehension to make sure everything is one-to-one)
df['cons'] = df['syll'].apply(lambda x: x[:-2] if x.split('-')[-1] in
                              ('0', '1', '2') else x)
cons_dict = dict()
lang_dict = dict()
for key, cons, lang in zip(df['wav_idx'], df['cons'], df['lang']):
    if key in cons_dict.keys() and cons_dict[key] != cons:
        raise RuntimeError
    if key in lang_dict.keys() and lang_dict[key] != lang:
        raise RuntimeError
    cons_dict[key] = cons
    lang_dict[key] = lang

# init some global containers
epochs = []
events = []
subjs = []
cons = []

# read in cleaned EEG data
for subj_code, subj in subjects.items():
    basename = '{0:03}-{1}-{2}-aligned-'.format(int(subj), subj_code, align)
    if have_dss and use_dss:
        this_data = np.load(op.join(eegdir, basename + 'dssdata.npy'))
    else:
        this_epochs = mne.read_epochs(op.join(eegdir, basename + 'epo.fif.gz'),
                                      preload=True, proj=False, verbose=False)
        this_data = this_epochs.get_data()
        del this_epochs
        if use_dss:
            dss_mat = np.load(op.join(eegdir, basename + 'dssmat.npy'))
            this_data = np.einsum('ij,hjk->hik', dss_mat, this_data)
            del dss_mat
    # can't use mne.read_events with 0-valued event_ids
    this_events = np.loadtxt(op.join(eegdir, basename + 'eve.txt'),
                             dtype=np.float64).astype(int)[:, -1]
    this_cons = np.array([cons_dict[e] for e in this_events])
    this_train_mask = np.array([training_dict[e] for e in this_events])
    this_valid_mask = np.array([validate_dict[e] for e in this_events])
    this_test_mask = np.array([testing_dict[e] for e in this_events])
    this_lang = np.array([lang_dict[e] for e in this_events])
    # add current subj data to global container
    epochs.extend(this_data)
    events.extend(this_events)
    subjs.extend([subj] * this_data.shape[0])
    cons.extend(this_cons)
    # concatenate DSS components
    data_cat = this_data[:, :n_dss_channels, :].reshape(this_data.shape[0], -1)
    # do LDA
    lda_classifier = LDA(solver='svd')
    lda_trained = lda_classifier.fit(X=data_cat[this_train_mask],
                                     y=this_cons[this_train_mask])
    # validate
    english_validation = lda_trained.predict(data_cat[this_valid_mask])
    # english_prob = lda_trained.predict_proba(data_cat[this_valid_mask])
    n_correct = np.sum(this_cons[this_valid_mask] == english_validation)
    print('{}: {} / {} correct (LDA, {})'.format(subj_code, n_correct,
                                                 this_valid_mask.sum(),
                                                 'English'))
    # test
    foreign_langs = ', '.join(np.unique(this_lang[this_test_mask]))
    foreign_testing = lda_trained.predict(data_cat[this_test_mask])
    n_correct = np.sum(this_cons[this_test_mask] == foreign_testing)
    print('{}: {} / {} correct (LDA, {})'.format(subj_code, n_correct,
                                                 this_test_mask.sum(),
                                                 foreign_langs))
    '''
    # do SVM
    svm_classifier = LinearSVC(dual=False, class_weight='balanced',
                               random_state=rand)
    svm_trained = svm_classifier.fit(X=data_cat[this_train_mask],
                                     y=this_cons[this_train_mask])
    english_validation = svm_trained.predict(data_cat[this_valid_mask])
    n_correct = np.sum(this_cons[this_valid_mask] == english_validation)
    print('{}: {} / {} correct (SVM)'.format(subj_code, n_correct,
                                             this_valid_mask.sum()))
    '''
# convert global containers to arrays
epochs = np.array(epochs)
events = np.array(events)
subjs = np.squeeze(subjs)
cons = np.array(cons)
epochs_cat = epochs[:, :n_dss_channels, :].reshape(epochs.shape[0], -1)
train_mask = np.array([training_dict[e] for e in events])
valid_mask = np.array([validate_dict[e] for e in events])
test_mask = np.array([testing_dict[e] for e in events])

# do across-subject LDA
lda_classifier = LDA(solver='svd')
lda_trained = lda_classifier.fit(X=epochs_cat[train_mask], y=cons[train_mask])
# validate
english_validation = lda_trained.predict(epochs_cat[valid_mask])
n_correct = np.sum(cons[valid_mask] == english_validation)
print('{}: {} / {} correct (LDA, English)'.format('All subjs', n_correct,
                                                  valid_mask.sum()))
# test
foreign_testing = lda_trained.predict(epochs_cat[test_mask])
foreign_prob = lda_trained.predict_proba(epochs_cat[test_mask])
n_correct = np.sum(cons[test_mask] == foreign_testing)
print('{}: {} / {} correct (LDA, {})'.format('All subjs', n_correct,
                                             test_mask.sum(), 'All languages'))

"""
import matplotlib.pyplot as plt
for ev in np.unique(this_events)[:1]:
    ixs = this_events == ev
    plt.plot(data[ixs, 0, :].T, linewidth=0.5)
    plt.plot(data[ixs, 0, :].mean(0), linewidth=1.5)
"""
