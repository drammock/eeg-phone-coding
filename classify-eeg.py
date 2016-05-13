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

# TODO: try averaging different # of tokens (2,4,5) before training classifier?

from __future__ import division, print_function
import mne
import json
import numpy as np
from numpy.lib.recfunctions import merge_arrays
from os import mkdir
from os import path as op
from pandas import DataFrame, read_csv
from mne_sandbox.preprocessing._dss import _pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.svm import LinearSVC

rand = np.random.RandomState(seed=0)

# flags
align = 'v'  # whether epochs are aligned to consonant (c) or vowel (v) onset
have_dss = True
use_dss = True
n_dss_channels = 1
classify_individ_subjs = False

# file i/o
eegdir = 'eeg-data-clean'
paramdir = 'params'
outdir = 'processed-data'
if not op.isdir(outdir):
    mkdir(outdir)

# load global params
subjects = np.load(op.join(paramdir, 'subjects.npz'))
df_cols = ['subj', 'talker', 'syll', 'train', 'wav_idx']
df_types = dict(subj=int, talker=str, syll=str, train=bool, wav_idx=int)
df = read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
              usecols=df_cols, dtype=df_types)
df['lang'] = df['talker'].apply(lambda x: x[:3])
df['valid'] = (df['lang'] == 'eng') & ~df['train']
df['test'] = ~(df['lang'] == 'eng')
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
    # reduce dimensionality of time domain with PCA
    time_cov = np.sum([np.dot(trial.T, trial) for trial in this_data], axis=0)
    eigval, eigvec = _pca(time_cov, max_components=60)
    W = np.sqrt(1 / eigval)  # whitening diagonal
    this_data = np.array([np.dot(trial, eigvec) * W[np.newaxis, :]
                          for trial in this_data])
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
    if classify_individ_subjs:
        # concatenate DSS components
        data_cat = this_data[:, :n_dss_channels, :].reshape(this_data.shape[0],
                                                            -1)
        # do LDA
        feat_classifiers = dict()
        for fname in feat_ref.columns:
            lda_classif = LDA(solver='svd')
            lda_trained = lda_classif.fit(X=data_cat[this_train_mask],
                                          y=this_feats[fname][this_train_mask])
            feat_classifiers[fname] = lda_trained
            # validate
            eng_validate = lda_trained.predict(data_cat[this_valid_mask])
            # eng_prob = lda_trained.predict_proba(data_cat[this_valid_mask])
            n_corr = np.sum(this_feats[fname][this_valid_mask] == eng_validate)
            print('{}: {} / {} correct ({}, {})'.format(subj_code, n_corr,
                                                        this_valid_mask.sum(),
                                                        fname, 'English'))
            # test
            foreign_langs = ', '.join(np.unique(this_lang[this_test_mask]))
            foreign_test = lda_trained.predict(data_cat[this_test_mask])
            n_corr = np.sum(this_feats[fname][this_test_mask] == foreign_test)
            print('{}: {} / {} correct ({}, {})'.format(subj_code, n_corr,
                                                        this_test_mask.sum(),
                                                        fname, foreign_langs))
            '''
            # do SVM
            svm_classifier = LinearSVC(dual=False, class_weight='balanced',
                                       random_state=rand)
            svm_trained = svm_classifier.fit(X=data_cat[this_train_mask],
                                             y=this_cons[this_train_mask])
            eng_validate = svm_trained.predict(data_cat[this_valid_mask])
            n_corr = np.sum(this_cons[this_valid_mask] == eng_validate)
            print('{}: {} / {} correct (SVM)'.format(subj_code, n_corr,
                                                     this_valid_mask.sum()))
            '''
        subj_feat_classifiers[subj_code] = feat_classifiers
print()

# convert global containers to arrays
epochs = np.array(epochs)
events = np.array(events)
subjs = np.squeeze(subjs)
langs = np.array(langs)
cons = np.array(cons)
feats = np.array(feats, dtype=rec_dtypes)
epochs_cat = epochs[:, :n_dss_channels, :].reshape(epochs.shape[0], -1)
train_mask = np.array([training_dict[e] for e in events])
valid_mask = np.array([validate_dict[e] for e in events])
test_mask = np.array([testing_dict[e] for e in events])

# more containers
classifier_dict = dict()
validation = list()
language_dict = {lang: list() for lang in foreign_langs}

# do across-subject LDA
print('training classifiers:')
for fname in feat_ref.columns:
    print('  {}'.format(fname), end=': ')
    lda_classif = LDA(solver='svd')
    lda_trained = lda_classif.fit(X=epochs_cat[train_mask],
                                  y=feats[fname][train_mask])
    # handle class names and dtypes for structured array
    dtype_names = ['{}{}'.format(['+', '-'][val], fname)
                   for val in lda_trained.classes_]
    dtype_forms = [float] * len(lda_trained.classes_)
    dtype_dict = dict(names=dtype_names, formats=dtype_forms)
    # validate on new English talkers
    eng_prob = lda_trained.predict_proba(epochs_cat[valid_mask])
    eng_prob = np.array([tuple(x) for x in eng_prob], dtype=dtype_dict)
    validation.append(eng_prob)
    # foreign sounds: classification results
    for lang in foreign_langs:
        print(lang, end=' ')
        lang_mask = langs == lang
        test_data = epochs_cat[(test_mask & lang_mask)]
        foreign_prob = lda_trained.predict_proba(test_data)
        foreign_prob = np.array([tuple(x) for x in foreign_prob],
                                dtype=dtype_dict)
        language_dict[lang].append(foreign_prob)
    print()
    # save classifier objects
    classifier_dict[fname] = lda_trained
np.savez(op.join(outdir, 'classifiers.npz'), **classifier_dict)

# convert to DataFrames and save
validation_df = DataFrame(merge_arrays(validation, flatten=True),
                          index=cons[valid_mask])
validation_df.to_csv(op.join(outdir, 'classifier-probabilities-eng.tsv'),
                     sep='\t')

for lang in foreign_langs:
    lang_mask = langs == lang
    test_probs_df = DataFrame(merge_arrays(language_dict[lang], flatten=True),
                              index=cons[(test_mask & lang_mask)])
    test_probs_df.to_csv(op.join(outdir, 'classifier-probabilities-{}.tsv'
                                 ''.format(lang)), sep='\t')
