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
from numpy.core.records import fromarrays
# from os import mkdir
from os import path as op
from pandas import read_csv
from mne_sandbox.preprocessing._dss import _pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.svm import LinearSVC

rand = np.random.RandomState(seed=0)

# flags
align = 'v'  # whether epochs are aligned to consonant (c) or vowel (v) onset
have_dss = True
use_dss = True
n_dss_channels = 1
do_individ_subjs = False

# TODO: try averaging various numbers of tokens (2, 4, 5) prior to training
# classifier

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
df['cons'] = df['syll'].apply(lambda x: x[:-2].replace('-', '_')
                              if x.split('-')[-1] in ('0', '1', '2')
                              else x.replace('-', '_'))
df['cons']
cons_dict = dict()
lang_dict = dict()
for key, cons, lang in zip(df['wav_idx'], df['cons'], df['lang']):
    if key in cons_dict.keys() and cons_dict[key] != cons:
        raise RuntimeError
    if key in lang_dict.keys() and lang_dict[key] != lang:
        raise RuntimeError
    cons_dict[key] = cons.replace('-', '_')
    lang_dict[key] = lang

# convert ASCII to IPA for feature lookup
ipa = dict(t_dental=u't̪', gamma=u'ɣ', t_dental_aspirated=u't̪ʰ',
           g_breathy=u'ɡʱ', t_retroflex_aspirated=u'ʈʰ',
           tesh_aspirated=u't̠ʃʰ', r_cap_inverted=u'ʁ', d_implosive=u'ɗ',
           tc_curl=u'tɕ', d_retroflex_breathy=u'ɖʱ', j_bar=u'ɟ',
           engma=u'ŋ', fronted_x=u'x̟', r_turned=u'ɹ', theta=u'θ',
           d_dental_breathy=u'd̪ʱ', ts_aspirated=u'tsʰ',
           b_prenasalized=u'ᵐb', ezh=u'ʒ', b_implosive=u'ɓ',
           s_retroflex=u'ʂ', b_breathy=u'bʱ', flap_retroflex_breathy=u'ɽʱ',
           z_prenasalized=u'ⁿz', g=u'ɡ', nu=u'ʋ', r_dental=u'r̪',
           p_aspirated=u'pʰ', g_prenasalized=u'ᵑɡ', c_cedilla=u'ç',
           dezh_breathy=u'd̠ʒʱ', ts_retroflex_aspirated=u'ʈʂʰ',
           t_aspirated=u'tʰ', l_dental=u'l̪', c_aspirated=u'cʰ', esh=u'ʃ',
           ts_retroflex=u'ʈʂ', k_aspirated=u'kʰ', eth=u'ð',
           v_prenasalized=u'ᶬv', n_palatal=u'ɲ', tesh=u't̠ʃ',
           n_dental=u'n̪', dezh=u'd̠ʒ', c_curl=u'ɕ',
           j_bar_prenasalized=u'ᶮɟ', d_retroflex=u'ɖ', uvular_trill=u'ʀ',
           tc_curl_aspirated=u'tɕʰ', d_dental=u'd̪', chi=u'χ',
           d_prenasalized=u'ⁿd', flap_retroflex=u'ɽ', t_retroflex=u'ʈ')
# fix some segment mismatches between feature table and dict
corrections = dict(fronted_x=u'x', v_prenasalized=u'ɱv', b_prenasalized=u'mb',
                   d_prenasalized=u'nd', j_bar_prenasalized=u'ɲɟ',
                   z_prenasalized=u'nz', g_prenasalized=u'ŋɡ', c_cedilla=u'ç')
ipa.update(corrections)
# add in consonants that are within ASCII
all_cons = np.unique(df['cons'].apply(str.replace, args=('-', '_'))
                     ).astype(unicode)
ascii_cons = all_cons[np.in1d(all_cons, ipa.keys(), invert=True)]
ipa.update({k: k for k in ascii_cons})
# reduce feature table to only the segments we need
feat_tab = read_csv('phoible-segments-features.tsv', sep='\t',
                    encoding='utf-8')
feat_tab = feat_tab.iloc[np.in1d(feat_tab['segment'], ipa.values())]
assert feat_tab.shape[0] == len(ipa)
# remove any features that are fully redundant within the training set
eng_cons = np.unique(df['cons'].loc[df['lang'] == 'eng'
                                    ].apply(str.replace, args=('-', '_')))
eng_cons = [ipa[x] for x in eng_cons.astype(unicode)]
eng_feat_tab = feat_tab.iloc[np.in1d(feat_tab['segment'].values, eng_cons)]
eng_nonredundant = eng_feat_tab.apply(lambda x: len(np.unique(x)) != 1,
                                      raw=True).values
feat_tab = feat_tab.iloc[:, eng_nonredundant]
feat_tab = feat_tab.set_index('segment')
# determine dtype for later use in record arrays
dty = 'a{}'.format(max([len(x) for x in np.unique(feat_tab).astype(str)]))
recarray_dtypes = [(str(f), dty) for f in feat_tab.columns]
del corrections, eng_cons, eng_feat_tab, eng_nonredundant, ascii_cons

# init some global containers
epochs = list()
events = list()
subjs = list()
cons = list()
feats = list()
subj_feat_classifiers = dict()

# read in cleaned EEG data
for subj_code, subj in subjects.items():
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
        this_feat = [feat_tab[feat].loc[ipa[con]] for feat in feat_tab.columns]
        this_feats.append(this_feat)
    this_feats = np.array(this_feats, dtype=str)
    this_feats = fromarrays(this_feats.T, dtype=recarray_dtypes)
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
    if do_individ_subjs:
        # concatenate DSS components
        data_cat = this_data[:, :n_dss_channels, :].reshape(this_data.shape[0],
                                                            -1)
        # do LDA
        feat_classifiers = dict()
        for fname in feat_tab.columns:
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

# convert global containers to arrays
epochs = np.array(epochs)
events = np.array(events)
subjs = np.squeeze(subjs)
cons = np.array(cons)
feats = np.array(feats, dtype=recarray_dtypes)
epochs_cat = epochs[:, :n_dss_channels, :].reshape(epochs.shape[0], -1)
train_mask = np.array([training_dict[e] for e in events])
valid_mask = np.array([validate_dict[e] for e in events])
test_mask = np.array([testing_dict[e] for e in events])

# more containers
feat_classifiers = dict()
test_performance = dict()
validation_performance = dict()

# do across-subject LDA
for fname in feat_tab.columns:
    lda_classif = LDA(solver='svd')
    lda_trained = lda_classif.fit(X=epochs_cat[train_mask],
                                  y=feats[fname][train_mask])
    feat_classifiers[fname] = lda_trained
    # validate
    eng_validate = lda_trained.predict(epochs_cat[valid_mask])
    n_corr = np.sum(feats[fname][valid_mask] == eng_validate)
    validation_performance[fname] = n_corr / valid_mask.sum()
    # pct_corr = np.round(100 * validation_performance[fname]).astype(int)
    # test
    foreign_prob = lda_trained.predict_proba(epochs_cat[test_mask])
    foreign_test = lda_trained.predict(epochs_cat[test_mask])
    foreign_corr = feats[fname][test_mask] == foreign_test
    probarray_dtypes = [(name, float) for name in lda_trained.classes_]
    foreign_prob = fromarrays(foreign_prob.T, dtype=probarray_dtypes)
    n_corr = np.sum(foreign_corr)
    test_performance[fname] = n_corr / test_mask.sum()
    # pct_corr = np.round(100 * test_performance[fname]).astype(int)
print('\n'.join(['{:0.2} ({})'.format(v, k)
                 for k, v in test_performance.items()]))

"""
import matplotlib.pyplot as plt
for ev in np.unique(this_events)[:1]:
    ixs = this_events == ev
    plt.plot(data[ixs, 0, :].T, linewidth=0.5)
    plt.plot(data[ixs, 0, :].mean(0), linewidth=1.5)
"""
