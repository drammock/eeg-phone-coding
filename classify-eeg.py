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
import numpy as np
import pandas as pd
import os.path as op
from os import mkdir
from numpy.lib.recfunctions import merge_arrays
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from aux_functions import time_domain_pca

np.set_printoptions(precision=6, linewidth=160)
pd.set_option('display.width', 160)
rand = np.random.RandomState(seed=0)

# flags
use_dss = True
n_dss_channels_to_use = 8
classify_individ_subjs = True
pca_time_domain = True
# chosen by visual inspection of plot_erp_dss.py (`None` uses all timepts):
truncate_pca_to_timepts = 20

# file i/o
paramdir = 'params'
outdir = 'processed-data'
infile = 'merged-dss-data.npz' if use_dss else 'merged-eeg-data.npz'


# functions
def train_classifier(data, labels, name):
    # train classifier
    print('  {}'.format(name), end=': ')
    classifier = LDA(solver='svd')
    trained_model = classifier.fit(X=data, y=labels)
    # handle class names and dtypes for structured array
    dtype_names = ['{}{}'.format(['-', '+'][val], name)
                   for val in trained_model.classes_]
    dtype_formats = [float] * len(trained_model.classes_)
    model_dtype_dict = dict(names=dtype_names, formats=dtype_formats)
    return trained_model, model_dtype_dict


def test_classifier(data, dtypes):
    prob = trained_model.predict_proba(data)
    return np.array([tuple(x) for x in prob], dtype=dtypes)


# load feature table
feat_ref = pd.read_csv(op.join(paramdir, 'reference-feature-table-cons.tsv'),
                       sep='\t', index_col=0, encoding='utf-8')

# load merged EEG data & other params
invars = np.load(op.join(outdir, infile))
epochs = invars['epochs']
events = invars['events']
langs = invars['langs']
foreign_langs = invars['foreign_langs']
feats = invars['feats']
cons = invars['cons']
subj_vec = invars['subjs']
test_mask = invars['test_mask']
train_mask = invars['train_mask']
validation_mask = invars['validation_mask']
subj_dict = np.load(op.join(paramdir, 'subjects.npz'))

# reduce dimensionality of time domain with PCA
if pca_time_domain:
    print('running PCA on time domain')
    epochs = time_domain_pca(epochs)
    if truncate_pca_to_timepts is not None:
        epochs = epochs[:, :, :truncate_pca_to_timepts]
epochs_cat = epochs[:, :n_dss_channels_to_use, :].reshape(epochs.shape[0], -1)

# more containers
classifier_dict = dict()
validation = list()
language_dict = {lang: list() for lang in foreign_langs}

# across-subject classification
print('training classifiers across all subjects:')
for featname in feat_ref.columns:
    train_data = epochs_cat[train_mask]
    train_labels = feats[featname][train_mask]
    trained_model, model_dtypes = train_classifier(train_data, train_labels,
                                                   featname)
    # validate on new English talkers
    eng_prob = test_classifier(epochs_cat[validation_mask], model_dtypes)
    validation.append(eng_prob)
    # test on foreign sounds
    for lang in foreign_langs:
        print(lang, end=' ')
        lang_mask = langs == lang
        prob = test_classifier(data=epochs_cat[(test_mask & lang_mask)],
                               dtypes=model_dtypes)
        language_dict[lang].append(prob)
    print()
    # save classifier objects
    classifier_dict[featname] = trained_model
np.savez(op.join(outdir, 'classifiers.npz'), **classifier_dict)
# convert predictions to DataFrames and save
validation_df = pd.DataFrame(merge_arrays(validation, flatten=True),
                             index=cons[validation_mask])
validation_df.to_csv(op.join(outdir, 'classifier-probabilities-eng.tsv'),
                     sep='\t')
for lang in foreign_langs:
    lang_mask = langs == lang
    test_probs = pd.DataFrame(merge_arrays(language_dict[lang], flatten=True),
                              index=cons[(test_mask & lang_mask)])
    test_probs.to_csv(op.join(outdir, 'classifier-probabilities-{}.tsv'
                              ''.format(lang)), sep='\t')

# classify individual subjects
if classify_individ_subjs:
    subj_classifier_dict = dict()
    for subj_id, subj_num in subj_dict.items():
        subj_outdir = op.join(outdir, subj_id)
        if not op.isdir(subj_outdir):
            mkdir(subj_outdir)
        print('training on subject {}:'.format(subj_id))
        classifier_dict = dict()
        validation = list()
        language_dict = {lang: list() for lang in foreign_langs}
        # subset the data
        subj_mask = subj_vec == subj_num
        for featname in feat_ref.columns:
            train_data = epochs_cat[(subj_mask & train_mask)]
            train_labels = feats[featname][(subj_mask & train_mask)]
            trained_model, model_dtypes = train_classifier(train_data,
                                                           train_labels,
                                                           featname)
            # validate on new English talkers
            val_data = epochs_cat[(subj_mask & validation_mask)]
            eng_prob = test_classifier(val_data, model_dtypes)
            validation.append(eng_prob)
            # test on foreign sounds
            for lang in foreign_langs:
                print(lang, end=' ')
                lang_mask = langs == lang
                # not all subjs heard all foreign langs
                if np.any(subj_mask & test_mask & lang_mask):
                    test_data = epochs_cat[(subj_mask & test_mask & lang_mask)]
                    prob = test_classifier(test_data, model_dtypes)
                    language_dict[lang].append(prob)
            print()
            # save classifier objects
            classifier_dict[featname] = trained_model
        np.savez(op.join(subj_outdir, 'classifiers-{}.npz'.format(subj_id)),
                 **classifier_dict)
        # convert predictions to DataFrames and save
        validation_df = pd.DataFrame(merge_arrays(validation, flatten=True),
                                     index=cons[(subj_mask & validation_mask)])
        val_fname = 'classifier-probabilities-eng-{}.tsv'.format(subj_id)
        validation_df.to_csv(op.join(subj_outdir, val_fname), sep='\t')
        for lang in foreign_langs:
            lang_mask = langs == lang
            # not all subjs heard all foreign langs
            if np.any(subj_mask & test_mask & lang_mask):
                index = cons[(subj_mask & test_mask & lang_mask)]
                test_probs = pd.DataFrame(merge_arrays(language_dict[lang],
                                                       flatten=True),
                                          index=index)
                test_fname = ('classifier-probabilities-{}-{}.tsv'
                              ''.format(lang, subj_id))
                test_probs.to_csv(op.join(subj_outdir, test_fname), sep='\t')
