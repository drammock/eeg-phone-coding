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
import yaml
import numpy as np
import pandas as pd
import os.path as op
from os import mkdir
from time import time
from numpy.lib.recfunctions import merge_arrays
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from aux_functions import time_domain_pca, print_elapsed

np.set_printoptions(precision=6, linewidth=160)
pd.set_option('display.width', 160)
rand = np.random.RandomState(seed=0)

# file i/o
paramdir = 'params'
outdir = 'processed-data'
analysis_params = 'current-analysis-settings.yaml'

# load analysis params
with open(op.join(paramdir, analysis_params), 'r') as paramfile:
    params = yaml.load(paramfile)
clf_type = params['clf_type']
use_dss = params['dss']['use']
dss_n_channels = params['dss']['use_n_channels']
pca_time_domain = params['pca']['time_domain']
pca_truncate = params['pca']['truncate_to_n_timepts']
process_individual_subjs = params['process_individual_subjs']
fname_suffix = '-dss-{}'.format(dss_n_channels) if use_dss else ''
infile = 'merged-dss-data.npz' if use_dss else 'merged-eeg-data.npz'


# functions
def new_classifier(clf_type, **kwargs):
    if clf_type == 'lda':
        clf = LDA(solver='svd', **kwargs)
    elif clf_type == 'svm':
        clf = SVC(**kwargs)
    else:
        raise ValueError('unrecognized value for classifier type (clf_type)')
    return clf


def train_classifier(classifier, data, labels, msg):
    # train classifier
    print('  {}'.format(msg), end=': ')
    _st = time()
    classifier.fit(X=data, y=labels)
    # handle class names and dtypes for structured array
    dtype_names = ['{}{}'.format(['-', '+'][val], msg)
                   for val in np.unique(labels)]
    dtype_formats = [float] * np.unique(labels).size
    model_dtype_dict = dict(names=dtype_names, formats=dtype_formats)
    print_elapsed(_st)
    return classifier, model_dtype_dict


def test_classifier(classifier, data, dtypes):
    prob = classifier.predict_proba(data)
    return np.array([tuple(x) for x in prob], dtype=dtypes)


def score_EER(estimator, X, y):
    steps = np.linspace(0, 1, 11)
    converged = False
    iteration = 0
    threshold = -1
    while not converged:
        old_threshold = threshold
        iteration += 1
        probs = estimator.predict_proba(X)
        preds = np.array([probs[:, 1] >= thresh for thresh in steps])
        trues = np.tile(y.astype(bool), (steps.size, 1))
        falses = np.logical_not(trues)
        # false pos / false neg rates
        fpr = (falses & preds).sum(axis=1) / falses.sum(axis=1)
        fnr = (trues & np.logical_not(preds)).sum(axis=1) / trues.sum(axis=1)
        ratios = fpr / fnr
        if np.isinf(ratios[0]) or ratios[0] > ratios[1]:
            ratios = ratios[::-1]
            steps = steps[::-1]
        ix = np.searchsorted(ratios, v=1)
        threshold = steps[ix]
        converged = (np.isclose(ratios[ix], 1.) or
                     np.isclose(threshold, old_threshold))
        steps = np.linspace(steps[ix - 1], steps[ix], 11)
    eer = fpr[ix]
    return 1 - eer


print('loading data', end=': ')
_st = time()
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
print_elapsed(_st)

# reduce dimensionality of time domain with PCA
if pca_time_domain:
    print('running PCA on time domain', end=': ')
    _st = time()
    epochs = time_domain_pca(epochs)
    if pca_truncate is not None:
        epochs = epochs[:, :, :pca_truncate]
    print_elapsed(_st)
epochs_cat = epochs[:, :dss_n_channels, :].reshape(epochs.shape[0], -1)

# more containers
classifier_dict = dict()
validation = list()
language_dict = {lang: list() for lang in foreign_langs}

# across-subject classification
print('starting cross-validation')
print('training classifiers across all subjects')
for featname in feat_ref.columns:
    train_data = epochs_cat[train_mask]
    train_labels = feats[featname][train_mask]
    # grid search for hyperparameters
    param_grid = [dict(C=[10, 30, 100, 300, 1000, 3000],
                       gamma=[1e-2, 1e-3, 1e-4])]
    clf_kwargs = dict(probability=True, kernel='rbf',
                      decision_function_shape='ovr', random_state=rand)
    clf = GridSearchCV(new_classifier(clf_type, **clf_kwargs),
                       param_grid=param_grid,
                       scoring=score_EER, n_jobs=6, pre_dispatch=9,
                       cv=5, refit=True, verbose=3)
    trained_model, model_dtypes = train_classifier(classifier=clf,
                                                   data=train_data,
                                                   labels=train_labels,
                                                   msg=featname)
    # test on new English talkers
    print('    testing: eng', end=' ')
    eng_prob = test_classifier(clf, epochs_cat[validation_mask], model_dtypes)
    validation.append(eng_prob)
    # test on foreign sounds
    for lang in foreign_langs:
        print(lang, end=' ')
        lang_mask = langs == lang
        prob = test_classifier(clf, data=epochs_cat[(test_mask & lang_mask)],
                               dtypes=model_dtypes)
        language_dict[lang].append(prob)
    print()
    # save classifier objects
    classifier_dict[featname] = trained_model
np.savez(op.join(outdir, 'classifiers.npz'), **classifier_dict)
# convert predictions to DataFrames and save
validation_df = pd.DataFrame(merge_arrays(validation, flatten=True),
                             index=cons[validation_mask])
validation_fname = 'classifier-probabilities-eng-{}{}.tsv'.format(clf_type,
                                                                  fname_suffix)
validation_df.to_csv(op.join(outdir, validation_fname), sep='\t')
for lang in foreign_langs:
    lang_mask = langs == lang
    test_probs = pd.DataFrame(merge_arrays(language_dict[lang], flatten=True),
                              index=cons[(test_mask & lang_mask)])
    test_fname = 'classifier-probabilities-{}-{}{}.tsv'.format(lang, clf_type,
                                                               fname_suffix)
    test_probs.to_csv(op.join(outdir, test_fname), sep='\t')

# classify individual subjects
if process_individual_subjs:
    subj_classifier_dict = dict()
    for subj_id, subj_num in subj_dict.items():
        subj_outdir = op.join(outdir, subj_id)
        if not op.isdir(subj_outdir):
            mkdir(subj_outdir)
        print('training on subject {}'.format(subj_id))
        classifier_dict = dict()
        validation = list()
        language_dict = {lang: list() for lang in foreign_langs}
        # subset the data
        subj_mask = subj_vec == subj_num
        for featname in feat_ref.columns:
            train_data = epochs_cat[(subj_mask & train_mask)]
            train_labels = feats[featname][(subj_mask & train_mask)]
            # grid search for hyperparameters
            param_grid = [dict(C=[10, 30, 100, 300, 1000, 3000],
                               gamma=[1e-2, 1e-3, 1e-4])]
            clf_kwargs = dict(probability=True, kernel='rbf',
                              decision_function_shape='ovr', random_state=rand)
            clf = GridSearchCV(new_classifier(clf_type, **clf_kwargs),
                               param_grid=param_grid,
                               scoring=score_EER, n_jobs=6, pre_dispatch=9,
                               cv=5, refit=True, verbose=3)
            trained_model, model_dtypes = train_classifier(classifier=clf,
                                                           data=train_data,
                                                           labels=train_labels,
                                                           msg=featname)
            # test on new English talkers
            print('    testing: eng', end=' ')
            val_data = epochs_cat[(subj_mask & validation_mask)]
            eng_prob = test_classifier(clf, val_data, model_dtypes)
            validation.append(eng_prob)
            # test on foreign sounds
            for lang in foreign_langs:
                print(lang, end=' ')
                lang_mask = langs == lang
                # not all subjs heard all foreign langs
                if np.any(subj_mask & test_mask & lang_mask):
                    test_data = epochs_cat[(subj_mask & test_mask & lang_mask)]
                    prob = test_classifier(clf, test_data, model_dtypes)
                    language_dict[lang].append(prob)
            print()
            # save classifier objects
            classifier_dict[featname] = trained_model
        clfdict_fname = ('classifiers-{}{}-{}.tsv'
                         ''.format(clf_type, fname_suffix, subj_id))
        np.savez(op.join(subj_outdir, clfdict_fname), **classifier_dict)
        # convert predictions to DataFrames and save
        validation_df = pd.DataFrame(merge_arrays(validation, flatten=True),
                                     index=cons[(subj_mask & validation_mask)])
        val_fname = ('classifier-probabilities-eng-{}{}-{}.tsv'
                     ''.format(clf_type, fname_suffix, subj_id))
        validation_df.to_csv(op.join(subj_outdir, val_fname), sep='\t')
        for lang in foreign_langs:
            lang_mask = langs == lang
            # not all subjs heard all foreign langs
            if np.any(subj_mask & test_mask & lang_mask):
                index = cons[(subj_mask & test_mask & lang_mask)]
                test_probs = pd.DataFrame(merge_arrays(language_dict[lang],
                                                       flatten=True),
                                          index=index)
                test_fname = ('classifier-probabilities-{}-{}{}-{}.tsv'
                              ''.format(lang, clf_type, fname_suffix, subj_id))
                test_probs.to_csv(op.join(subj_outdir, test_fname), sep='\t')
