#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-confmats-from-classifier-output.py'
===============================================================================

This script converts output from a series of binary phonological feature
classifiers into segment probabilities formatted as a confusion matrix.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)


from __future__ import division, print_function
import json
import numpy as np
import os.path as op
from os import mkdir
from pandas import Series, DataFrame, Panel, read_csv, concat
from numpy import logical_not as negate

# flags
add_missing_feats = False
process_individual_subjs = True

# file i/o
paramdir = 'params'
outdir = 'processed-data'
if not op.isdir(outdir):
    mkdir(outdir)


def compute_classifier_scores_from_probs(featprob):
    # use probabilities as scores
    pos_class = featprob.columns[[x.startswith('+') for x in featprob.columns]]
    featscore = featprob[pos_class]
    featscore.index = [ipa[x] for x in featscore.index]     # convert to IPA
    featscore.columns = [x[1:] for x in featscore.columns]  # remove +/- sign
    assert len(featscore.columns) == len(np.unique(featscore.columns))
    # ground truth
    feattruth = DataFrame([feat_ref.loc[seg] for seg in featscore.index])
    return featscore, feattruth


def find_EER_thresholds(featscore, feattruth):
    steps = np.linspace(0, 1, 11)
    thresh_mat = np.tile(steps, (featscore.shape[1], 1))
    false_pos = Panel(np.zeros((steps.size,) + featscore.shape), dtype=bool,
                      major_axis=featscore.index, minor_axis=featscore.columns)
    false_neg = Panel(np.zeros((steps.size,) + featscore.shape), dtype=bool,
                      major_axis=featscore.index, minor_axis=featscore.columns)
    converged = False
    iteration = 0
    print('Finding EER thresholds ({}): iteration'.format(lang), end=' ')
    while not converged:
        iteration += 1
        print(str(iteration), end=' ')
        for thresh, feat in zip(thresh_mat, featscore):
            for ix, thr in enumerate(thresh):
                false_pos.loc[ix, :, feat] = (negate(feattruth[feat]) &
                                              (featscore[feat] >= thr))
                false_neg.loc[ix, :, feat] = (feattruth[feat].astype(bool) &
                                              (featscore[feat] < thr))
        ratios = (false_pos.sum() / false_neg.sum()).T.iloc[::-1]
        lowbound_ix = ratios.apply(np.searchsorted, raw=True, v=1)
        lowvalue = np.array([ratios.loc[b, i] for b, i in
                             zip(ratios.index[lowbound_ix],
                                 lowbound_ix.index)])
        converged = np.allclose(lowvalue[negate(np.isnan(lowvalue))], 1)
        thresholds = thresh_mat[range(thresh_mat.shape[0]),
                                ratios.index[lowbound_ix]]
        steps = steps / 10.
        thresh_mat = np.atleast_2d(thresholds).T + steps
    nans = np.isnan(lowvalue)
    print(' NaNs: {0} ({1})'.format(nans.sum(),
                                    ' '.join(featscore.columns[nans])))
    return thresholds


def add_missing_feat_EERs(featscore, equal_error_rate):
    # add EER of 0.5 for all missing features
    missing_feats = feat_tab.columns[np.in1d(feat_tab.columns,
                                             featscore.columns, invert=True)]
    equal_error_rate = np.r_[equal_error_rate,
                             0.5 * np.ones(missing_feats.size)]
    equal_error_rate = Series(equal_error_rate,
                              index=np.r_[featscore.columns, missing_feats])
    return missing_feats, equal_error_rate


def add_missing_feats_to_ref(feat_ref, missing_feats):
    z = np.zeros((feat_ref.shape[0], missing_feats.size), dtype=int)
    missing_df = DataFrame(z, columns=missing_feats, index=feat_ref.index)
    return concat((feat_ref, missing_df), axis=1)


def make_weights_matrix(equal_error_rate):
    # make binary mask of feature matches: shape=(foreign_cons, eng_cons, feat)
    # 1. add 1 to remap binary feature values as 1 (absence) or 2 (presence)
    # 2. multiply each English phone's feature value with each foreign one
    # 3. if product is 1 or 4 then feats. match, if product is 2 then mismatch
    mask = np.einsum('ik,jk->ijk', 1 + feat_ref_foreign,
                     1 + feat_ref_eng_expanded) != 2
    # convert equal_error_rate to (1 - equal_error_rate) when features match
    # (this yields probability of "correct" classification for each feature
    # when considering feature values of each English phone as a prior)
    feat_prob_mat = np.abs(mask.astype(int) - equal_error_rate.values)
    # aggregate feature classif. probs. into phone classif. probs.
    weights_matrix = np.exp(-np.sum(-np.log(feat_prob_mat), axis=-1))
    return weights_matrix


# load ancillary data
subj_dict = np.load(op.join(paramdir, 'subjects.npz'))
with open(op.join(paramdir, 'ascii-to-ipa.json'), 'r') as ipafile:
    ipa = json.load(ipafile)
feat_path_eng = op.join(paramdir, 'reference-feature-table-english.tsv')
feat_ref_eng = read_csv(feat_path_eng, sep='\t', index_col=0, encoding='utf-8')
feat_ref_all = read_csv(op.join(paramdir, 'reference-feature-table-all.tsv'),
                        sep='\t', index_col=0, encoding='utf-8')
feat_ref = read_csv(op.join(paramdir, 'reference-feature-table-cons.tsv'),
                    sep='\t', index_col=0, encoding='utf-8')
# this sort order is for the classifier features only
sort_by = ['consonantal', 'labial', 'coronal', 'dorsal', 'continuant',
           'sonorant', 'periodicGlottalSource', 'distributed', 'strident']
feat_ref_cons = feat_ref.sort_values(by=sort_by, ascending=False)
feat_ref_all = feat_ref_all.sort_values(by=sort_by, ascending=False)
feat_ref_eng = feat_ref_eng.sort_values(by=sort_by, ascending=False)
feat_ref = feat_ref_all
# convert to binary if needed
if isinstance(feat_ref.iloc[0, 0], (str, unicode)):
    feat_ref = feat_ref.apply(lambda x: x == '+').astype(int)

# load each language's phone sets
phonesets = np.load(op.join(paramdir, 'phonesets.npz'))
all_phones = np.load(op.join(paramdir, 'allphones.npy')).tolist()
langs = np.load(op.join(paramdir, 'langs.npy'))

# read in PHOIBLE feature data
feat_tab = read_csv(op.join(paramdir, 'phoible-segments-features.tsv'),
                    sep='\t', encoding='utf-8', index_col=0)
feat_tab = feat_tab.loc[all_phones]
assert feat_tab.shape[0] == len(all_phones)

# eliminate redundant features
vacuous = feat_tab.apply(lambda x: len(np.unique(x)) == 1).values
privative = feat_tab.apply(lambda x: len(np.unique(x)) == 2 and
                           '0' in np.unique(x)).values
feat_tab = feat_tab.iloc[:, negate(vacuous | privative)]

# add 'syllabic' to beginning of sort order to group vowels together
sort_by = ['syllabic'] + sort_by
feat_tab = feat_tab.sort_values(by=sort_by, ascending=False)

# init some containers
featscores = dict()
weights_mats = dict()
equal_error_rates = DataFrame()

# iterate over languages
for lang in langs:
    # load classification results
    fname = op.join(outdir, 'classifier-probabilities-{}.tsv'.format(lang))
    featprob = read_csv(fname, sep='\t', index_col=0)
    # use probabilities as scores
    featscore, feattruth = compute_classifier_scores_from_probs(featprob)
    featscores[lang] = featscore
    # find threshold for each feat to equalize error rate
    thresholds = find_EER_thresholds(featscore, feattruth)
    # check thresholds are actually yielding equal error rates
    predictions = (featscore >= thresholds).astype(int)
    false_pos = (predictions.values & negate(feattruth.values)).sum(axis=0)
    false_neg = (negate(predictions.values) & feattruth.values).sum(axis=0)
    assert np.array_equal(false_pos, false_neg)
    # calculate equal error rates for each feature
    equal_error_rate = false_pos / predictions.shape[0]
    if add_missing_feats:
        (missing_feats,
         equal_error_rate) = add_missing_feat_EERs(featscore, equal_error_rate)
        feat_ref_expanded = add_missing_feats_to_ref(feat_ref, missing_feats)
    else:
        equal_error_rate = Series(equal_error_rate, index=featscore.columns)
        feat_ref_expanded = feat_ref
    # propogate add'l features to english feature table
    feat_ref_eng_expanded = feat_ref_expanded.loc[phonesets['eng']]
    # create confusion matrix
    feat_ref_foreign = feat_ref_expanded.loc[phonesets[lang]]
    # aggregate feature classif. probs. into phone classif. probs.
    weights_matrix = make_weights_matrix(equal_error_rate)
    # save to global variables
    equal_error_rates[lang] = equal_error_rate
    weights_mats[lang] = DataFrame(weights_matrix, index=phonesets[lang],
                                   columns=phonesets['eng'])
# save results
equal_error_rates.to_csv(op.join(outdir, 'equal-error-rates.tsv'), sep='\t')
for lang, wmat in weights_mats.items():
    wmat.to_csv(op.join(outdir, 'eeg-confusion-matrix-{}.tsv'.format(lang)),
                sep='\t', encoding='utf-8')

# process individual subjects
if process_individual_subjs:
    for subj_id in subj_dict.keys():
        subj_outdir = op.join(outdir, subj_id)
        # init some containers
        featscores = dict()
        weights_mats = dict()
        equal_error_rates = DataFrame()
        print('processing subject {}'.format(subj_id))
        for lang in langs:
            # load classification results
            fname = 'classifier-probabilities-{}-{}.tsv'.format(lang, subj_id)
            if op.exists(op.join(subj_outdir, fname)):
                featprob = read_csv(op.join(subj_outdir, fname), sep='\t',
                                    index_col=0)
                # use probabilities as scores
                featscore, feattruth = \
                    compute_classifier_scores_from_probs(featprob)
                featscores[lang] = featscore
                # find threshold for each feat to equalize error rate
                thresholds = find_EER_thresholds(featscore, feattruth)
                # check thresholds are actually yielding equal error rates
                predictions = (featscore >= thresholds).astype(int)
                false_pos = (predictions.values &
                             negate(feattruth.values)).sum(axis=0)
                false_neg = (negate(predictions.values) &
                             feattruth.values).sum(axis=0)
                assert np.array_equal(false_pos, false_neg)
                # calculate equal error rates for each feature
                equal_error_rate = false_pos / predictions.shape[0]
                if add_missing_feats:
                    missing_feats, equal_error_rate = \
                        add_missing_feat_EERs(featscore, equal_error_rate)
                    feat_ref_expanded = \
                        add_missing_feats_to_ref(feat_ref, missing_feats)
                else:
                    equal_error_rate = Series(equal_error_rate,
                                              index=featscore.columns)
                    feat_ref_expanded = feat_ref
                # propogate add'l features to english feature table
                feat_ref_eng_expanded = feat_ref_expanded.loc[phonesets['eng']]
                # create confusion matrix
                feat_ref_foreign = feat_ref_expanded.loc[phonesets[lang]]
                # aggregate feature classif. probs. into phone classif. probs.
                weights_matrix = make_weights_matrix(equal_error_rate)
                # save to global variables
                equal_error_rates[lang] = equal_error_rate
                weights_mats[lang] = DataFrame(weights_matrix,
                                               index=phonesets[lang],
                                               columns=phonesets['eng'])
        # save results
        eer_fname = 'equal-error-rates-{}.tsv'.format(subj_id)
        equal_error_rates.to_csv(op.join(subj_outdir, eer_fname), sep='\t')
        for lang, wmat in weights_mats.items():
            wmat_fname = 'eeg-confusion-matrix-{}-{}.tsv'.format(lang, subj_id)
            wmat.to_csv(op.join(subj_outdir, wmat_fname), sep='\t',
                        encoding='utf-8')
