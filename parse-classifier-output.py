#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'parse-classifier-output.py'
===============================================================================

This script converts phonological feature values (output from a series of
binary feature classifiers) into segments bearing those feature values.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)


from __future__ import division, print_function
import json
import numpy as np
from os import mkdir
from os import path as op
from glob import glob
from pandas import Series, DataFrame, Panel, read_csv, concat

# file i/o
paramdir = 'params'
outdir = 'processed-data'
if not op.isdir(outdir):
    mkdir(outdir)

# load ancillary data
with open(op.join(paramdir, 'ascii-to-ipa.json'), 'r') as ipafile:
    ipa = json.load(ipafile)
feat_path_eng = op.join(paramdir, 'reference-feature-table-english.tsv')
feat_ref_eng = read_csv(feat_path_eng, sep='\t', index_col=0, encoding='utf-8')
feat_ref_all = read_csv(op.join(paramdir, 'reference-feature-table-all.tsv'),
                        sep='\t', index_col=0, encoding='utf-8')
feat_ref = read_csv(op.join(paramdir, 'reference-feature-table-cons.tsv'),
                    sep='\t', index_col=0, encoding='utf-8')
# this sort order is for the classifier features only
sort_order = ['consonantal', 'labial', 'coronal', 'dorsal', 'continuant',
              'sonorant', 'periodicGlottalSource', 'distributed', 'strident']
feat_ref_cons = feat_ref.sort_values(by=sort_order, ascending=False)
feat_ref_all = feat_ref_all.sort_values(by=sort_order, ascending=False)
feat_ref_eng = feat_ref_eng.sort_values(by=sort_order, ascending=False)
feat_ref = feat_ref_all
# convert to binary if needed
if isinstance(feat_ref.iloc[0, 0], (str, unicode)):
    feat_ref = feat_ref.apply(lambda x: x == '+').astype(int)
# init some containers
featscores = dict()
feattruths = dict()
weights_mats = dict()
confmats_normed = dict()
equal_error_rates = DataFrame()

# load each language's phone sets
# these phone sets are determined by the output of the probabilistic
# transcription system; may not correspond to the languages' phonemes
fnames = glob(op.join(paramdir, '*-phones.tsv'))
langs = [op.split(x)[1][:3] for x in fnames]
phonesets = dict()
for lang in langs:
    this_phones = read_csv(op.join(paramdir, '{}-phones.tsv'.format(lang)),
                           encoding='utf-8', header=None)
    this_phones = np.squeeze(this_phones.values).astype(unicode).tolist()
    phonesets[lang] = this_phones
all_phones = [phone for inventory in phonesets.values() for phone in inventory]
all_phones = list(set(all_phones))

# read in PHOIBLE feature data
feat_tab = read_csv(op.join(paramdir, 'phoible-segments-features.tsv'),
                    sep='\t', encoding='utf-8', index_col=0)
feat_tab = feat_tab.loc[all_phones]
assert feat_tab.shape[0] == len(all_phones)

# eliminate redundant features
vacuous = feat_tab.apply(lambda x: len(np.unique(x)) == 1).values
privative = feat_tab.apply(lambda x: len(np.unique(x)) == 2 and
                           '0' in np.unique(x)).values
feat_tab = feat_tab.iloc[:, ~(vacuous | privative)]

# add 'syllabic' to beginning of sort order to group vowels together
sort_order = ['syllabic'] + sort_order
feat_tab = feat_tab.sort_values(by=sort_order, ascending=False)

# sort phonesets to fall in same order as feat_tab
for lang in langs:
    indices = [np.where(feat_tab.index.values.astype(unicode) == x)[0][0]
               for x in phonesets[lang]]
    phonesets[lang] = np.array(phonesets[lang])[np.argsort(indices)].tolist()

# load each language's classification results
foreign_files = glob(op.join(outdir, 'classifier-probabilities-*.tsv'))
foreign_langs = [op.split(x)[1][-7:-4] for x in foreign_files]
# iterate over languages
for fname, lang in zip(foreign_files, foreign_langs):
    # load classification results
    featprob = read_csv(fname, sep='\t', index_col=0)
    # use probabilities as scores
    pos_class = featprob.columns[[x.startswith('+') for x in featprob.columns]]
    featscore = featprob[pos_class]
    featscore.index = [ipa[x] for x in featscore.index]     # convert to IPA
    featscore.columns = [x[1:] for x in featscore.columns]  # remove +/- sign
    assert len(featscore.columns) == len(np.unique(featscore.columns))
    # ground truth
    feattruth = DataFrame([feat_ref.loc[seg] for seg in featscore.index])
    featscores[lang] = featscore
    feattruths[lang] = feattruth
    # find threshold for each feat to equalize error rate
    steps = np.linspace(0, 1, 11)
    thresh_mat = np.tile(steps, (featscore.shape[1], 1))
    false_pos = Panel(np.zeros((steps.size,) + featscore.shape), dtype=bool,
                      major_axis=featscore.index, minor_axis=featscore.columns)
    false_neg = Panel(np.zeros((steps.size,) + featscore.shape), dtype=bool,
                      major_axis=featscore.index, minor_axis=featscore.columns)
    converged = False
    iteration = 0
    print('Finding thresholds ({}): iteration'.format(lang), end=' ')
    while not converged:
        iteration += 1
        print(str(iteration), end=' ')
        for thresh, feat in zip(thresh_mat, featscore):
            for ix, thr in enumerate(thresh):
                false_pos.loc[ix, :, feat] = (~feattruth[feat].astype(bool) &
                                              (featscore[feat] >= thr))
                false_neg.loc[ix, :, feat] = (feattruth[feat].astype(bool) &
                                              (featscore[feat] < thr))
        ratios = (false_pos.sum() / false_neg.sum()).T.iloc[::-1]
        lowbound_ix = ratios.apply(np.searchsorted, raw=True, v=1)
        lowvalue = np.array([ratios.loc[b, i] for b, i in
                             zip(ratios.index[lowbound_ix],
                                 lowbound_ix.index)])
        converged = np.allclose(lowvalue[~np.isnan(lowvalue)], 1)
        thresholds = thresh_mat[range(thresh_mat.shape[0]),
                                ratios.index[lowbound_ix]]
        steps = steps / 10
        thresh_mat = np.atleast_2d(thresholds).T + steps
    nans = np.isnan(lowvalue)
    print(' NaNs: {0} ({1})'.format(nans.sum(),
                                    ' '.join(featscore.columns[nans])))
    del (iteration, thresh, feat, ix, thr, false_pos, false_neg, ratios,
         lowbound_ix, lowvalue, converged, steps, thresh_mat)
    # check thresholds are actually yielding equal error rates
    predictions = (featscore >= thresholds).astype(int)
    false_pos = (predictions.values & ~feattruth.values).sum(axis=0)
    false_neg = (~predictions.values & feattruth.values).sum(axis=0)
    assert np.array_equal(false_pos, false_neg)
    # calculate equal error rates for each feature
    equal_error_rate = false_pos / predictions.shape[0]
    # add EER of 0.5 for all missing features
    missing_feats = feat_tab.columns[np.in1d(feat_tab.columns,
                                             featscore.columns, invert=True)]
    equal_error_rate = np.r_[equal_error_rate,
                             0.5 * np.ones(missing_feats.size)]
    # convert to Series & save to global variable
    equal_error_rate = Series(equal_error_rate,
                              index=np.r_[featscore.columns, missing_feats])
    equal_error_rates[lang] = equal_error_rate
    # add missing features to reference feature tables
    z = np.zeros((feat_ref.shape[0], missing_feats.size), dtype=int)
    missing_df = DataFrame(z, columns=missing_feats, index=feat_ref.index)
    feat_ref_expanded = concat((feat_ref, missing_df), axis=1)
    # repeat for english feature table
    """
    z = np.zeros((feat_ref_eng.shape[0], missing_feats.size), dtype=int)
    missing_df = DataFrame(z, columns=missing_feats, index=feat_ref_eng.index)
    feat_ref_eng_expanded = concat((feat_ref_eng, missing_df), axis=1)
    """
    feat_ref_eng_expanded = feat_ref_expanded.loc[phonesets['eng']]
    # create confusion matrix
    """
    phonesets[lang] = feat_ref.index[np.in1d(feat_ref_expanded.index,
                                     featscore.index.unique())]
    """
    feat_ref_foreign = feat_ref_expanded.loc[phonesets[lang]]
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
    # compute product of feature classif. probs. to get phone classif. prob.
    weights_matrix = feat_prob_mat.prod(axis=-1)
    weights_mats[lang] = DataFrame(weights_matrix, index=phonesets[lang],
                                   columns=phonesets['eng'])

# save results
np.save(op.join(paramdir, 'foreign-langs.npy'), foreign_langs)
equal_error_rates.to_csv(op.join(outdir, 'equal-error-rates.tsv'), sep='\t')
for lang, wmat in weights_mats.items():
    wmat.to_csv(op.join(outdir, 'eeg-weights-matrix-{}.tsv'.format(lang)),
                sep='\t', encoding='utf-8')
