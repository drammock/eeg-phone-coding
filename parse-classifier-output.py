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
from pandas import DataFrame, Panel, read_csv
import matplotlib.pyplot as plt

# flags
plot = True

# file i/o
outdir = 'processed-data'
if not op.isdir(outdir):
    mkdir(outdir)

# load ancillary data
with open(op.join(outdir, 'ascii-to-ipa.json'), 'r') as ipafile:
    ipa = json.load(ipafile)
feat_ref = read_csv(op.join(outdir, 'reference-feature-table.tsv'), sep='\t',
                    index_col=0, encoding='utf-8')
feat_ref_eng = read_csv(op.join(outdir, 'english-reference-feature-table.tsv'),
                        sep='\t', index_col=0, encoding='utf-8')
eng_segments = feat_ref_eng.index.values.astype(unicode)
# convert to binary if needed
if isinstance(feat_ref.iloc[0, 0], (str, unicode)):
    feat_ref = feat_ref.apply(lambda x: x == '+').astype(int)
# init some containers
featscores = dict()
feattruths = dict()
confmats = dict()
equal_error_rates = dict()
# load each language's classification results
foreign_files = glob(op.join(outdir, 'classifier-probabilities-*.tsv'))
foreign_langs = [op.split(x)[1][-7:-4] for x in foreign_files]
for fname, lang in zip(foreign_files, foreign_langs):
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
        converged = np.allclose(lowvalue, 1)
        thresholds = thresh_mat[range(thresh_mat.shape[0]),
                                ratios.index[lowbound_ix]]
        steps = steps / 10
        thresh_mat = np.atleast_2d(thresholds).T + steps
    print()
    del (iteration, thresh, feat, ix, thr, false_pos, false_neg, ratios,
         lowbound_ix, lowvalue, converged, steps, thresh_mat)
    # calculate equal error rates for each feature
    predictions = (featscore >= thresholds).astype(int)
    false_pos = (predictions.values & ~feattruth.values).sum(axis=0)
    false_neg = (~predictions.values & feattruth.values).sum(axis=0)
    assert np.array_equal(false_pos, false_neg)
    equal_error_rate = false_pos / predictions.shape[0]
    equal_error_rates[lang] = equal_error_rate
    # create confusion matrix
    foreign_segments = predictions.index.unique().tolist()
    feat_ref_foreign = feat_ref.loc[foreign_segments]
    # binary mask of feature matches, shape: (foreign_cons, eng_cons, feat)
    mask = np.einsum('ik,jk->ijk', 1 + feat_ref_foreign, 1 + feat_ref_eng) != 2
    eer_mat = np.abs(mask.astype(int) - equal_error_rate)
    confusion_matrix = eer_mat.prod(axis=-1)
    confmats[lang] = DataFrame(confusion_matrix, index=foreign_segments,
                               columns=eng_segments)

if plot:
    plt.rc('font', serif='Charis SIL', family='serif')
    plt.rc('axes.spines', top=False, right=False)
    plt.rc('xtick.major', size=2)
    plt.rc('ytick.major', size=2)
    plt.rc('ytick', right=False)
    plt.rc('xtick', top=False)
    fig, axs = plt.subplots(2, 2, squeeze=False)
    for ix, lang in enumerate(foreign_langs):
        confmat = confmats[lang]
        ax = axs[ix % axs.shape[0], ix // axs.shape[0]]
        _ = ax.imshow(confmat)
        _ = ax.yaxis.set_ticks(range(confmat.shape[0]))
        _ = ax.xaxis.set_ticks(range(confmat.shape[1]))
        _ = ax.yaxis.set_ticklabels(confmat.index)
        _ = ax.xaxis.set_ticklabels(confmat.columns)
        _ = ax.set_title(lang)
