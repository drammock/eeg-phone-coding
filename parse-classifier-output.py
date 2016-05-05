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
from pandas import DataFrame, Panel, read_csv

# flags

# file i/o
outdir = 'processed-data'
if not op.isdir(outdir):
    mkdir(outdir)

# load data
with open(op.join(outdir, 'ascii-to-ipa.json'), 'r') as ipafile:
    ipa = json.load(ipafile)
"""
featvals = read_csv(op.join(outdir, 'classifier-output.tsv'), sep='\t',
                    index_col=0)
"""
featprob = read_csv(op.join(outdir, 'classifier-probabilities.tsv'), sep='\t',
                    index_col=0)
feat_ref = read_csv(op.join(outdir, 'reference-feature-table.tsv'), sep='\t',
                    index_col=0, encoding='utf-8')
feat_ref_eng = read_csv(op.join(outdir, 'english-reference-feature-table.tsv'),
                        sep='\t', index_col=0, encoding='utf-8')
eng_segments = feat_ref_eng.index.values.astype(unicode)
del feat_ref_eng

# convert (pairs of) probabilities to scores
pos_class = (featprob.apply(lambda x: x >= 0.5).apply(np.sum) >=
             featprob.shape[0] / 2)
assert pos_class.sum() == pos_class.size / 2
pos_class = pos_class.index[pos_class]
featscore = featprob[pos_class]
featscore.index = [ipa[x] for x in featscore.index]     # convert to IPA
featscore.columns = [x[1:] for x in featscore.columns]  # remove +/- sign
assert len(featscore.columns) == len(np.unique(featscore.columns))
# ground truth
feat_ref_bin = feat_ref.apply(lambda x: x == '+').astype(int)
feattruth = DataFrame([feat_ref_bin.loc[seg] for seg in featscore.index])
# find threshold for each feat to equalize error rate
steps = np.linspace(0, 1, 11)
thresh_mat = np.tile(steps, (featscore.shape[1], 1))
false_pos = Panel(np.zeros((steps.size,) + featscore.shape), dtype=bool,
                  major_axis=featscore.index, minor_axis=featscore.columns)
false_neg = Panel(np.zeros((steps.size,) + featscore.shape), dtype=bool,
                  major_axis=featscore.index, minor_axis=featscore.columns)
converged = False
iteration = 0
print('Finding thresholds: iteration', end=' ')
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
                         zip(ratios.index[lowbound_ix], lowbound_ix.index)])
    converged = np.allclose(lowvalue, 1)
    thresholds = thresh_mat[range(thresh_mat.shape[0]),
                            ratios.index[lowbound_ix]]
    steps = steps / 10
    thresh_mat = np.atleast_2d(thresholds).T + steps
print()
del (iteration, thresh, feat, ix, thr, false_pos, false_neg, ratios,
     lowbound_ix, lowvalue, converged, steps, thresh_mat)
predictions = (featscore >= thresholds).astype(int)
pred_agg = predictions.groupby(predictions.index).aggregate(np.mean)

confusion_matrix = np.zeros((pred_agg.shape[0], len(eng_segments)))
for fix, foreign_seg in enumerate(pred_agg.index):
    predfeats = pred_agg.loc[foreign_seg]
    for eix, eng_seg in enumerate(eng_segments):
        truefeats = feat_ref_bin.loc[eng_seg].astype(bool)
        prob = (np.prod(predfeats[truefeats]) *
                np.prod(1 - predfeats[~truefeats]))
        confusion_matrix[fix, eix] = prob

conf_mat = DataFrame(confusion_matrix, index=pred_agg.index,
                     columns=eng_segments)


"""
import matplotlib.pyplot as plt
aximg = plt.matshow(conf_mat)
ax = aximg.axes
ax.yaxis.set_ticks(range(conf_mat.shape[0]))
ax.xaxis.set_ticks(range(conf_mat.shape[1]))
ax.yaxis.set_ticklabels(conf_mat.index)
ax.xaxis.set_ticklabels(conf_mat.columns)
"""

"""
# generate IPA classifications from features
classification_probs = np.zeros((featvals.shape[0], eng_segments.size),
                                dtype=float)
for rix, row in enumerate(featvals.itertuples()):
    predicted_feats = np.array(row[1:])
    feat_dist = np.zeros(len(eng_segments), dtype=int)
    for ix, ref in enumerate(feat_ref.itertuples()):
        this_ref = np.array(ref[1:], dtype=str)
        feat_dist[ix] = np.sum(predicted_feats != this_ref)
    assert np.sum(feat_dist == 0) < 2
    short_dist_ix = np.where(feat_dist == feat_dist.min())[0]
    # predicted_segs = ' '.join(eng_segments[short_dist_ix])
    classification_probs[rix, short_dist_ix] = 1. / short_dist_ix.size

# convert to DataFrame
inputs = [ipa[segment] for segment in featvals.index]
df = DataFrame(classification_probs, index=inputs, columns=eng_segments)
# aggregate
unique_inputs = sorted(np.unique(inputs).tolist())
groups = df.groupby(df.index)
agg = groups.aggregate(lambda x: 1. - np.prod(1. - x))
"""

