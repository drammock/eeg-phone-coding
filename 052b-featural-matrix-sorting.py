#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'featural-matrix-sorting.py'
===============================================================================

This script uses classifier EERs to sort the rows and columns of the
confusion matrices.
"""
# @author: drmccloy
# Created on Fri Sep 22 15:13:27 PDT 2017
# License: BSD (3-clause)

import yaml
import os.path as op
from os import mkdir
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hy
from scipy.spatial.distance import pdist


def get_leaf_label(df, ix):
    return df.index[ix]


def order_featmat_rows(featmat, return_intermediates=False):
    # optimally orders the rows of the feature matrix while preserving the
    # hierarchy implicit in the order of the columns.
    mult = np.tile((2 ** np.arange(featmat.shape[1]))[::-1],
                   (featmat.shape[0], 1))
    fm = featmat * mult

    def nanhattan(u, v):
        # get it? NaN-hattan? Treats NaN values as zero distance from any value
        return np.nansum(np.abs(u - v))

    dists = pdist(fm, nanhattan)
    z = hy.linkage(dists, optimal_ordering=True)
    dg = hy.dendrogram(z, no_plot=True)
    returns = featmat.iloc[dg['leaves']]
    if return_intermediates:
        returns = (returns, z, dg)
    return returns


np.set_printoptions(precision=4, linewidth=160)
pd.set_option('display.width', 250)
plt.ion()

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data'
indir = op.join(datadir, 'confusion-matrices')
dgdir = op.join(datadir, 'dendrograms')
rankdir = op.join(datadir, 'feature-rankings')
outdir = op.join(datadir, 'ordered-confusion-matrices')
for _dir in [outdir, dgdir, rankdir]:
    if not op.isdir(_dir):
        mkdir(_dir)

# LOAD PARAMS FROM YAML
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    feature_systems = analysis_params['feature_systems']
    accuracies = analysis_params['theoretical_accuracies']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    canonical_phone_order = analysis_params['canonical_phone_order']
    feature_fnames = analysis_params['feature_fnames']
    pretty_featsys_names = analysis_params['pretty_featsys_names']
    subj_langs = analysis_params['subj_langs']
    methods = analysis_params['methods']
    skip = analysis_params['skip']
del analysis_params

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# load EERs
eers = pd.read_csv(op.join(datadir, 'eers.tsv'), sep='\t', index_col=0)
valid_subjs = [s for s in subjects if s not in skip]
order = eers.mean(axis=1).sort_values().index.tolist()
sorted_eers = eers.loc[order, valid_subjs]
eng_phones = canonical_phone_order['eng']

# load phone-feature matrix
featmat_fname = 'all-features.tsv'
featmat = pd.read_csv(op.join(paramdir, featmat_fname), sep='\t', index_col=0,
                      comment='#')

# loop over methods
for method in methods:
    _eers = sorted_eers.copy()
    _subjects = valid_subjs.copy()
    simulating = (method == 'theoretical')
    if simulating:
        _subjects = {str(accuracy): accuracy for accuracy in accuracies}
        # add the simulation columns to the EER dataframe
        for acc in accuracies:
            _eers[str(acc)] = 1. - acc
        # keep only the simulation columns (not real subject data)
        _eers = _eers[[str(acc) for acc in accuracies]]
    # loop over subjects
    for subj_code in _subjects:
        key = 'theoretical' if simulating else subj_code
        # loop over languages
        for lang in subj_langs[key]:
            # loop over feature systems
            for feat_sys, feats in feature_systems.items():
                # load the data
                middle_arg = '' if simulating else cv + nc
                args = [method, sfn, lang, middle_arg + feat_sys, subj_code]
                fname = '{}-confusion-matrix-{}-{}-{}-{}.tsv'.format(*args)
                fpath = op.join(indir, fname)
                joint_prob = pd.read_csv(fpath, index_col=0, sep='\t')
                # subset the EER dataframe. Here we set the column order based
                # on the mean EER across subjects, then order the rows using a
                # hack of optimal leaf ordering that respects the hierarchy
                # implicit in the column order.  The commented-out lines
                # would instead sort based on EERs for only this subject.
                ordered_feats = np.array(order)[np.in1d(order, feats)]
                this_featmat = featmat.loc[eng_phones, ordered_feats]
                this_featmat = order_featmat_rows(this_featmat)  # asterisk
                '''
                this_eers = _eers.loc[feats, subj_code]
                this_eers.sort_values(inplace=True)
                this_featmat = featmat.loc[eng_phones, this_eers.index]
                this_featmat.sort_values(this_eers.index.tolist(),
                                         inplace=True)
                '''
                # transfer optimal ordering to confusion matrix
                col_ord = this_featmat.index.tolist()
                if lang == 'eng':
                    ordered_prob = joint_prob.loc[col_ord, col_ord]
                else:
                    row_ord = canonical_phone_order[lang]
                    ordered_prob = joint_prob.loc[row_ord, col_ord]
                # save ordered matrix
                out = op.join(outdir, 'feat-ordered-' + fname)
                ordered_prob.to_csv(out, sep='\t')

# asterisk (for testing):
(this_featmat, z, dg) = order_featmat_rows(this_featmat,
                                           return_intermediates=True)
if feat_sys.startswith('phoible'):
    hy.dendrogram(z, leaf_rotation=0, leaf_font_size=14,
                  leaf_label_func=partial(get_leaf_label, this_featmat))
    raise RuntimeError
