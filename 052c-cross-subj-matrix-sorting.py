#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'cross-subj-matrix-sorting.py'
===============================================================================

This script uses "optimal leaf ordering" to sort the rows and columns of the
confusion matrices.
"""
# @author: drmccloy
# Created on Mon Sep  4 11:13:58 PDT 2017
# License: BSD (3-clause)

import yaml
from os import mkdir
import pandas as pd
import os.path as op
from aux_functions import optimal_leaf_ordering

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data'
indir = op.join(datadir, 'confusion-matrices')
dgdir = op.join(datadir, 'dendrograms')
outdir = op.join(datadir, 'ordered-confusion-matrices')
for _dir in [outdir, dgdir]:
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
    subj_langs = analysis_params['subj_langs']
    accuracies = analysis_params['theoretical_accuracies']
    methods = analysis_params['methods']
    skip = analysis_params['skip']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
del analysis_params

# only do the 3 original feature systems
del feature_systems['jfh_dense']
del feature_systems['spe_dense']
del feature_systems['phoible_sparse']

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# sort by cross-subj similarity by stacking confmats before running OLO
confmats = dict()
for feat_sys in feature_systems:
    confmats[feat_sys] = dict()
    for subj_code in subjects:
        if subj_code in skip:
            continue
        # load the data
        args = [sfn, cv + nc + feat_sys, subj_code]
        fname = 'eer-confusion-matrix-{}-eng-{}-{}.tsv'.format(*args)
        fpath = op.join(indir, fname)
        confmat = pd.read_csv(fpath, index_col=0, sep='\t')
        confmats[feat_sys][subj_code] = confmat

# pick some standard initial row order for everyone before concatenating
row_order = confmats['phoible_redux']['IJ'].index.tolist()

# do OLO on a per-feat-sys level and on a cross-feat-sys level
confmat_list = list()
for feat_sys in feature_systems:
    this_confmat_list = list()
    for subj_code in subjects:
        if subj_code in skip:
            continue
        this_confmat_list.append(confmats[feat_sys][subj_code].loc[row_order])
        confmat_list.append(confmats[feat_sys][subj_code].loc[row_order])

    # run optimal leaf ordering algorithm for this feature system
    this_merged_confmat = pd.concat(this_confmat_list, axis=1)
    olo = optimal_leaf_ordering(this_merged_confmat)
    dendrograms, linkages = olo['dendrograms'], olo['linkages']
    new_row_order = dendrograms['row']['ivl']

    # re-loop over subjects to apply new row order & save
    prefix = 'cross-subj-row-ordered-'
    for subj_code in subjects:
        if subj_code in skip:
            continue
        # save confmat
        args = [sfn, cv + nc + feat_sys, subj_code]
        fname = 'eer-confusion-matrix-{}-eng-{}-{}.tsv'.format(*args)
        out = op.join(outdir, prefix + fname)
        cm = confmats[feat_sys][subj_code].loc[new_row_order, new_row_order]
        cm.to_csv(out, sep='\t')
    # save dendrogram objects
    args = [sfn, cv + nc + feat_sys]
    dg_fname = 'dendrogram-{}-eng-{}.yaml'.format(*args)
    with open(op.join(dgdir, prefix + dg_fname), 'w') as dgf:
        yaml.dump(dendrograms['row'], dgf, default_flow_style=True)

    # make theoretical confmats of various accuracies with same order
    for accuracy in [0.6, 0.7, 0.8, 0.9, 0.99, 0.999]:
        fname = ('theoretical-confusion-matrix-eng-{}-{}.tsv'
                 ''.format(feat_sys, accuracy))
        simulated = pd.read_csv(op.join(indir, fname), sep='\t', index_col=0)
        simulated = simulated.loc[new_row_order, new_row_order]
        out_fname = 'cross-subj-row-ordered-' + fname
        simulated.to_csv(op.join(outdir, out_fname), sep='\t')

# run optimal leaf ordering algorithm across all feat systems
merged_confmat = pd.concat(confmat_list, axis=1)
olo = optimal_leaf_ordering(merged_confmat)
dendrograms, linkages = olo['dendrograms'], olo['linkages']
new_row_order = dendrograms['row']['leaves']

# apply new row order & save
prefix = 'cross-featsys-cross-subj-row-ordered-'
for feat_sys in feature_systems:
    for subj_code in subjects:
        if subj_code in skip:
            continue
        # save confmat
        args = [sfn, cv + nc + feat_sys, subj_code]
        fname = 'eer-confusion-matrix-{}-eng-{}-{}.tsv'.format(*args)
        out = op.join(outdir, prefix + fname)
        cm = confmats[feat_sys][subj_code].iloc[new_row_order, new_row_order]
        cm.to_csv(out, sep='\t')
# save dendrogram object
args = [sfn, cv + nc]
dg_fname = 'dendrogram-{}-eng-{}.yaml'.format(*args)
with open(op.join(dgdir, prefix + dg_fname), 'w') as dgf:
    yaml.dump(dendrograms['row'], dgf, default_flow_style=True)
