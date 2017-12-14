#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-single-feature-confmats.py'
===============================================================================

This script plots a series of single-feature confusion matrices.
"""
# @author: drmccloy
# Created on Thu Dec 14 11:21:52 PST 2017
# License: BSD (3-clause)

import copy
import yaml
from os import mkdir
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize  # LogNorm
from aux_functions import plot_confmat

# BASIC FILE I/O
paramdir = 'params'
indir = op.join('processed-data', 'single-feat-confmats')
outdir = op.join('figures', 'single-feat-confmats')
feature_sys_fname = 'all-features.tsv'
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS FROM YAML
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    feature_systems = analysis_params['feature_systems']
    canonical_phone_order = analysis_params['canonical_phone_order']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    skip = analysis_params['skip']
del analysis_params

# only do the 3 original feature systems
del feature_systems['jfh_dense']
del feature_systems['spe_dense']
del feature_systems['phoible_sparse']

feat_sys_names = dict(jfh_sparse='PSA', spe_sparse='SPE',
                      phoible_redux='PHOIBLE')

# get the preferred cross-subject phone ordering (same for all subjs.)
dg_file = ('cross-subj-row-ordered-dendrogram-nonan-eng-'
           'cvalign-dss5-jfh_dense-CQ.yaml')
with open(op.join('processed-data', 'dendrograms', dg_file), 'r') as f:
    dg = yaml.load(f)
    phone_order = dg['ivl']

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# load plot style; make colormap with NaN data (from log(0)) mapped as gray
plt.style.use(op.join(paramdir, 'matplotlib-style-confmats.yaml'))
cmap_copy = copy.copy(get_cmap())
# https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems
gray = cmap_copy.colors[127] @ np.array([0.299, 0.587, 0.114])
cmap_copy.set_bad(str(gray))
normalizer = Normalize(vmin=0, vmax=1)

# loop over feature systems
for this_sys, this_feats in feature_systems.items():

    # initialize figure
    grid_shape = np.array([len(this_feats), len(subjects)])
    figsize = tuple(grid_shape[::-1] * 2.5)
    fig, axs = plt.subplots(*grid_shape, figsize=figsize)

    # loop over subjects
    for col_ix, subj_code in enumerate(subjects):
        if subj_code in skip:
            continue

        # loop over features
        for row_ix, feat in enumerate(this_feats):
            ax = axs[row_ix, col_ix]
            # load confusion matrix
            args = [sfn, cv + nc + feat, subj_code]
            fname = 'eer-confusion-matrix-{}-eng-{}-{}.tsv'.format(*args)
            data = pd.read_csv(op.join(indir, fname), sep='\t', index_col=0)
            data = data.loc[phone_order, phone_order]
            '''
            if not sparse_feature_nan:
                data[np.isnan(data)] = 0.5
            '''
            title = '' if row_ix else subj_code
            ylabel = '' if col_ix else feat
            kwargs = dict(norm=normalizer, cmap=cmap_copy, title=title,
                          xlabel='', ylabel=ylabel)
            ax = plot_confmat(data, ax, **kwargs)
    fig.suptitle(feat_sys_names[this_sys])
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        wspace=0.2, hspace=0.2)
    fig.savefig(op.join(outdir, '{}.pdf'.format(feat_sys_names[this_sys])))
    plt.close(fig)
