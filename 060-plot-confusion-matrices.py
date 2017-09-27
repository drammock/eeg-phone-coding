#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-confusion-matrices.py'
===============================================================================

This script plots confusion matrices.
"""
# @author: drmccloy
# Created on Mon Aug 21 16:52:12 PDT 2017
# License: BSD (3-clause)

import copy
import yaml
from os import mkdir
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from aux_functions import plot_confmat

plt.ioff()
pd.set_option('display.width', 130)

# BASIC FILE I/O
paramdir = 'params'
# indir defined below, after loading YAML parameters
outdir = op.join('figures', 'confusion-matrices')
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
    subj_langs = analysis_params['subj_langs']
    use_eer = analysis_params['use_eer_in_plots']
    accuracies = analysis_params['theoretical_accuracies']
    methods = analysis_params['methods']
    use_ordered = analysis_params['sort_matrices']
    lang_names = analysis_params['pretty_lang_names']
    feat_sys_names = analysis_params['pretty_legend_names']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    skip = analysis_params['skip']
del analysis_params

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
eer = 'eer-' if use_eer else ''
ordered = 'ordered-' if use_ordered else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'
indir = op.join('processed-data', '{}confusion-matrices'.format(ordered))

# load plot style; make colormap with NaN data mapped as 0 (to handle log(0))
plt.style.use(op.join(paramdir, 'matplotlib-style-confmats.yaml'))
cmap_copy = copy.copy(get_cmap())
cmap_copy.set_bad(cmap_copy.colors[0])

# loop over methods (phone-level, feature-level-eer, system-level-simulated)
for method in methods:
    simulating = (method == 'theoretical')
    _subjs = {str(acc): acc for acc in accuracies} if simulating else subjects
    # init containers. Must be done as nested dict and converted afterwards;
    # creating as pd.Panel converts embedded DataFrames to ndarrays.
    confmats_dict = {s: dict() for s in list(_subjs) if s not in skip}
    # load the data
    for subj_code in _subjs:
        if subj_code in skip:
            continue
        key = 'theoretical' if simulating else subj_code
        # loop over matrix row/column sortings
        for sorting in ['feat-', 'row-', 'col-', '']:
            sort_key = sorting[:-1] if len(sorting) else 'rowcol'
            confmats_dict[subj_code][sort_key] = dict()
            for feat_sys in feature_systems:
                middle_arg = feat_sys if simulating else cv + nc + feat_sys
                args = [sorting, ordered, method, sfn, middle_arg, subj_code]
                fn = '{}{}{}-confusion-matrix-{}-eng-{}-{}.tsv'.format(*args)
                confmat = pd.read_csv(op.join(indir, fn), sep='\t',
                                      index_col=0)
                confmats_dict[subj_code][sort_key][feat_sys] = confmat

    # convert to Panel. axes: (subj_code, feat_sys, sort_key)
    confmats = pd.Panel.from_dict(confmats_dict, dtype=object)
    # set common color scale
    maxima = confmats.apply(lambda x: x.applymap(lambda y: y.max().max()
                                                 ), axis=(0, 1))
    maximum = maxima.loc['row'].max().max()  # invariant across sortings
    normalizer = LogNorm(vmax=maximum)

    # loop over matrix sorting methods
    for sorting in ['feat-', 'row-', 'col-', '']:
        sort_key = sorting[:-1] if len(sorting) else 'rowcol'
        this_confmats = confmats.loc[:, :, sort_key]
        # init figure
        figsize = tuple(np.array(this_confmats.shape)[::-1] * 2.6)
        fig, axs = plt.subplots(*this_confmats.shape, figsize=figsize)
        for row, feat_sys in enumerate(this_confmats.index):
            for col, subj_code in enumerate(this_confmats.columns):
                ax = axs[row, col]
                data = this_confmats.loc[feat_sys, subj_code]
                title = '' if row else subj_code
                ylabel = '' if col else feat_sys_names[feat_sys]
                xlabel = ('' if row != (len(this_confmats.index) - 1) else
                          'English')
                kwargs = dict(norm=normalizer, cmap=cmap_copy, title=title,
                              xlabel=xlabel, ylabel=ylabel)
                plot_confmat(data, ax, **kwargs)
        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.05, top=0.97,
                            wspace=0.3, hspace=0.4)
        args = [method, sorting, ordered, sfn]
        figname = '{}-{}{}confusion-matrices-{}-eng.pdf'.format(*args)
        fig.savefig(op.join(outdir, figname))
