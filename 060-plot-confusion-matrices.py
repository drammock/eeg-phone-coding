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
from matplotlib.colors import Normalize, LogNorm, NoNorm

plt.ioff()
pd.set_option('display.width', 140)

# BASIC FILE I/O
paramdir = 'params'
indir = op.join('processed-data', 'confusion-matrices')
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
    skip = analysis_params['skip']
del analysis_params

subjects['theory'] = -1  # dummy value

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
eer = 'eer-' if use_eer else ''

# load plot style; make colormap with NaN data mapped as 0 (to handle log(0))
plt.style.use(op.join(paramdir, 'matplotlib-style-confmats.yaml'))
cmap_copy = copy.copy(get_cmap())
cmap_copy.set_bad(cmap_copy.colors[0])

# pretty names for axis labels
lang_names = dict(hin='Hindi', swh='Swahili', hun='Hungarian', nld='Dutch',
                  eng='English')
feat_sys_names = dict(jfh_dense='Jakobson Fant &\nHalle (dense)',
                      jfh_sparse='Jakobson Fant &\nHalle (dense)',
                      spe_dense='Chomsky & Halle\n(dense)',
                      spe_sparse='Chomsky & Halle\n(sparse)',
                      phoible_sparse='Moran McCloy &\nWright (sparse)')

# loop over methods (phone-level vs. uniform-eer)
for prefix in ['phone', 'eer', 'theoretical']:
    simulating = prefix == 'theoretical'
    if simulating:
        subjects = {str(acc): acc for acc in accuracies}
    # init containers. Must be done as nested dict and converted afterwards;
    # creating as pd.Panel converts embedded DataFrames to ndarrays.
    confmats_dict = {s: dict() for s in list(subjects) if s not in skip}
    # load the data
    for subj_code in subjects:
        if subj_code in skip:
            continue
        key = 'theory' if simulating else subj_code
        for lang in subj_langs[key]:
            confmats_dict[subj_code][lang] = dict()
            for feat_sys in feature_systems:
                middle_arg = feat_sys if simulating else cv + nc + feat_sys
                args = [prefix, lang, middle_arg, subj_code]
                fname = '{}-confusion-matrix-{}-{}-{}.tsv'.format(*args)
                confmat = pd.read_csv(op.join(indir, fname), sep='\t',
                                      index_col=0)
                confmats_dict[subj_code][lang][feat_sys] = confmat
    # convert to Panel. axes: (subj_code, feat_sys, lang)
    confmats = pd.Panel.from_dict(confmats_dict, dtype=object)

    # set common color scale. Mapping NaN to -1 handles subject+language
    # combinations that didn't occur in the experiment (each subj heard only
    # 2 of the 4 non-English languages)
    maxima = confmats.apply(lambda x: x.applymap(lambda y: y.max().max()
                                                 if not np.all(np.isnan(y))
                                                 else -1.), axis=(0, 1))
    #maximum = maxima.max().max().max()
    #normalizer = LogNorm(vmax=maximum)
    #normalizer = Normalize(vmax=maximum)

    # English only plot, comparing subjs. and feature systems
    eng_confmats = confmats.loc[:, :, 'eng']
    eng_max = maxima.loc['eng'].max().max()
    normalizer = LogNorm(vmax=eng_max)

    figsize = tuple(np.array(eng_confmats.shape)[::-1] * 2.6)
    fig, axs = plt.subplots(*eng_confmats.shape, figsize=figsize)
    for row, feat_sys in enumerate(eng_confmats.index):
        for col, subj_code in enumerate(eng_confmats.columns):
            ax = axs[row, col]
            data = eng_confmats.loc[feat_sys, subj_code]
            ax.imshow(data, origin='upper', norm=normalizer, cmap=cmap_copy)
            if not row:
                ax.set_title(subj_code)
            if not col:
                ax.set_ylabel(feat_sys_names[feat_sys])
                #ax.set_ylabel(lang_names[lang])
            # label the axes
            ax.set_xticks(np.arange(data.shape[1])[1::2], minor=False)
            ax.set_xticks(np.arange(data.shape[1])[::2], minor=True)
            ax.set_yticks(np.arange(data.shape[0])[1::2], minor=False)
            ax.set_yticks(np.arange(data.shape[0])[::2], minor=True)
            ax.set_xticklabels(data.columns[1::2], minor=False)
            ax.set_xticklabels(data.columns[::2], minor=True)
            ax.set_yticklabels(data.index[1::2], minor=False)
            ax.set_yticklabels(data.index[::2], minor=True)
            if row == (len(eng_confmats.index) - 1):
                ax.set_xlabel('English')

    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.05, top=0.97,
                        wspace=0.3, hspace=0.4)
    figname = '{}-confusion-matrices-subj-x-featsys-ENG.pdf'.format(prefix)
    fig.savefig(op.join(outdir, figname))
    #fig.show()

"""
# calculate figure size
matrix_width = 3 * confmat.shape[1]
heights = np.array([confmats[lg].shape[0] for lg in langs])
figsize = np.array([matrix_width, heights.sum()]) * figwidth / matrix_width
# initialize figure
fig = plt.figure(figsize=figsize)
axs = ImageGrid(fig, 111, nrows_ncols=(len(langs), ncol), axes_pad=0.5,
                label_mode='all')
"""
