#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-row-ord-confmat-with-dgram.py'
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
from aux_functions import (plot_confmat, plot_dendrogram,
                           matrix_row_column_correlation)

plt.ioff()
accuracy = 0.9
savefig = True

# BASIC FILE I/O
paramdir = 'params'
indir = op.join('processed-data', 'ordered-confusion-matrices')
outdir = op.join('figures', 'confusion-matrices')
dgdir = op.join('processed-data', 'dendrograms')
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
    lang_names = analysis_params['pretty_lang_names']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    skip = analysis_params['skip']
del analysis_params

# load dendrogram labels
dg_label_file = op.join(paramdir, 'dendrogram-labels.yaml')
with open(dg_label_file, 'r') as f:
    dg_labels = yaml.load(f)

# only do the 3 original feature systems
del feature_systems['jfh_dense']
del feature_systems['spe_dense']
del feature_systems['phoible_sparse']

feat_sys_names = dict(jfh_sparse='PSA', spe_sparse='SPE',
                      phoible_redux='PHOIBLE')

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
eer = 'eer-' if use_eer else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# load plot style; make colormap with NaN data (from log(0)) mapped as gray
plt.style.use(op.join(paramdir, 'matplotlib-style-confmats.yaml'))
cmap_copy = copy.copy(get_cmap())
# https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems
gray = cmap_copy.colors[127] @ np.array([0.299, 0.587, 0.114])
cmap_copy.set_bad(str(gray))
# cmap_copy.set_bad(cmap_copy.colors[0])

# init containers. Must be done as nested dict and converted afterwards;
# creating as pd.Panel converts embedded DataFrames to ndarrays.
confmats_dict = dict()
# load the data
for feat_sys in feature_systems:
    confmats_dict[feat_sys] = dict()
    # loop over subjects
    for subj_code in subjects:
        if subj_code in skip:
            continue
        prefix = 'cross-subj-row-ordered-eer-confusion-matrix-'
        args = [sfn, cv + nc + feat_sys, subj_code]
        fn = prefix + '{}-eng-{}-{}.tsv'.format(*args)
        confmat = pd.read_csv(op.join(indir, fn), sep='\t', index_col=0)
        confmats_dict[feat_sys][subj_code] = confmat
    # add in theoretical confmats
    prefix = 'cross-subj-row-ordered-theoretical-confusion-matrix-eng-'
    fname = prefix + '{}-{}.tsv'.format(feat_sys, accuracy)
    confmat = pd.read_csv(op.join(indir, fname), sep='\t', index_col=0)
    confmats_dict[feat_sys]['simulated'] = confmat
    # add in average confmats
    prefix = 'cross-subj-row-ordered-eer-confusion-matrix-'
    args = [sfn, cv + nc + feat_sys, 'average']
    fn = prefix + '{}-eng-{}-{}.tsv'.format(*args)
    confmat = pd.read_csv(op.join(indir, fn), sep='\t', index_col=0)
    confmats_dict[feat_sys]['average'] = confmat

# convert to DataFrame of DataFrames. axes: (subj_code, feat_sys)
confmats = pd.DataFrame.from_dict(confmats_dict, dtype=object)

# set common color scale
maxima = confmats.applymap(lambda x: x.values.max())
maxima.to_csv(op.join('processed-data', 'cross-subj-matrix-maxima.tsv'),
              sep='\t')
maximum = maxima.values.max()
normalizer = LogNorm(vmax=maximum)

# init figure
grid_shape = np.array(confmats.shape[::-1]) + np.array([0, 2])
figsize = tuple(grid_shape[::-1] * 2.6)
gridspec_kw = dict(width_ratios=[1, 8] + [4] * (grid_shape[1] - 2))
fig, axs = plt.subplots(*grid_shape, figsize=figsize, gridspec_kw=gridspec_kw)
for row, feat_sys in enumerate(confmats.columns):
    # load dendrogram
    args = [sfn, cv + nc + feat_sys]
    prefix = 'cross-subj-row-ordered-dendrogram-'
    dgfn = prefix + '{}-eng-{}.yaml'.format(*args)
    with open(op.join(dgdir, dgfn), 'r') as dgf:
        dg = yaml.load(dgf)
        # suppress dendrogram colors
        dg['color_list'] = ['0.8'] * len(dg['color_list'])

    # plot vertical dendrogram. do it twice to allow breaking.
    for ix, ax in enumerate(axs[row, :2]):
        plot_dendrogram(dg, orientation='left', ax=ax, linewidth=0.5,
                        leaf_rotation=0)
        ax.invert_yaxis()
        ax.axis('off')  # comment this out to confirm correct ordering
        xlim = (28, 0) if ix else (1001, 999)
        ax.set_xlim(*xlim)
        # annotate dendrogram
        labels = dg_labels[feat_sys]
        order = np.argsort([y[1] for y in dg['dcoord']])[::-1]
        for i, d, n in zip(np.array(dg['icoord'])[order],
                           np.array(dg['dcoord'])[order],
                           labels):
            xy = (d[1], sum(i[1:3]) / 2.)
            kwargs = dict(xytext=(1, 0), textcoords='offset points',
                          va='center', ha='left', fontsize=4)
            if n in ['(strident)', 'continuant+labial']:
                kwargs.update(dict(va='bottom'))
            elif n in ['(strident+continuant)', 'consonantal|labial']:
                kwargs.update(dict(va='top'))
            ax.annotate(n, xy, **kwargs)
            if ix:
                ax.set_title(feat_sys_names[feat_sys])
    # plot simulated confmat
    ax = axs[row, 2]
    data = confmats.loc['simulated', feat_sys]
    diag = matrix_row_column_correlation(data)
    title = '' if row else 'Simulated ({} accuracy)'.format(accuracy)
    title = ' '.join([title, '({:.2f})'.format(diag)])
    kwargs = dict(norm=normalizer, cmap=cmap_copy, title=title,
                  xlabel='', ylabel='')
    plot_confmat(data, ax, **kwargs)
    # plot subject-specific confmats
    cm = confmats.loc[[s for s in list(subjects) if s not in skip]]
    for col, subj_code in enumerate(cm.index, start=3):
        ax = axs[row, col]
        data = confmats.loc[subj_code, feat_sys]
        diag = matrix_row_column_correlation(data)
        title = '' if row else subj_code
        title = ' '.join([title, '({:.2f})'.format(diag)])
        kwargs = dict(norm=normalizer, cmap=cmap_copy, title=title,
                      xlabel='', ylabel='')
        plot_confmat(data, ax, **kwargs)
    # plot average confmat
    ax = axs[row, -1]
    data = confmats.loc['average', feat_sys]
    diag = matrix_row_column_correlation(data)
    title = '' if row else 'Average'
    title = ' '.join([title, '({:.2f})'.format(diag)])
    kwargs = dict(norm=normalizer, cmap=cmap_copy, title=title,
                  xlabel='', ylabel='')
    plot_confmat(data, ax, **kwargs)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.95,
                    wspace=0.2, hspace=0.3)
if savefig:
    fname = 'cross-subj-row-ordered-confusion-matrices-{}-eng.pdf'.format(sfn)
    fig.savefig(op.join(outdir, fname))
else:
    plt.ion()
    plt.show()
