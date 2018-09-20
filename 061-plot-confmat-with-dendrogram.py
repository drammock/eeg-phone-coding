#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-confmat-with-dendrogram.py'
===============================================================================

This script plots confusion matrices.
"""
# @author: drmccloy
# Created on Thu Sep  7 15:44:17 PDT 2017
# License: BSD (3-clause)

import copy
import yaml
from os import mkdir
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from aux_functions import plot_confmat, plot_dendrogram

# FLAGS
savefig = True
svm = False
plt.ioff()

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    feature_systems = analysis_params['feature_systems']
    use_eer = analysis_params['use_eer_in_plots']
    # accuracies = analysis_params['theoretical_accuracies']
    # methods = analysis_params['methods']
    use_ordered = analysis_params['sort_matrices']
    # lang_names = analysis_params['pretty_lang_names']
    feat_sys_names = analysis_params['pretty_legend_names']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    skip = analysis_params['skip']
    scheme = analysis_params['classification_scheme']
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

# FILE NAMING VARIABLES
cv = 'cvalign-' if align_on_cv else ''
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
eer = 'eer-' if use_eer else ''
ordered = 'ordered-' if use_ordered else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# BASIC FILE I/O
datadir = f'processed-data{scheme}{trunc}'
indir = op.join(datadir, '{}confusion-matrices'.format(ordered))
outdir = op.join('figures', 'dendrograms')
dgdir = op.join(datadir, 'dendrograms')
outdir = op.join('figures', 'dendrograms')
if not op.isdir(outdir):
    mkdir(outdir)

# load matrix maxima (to set color scale max)
maxima = pd.read_csv(op.join(datadir, 'matrix-maxima.tsv'), sep='\t',
                     index_col=0)
maximum = maxima.values.max()
normalizer = LogNorm(vmax=maximum)

# load EERs
eers = pd.read_csv(op.join(datadir, 'eers.tsv'), sep='\t', index_col=0)
order = eers.mean(axis=1).sort_values().index.tolist()
sorted_eers = eers.loc[order]

# load plot style; make colormap with NaN data mapped as 0 (to handle log(0))
plt.style.use(op.join(paramdir, 'matplotlib-style-confmats.yaml'))
cmap_copy = copy.copy(get_cmap())
cmap_copy.set_bad(cmap_copy.colors[0])
grid_size = 1
grid_shape = (grid_size + 1, grid_size)

# skip some of the usual looping
method = 'eer'
lang = 'eng'
phoible_feats = ['vocalic', 'consonantal', 'nasal', 'flat', 'continuant_spe',
                 'voiced', 'coronal', 'dorsal', 'lateral', 'anterior_sparse',
                 'sonorant', 'distributed', 'delayedrelease',
                 'strident_phoible']
feature_systems = dict(phoible_sparse=phoible_feats)
sortings = ['feat-']  # ['feat-', 'row-', 'col-']
# loop over subjects
for subj_code in subjects:
    if subj_code in skip:
        continue
    # loop over orderings
    for sorting in sortings:
        # init figure
        fig = plt.figure(figsize=(4, 8))
        grid = gs.GridSpec(*grid_shape, left=0.1, right=0.95, bottom=0.01,
                           top=0.975, hspace=0.05)
        # loop over feature systems
        for feat_sys, feats in feature_systems.items():
            # load feature rankings
            indices = np.in1d(sorted_eers.index, feats)
            feat_rank = sorted_eers.loc[indices, subj_code]
            '''
            fn = '{}-feature-rankings.tsv'.format(feat_sys)
            feat_rank = pd.read_csv(op.join(datadir, 'feature-rankings', fn),
                                    sep='\t', header=[0, 1], comment='#')
            feat_rank = feat_rank[subj_code].set_index('feature')
            '''
            # load confmat
            args = [sorting, ordered, method, sfn, lang, cv + nc + feat_sys,
                    subj_code]
            fn = '{}{}{}-confusion-matrix-{}-{}-{}-{}.tsv'.format(*args)
            confmat = pd.read_csv(op.join(indir, fn), sep='\t', index_col=0)

            # plot confmat
            sp = grid.new_subplotspec((0, 0), rowspan=grid_size,
                                      colspan=grid_size)
            confmat_ax = fig.add_subplot(sp)
            title = '{} ({}ordered)'.format(subj_code, sorting)
            kwargs = dict(norm=normalizer, cmap=cmap_copy, title=title)
            plot_confmat(confmat, confmat_ax, **kwargs)

            # load dendrogram
            dgfn = '{}{}{}-dendrogram-{}-{}-{}-{}.yaml'.format(*args)
            with open(op.join(dgdir, dgfn), 'r') as dgf:
                dg = yaml.load(dgf)
            # suppress dendrogram colors
            dg['color_list'] = ['0.7'] * len(dg['color_list'])

            '''
            # plot vertical dendrogram
            sp = grid.new_subplotspec((0, 0), rowspan=grid_size)
            ax = fig.add_subplot(sp)
            plot_dendrogram(dg, orientation='left', ax=ax, linewidth=0.5,
                            leaf_rotation=0)
            ax.invert_yaxis()
            ax.axis('off')  # comment this out to confirm correct ordering
            '''

            # plot horizontal dendrogram
            sp = grid.new_subplotspec((grid_size, 0), colspan=grid_size)
            ax = fig.add_subplot(sp)
            plot_dendrogram(dg, orientation='bottom', ax=ax, linewidth=0.5,
                            leaf_rotation=0)
            ax.axis('off')  # comment this out to confirm correct ordering
            # ax.set_ylim(20, 0)  adjust to zoom in as needed

            # align confmat/dendrogram spacing
            # NB: doing this with transforms doesn't seem to work (good result
            # is contingent on symmetrical figure margins on all sides?!)
            confmat_xlims = confmat_ax.get_xlim()
            confmat_xticks = np.r_[confmat_ax.get_xticks(),
                                   confmat_ax.get_xticks(minor=True)]
            confmat_xticks = [confmat_xticks.min(), confmat_xticks.max()]
            excess_ratio = np.diff(confmat_xlims) / np.diff(confmat_xticks)
            dg_xlims = ax.get_xlim()
            dg_xticks = np.r_[np.array(dg['icoord']).min(),
                              np.array(dg['icoord']).max()]
            modifier = (np.diff(dg_xticks) * excess_ratio - dg_xticks[1])
            new_xlim = dg_xticks + np.r_[-modifier, modifier]
            ax.set_xlim(new_xlim)
            # annotate dendrogram
            order = np.argsort([y[1] for y in dg['dcoord']])[::-1]
            if scheme == 'svm':
                names = ([('− continuant +', False),
                          ('− dorsal +', False),
                          ('+ dorsal −', False),
                          ('− sonorant +', False),
                          ('− delayedrelease +', True),
                          ('− sonorant +', False)] +
                         [('− consonantal +', True)] * 2 +
                         [('− coronal +', True)] * 3 +
                         [('− voi +', False)] * 5 +
                         [('− voi +', True),
                          ('− distr +', True),
                          ('+ distr −', True),
                          ('+ strid −', False),
                          ('− strid +', False),
                          ('− labial +', False)])
            else:
                names = ([('+ nasal −', False),
                          ('− lateral +', False),
                          ('[continuant]', True),
                          ('+ sonorant −', False),
                          ('+ voiced −', True),
                          ('− voiced +', True),
                          ('+ cons −', False),
                          ('− dors +', True),
                          ('+ dors −', True),
                          ('− dors +', True),
                          ('−cont+', False),
                          ('+ cont −', False),
                          ('− lab +', False),
                          ('+ lab −', False),
                          ('[ant]', True),
                          ('− lab. +', False),
                          ('[strid]', True),
                          ('− lab +', False),
                          ('[ant]', True),
                          ('[distr]', True),
                          ('− lab +', False),
                          ('− lab +', False)])
            for i, d, (n, print_below) in zip(np.array(dg['icoord'])[order],
                                              np.array(dg['dcoord'])[order],
                                              names):
                xy = (sum(i[1:3]) / 2., d[1])
                kwargs = dict(xytext=(0, 0.8), textcoords='offset points',
                              va='bottom', ha='center', fontsize=4)
                if print_below:
                    kwargs.update(dict(xytext=(0, -0.8), va='top'))
                ax.annotate(n, xy, **kwargs)

        if savefig:
            args = [sorting, subj_code, scheme, trunc]
            figname = '{}ordered-confmat-dgram-{}-{}{}.pdf'.format(*args)
            fig.savefig(op.join(outdir, figname))

if not savefig:
    plt.ion()
    plt.show()
