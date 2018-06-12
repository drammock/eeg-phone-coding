#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-ovr.py'
===============================================================================

This script plots error rates from OVR consonant classifiers.  It
addresses the phoneme-level question: which consonants are best-discernable
based on the neural data?
"""
# @author: drmccloy
# Created on Fri Jan 19 14:26:44 PST 2018
# License: BSD (3-clause)

import yaml
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from aux_functions import plot_confmat, format_colorbar_percent


# FLAGS
savefig = True
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 30)
np.set_printoptions(linewidth=240)

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data-OVR'

# load params
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    order = analysis_params['canonical_phone_order']['eng']

# load EERs
fname = 'eers.tsv'
eers = pd.read_csv(op.join(datadir, fname), sep='\t', index_col=0)
eers['average'] = eers.mean(axis=1)
sorted_eers = eers.sort_values('average')

# load confmat
fpath = op.join(datadir, 'confusion-matrices',
                'eer-confusion-matrix-nonan-eng-cvalign-dss5-average.tsv')
confmat = pd.read_csv(fpath, sep='\t', index_col=0)
confmat = confmat.loc[order, order]
'''
# each column is prob(classify_row_token_as_column_label). Convert to
# prob(correct_classification)
diag_indices = np.diag_indices(confmat.shape[0])
correct = confmat.values[diag_indices]
corr_series = pd.Series(correct, index=confmat.index)
print(corr_series)
print(corr_series.describe())
confmat = 1 - confmat
for idx, idy, value in zip(*diag_indices, correct):
    confmat.iloc[idx, idy] = value
'''

# binary colormap
binarycmap = LinearSegmentedColormap.from_list(name='binary', N=3,
                                               colors=['0.9', '0.7', '0.3'])
binarycmap.set_bad('1')
# colormap bounds
cmaps = [('viridis', dict(vmin=1e-5, vmax=1)),
         (binarycmap, dict(vmin=0, vmax=1))]

# init figure
plt.style.use([op.join(paramdir, 'matplotlib-style-confmats.yaml'),
               op.join(paramdir, 'matplotlib-font-myriad.yaml'),
               {'xtick.labelsize': 10, 'ytick.labelsize': 10}])
fig = plt.figure(figsize=(6.75, 4.5))
gs = GridSpec(2, 2, left=0.09, right=0.97, bottom=0.1, top=0.83,
              wspace=0.22, hspace=0., height_ratios=[1, 19])

for ix, (cmap, bounds) in enumerate(cmaps):
    this_confmat = confmat.copy()
    # tweaks for binary
    normfunc = Normalize if cmap == binarycmap else LogNorm
    if cmap == binarycmap:
        maxes = (confmat.idxmax('columns').reset_index()
                 .rename(columns={'ipa': 'row', 0: 'col'}))
        this_confmat[this_confmat >= 0] = 0
        for row, col in maxes.itertuples(index=False):
            this_confmat.loc[row, col] = 0.5
            if row == col:
                this_confmat.loc[row, col] = 1

    # plot confmat
    ax = fig.add_subplot(gs[1, ix])
    ax = plot_confmat(this_confmat, ax=ax, cmap=cmap, norm=normfunc(**bounds))

    # colorbar
    with plt.style.context({'xtick.color': '0.5', 'xtick.major.size': 4,
                            'xtick.labelsize': 8}):
        cax = fig.add_subplot(gs[0, ix])
        cbar = fig.colorbar(ax.images[0], cax=cax, orientation='horizontal')
        cbar.outline.set_linewidth(0.2)
        # scale on top
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
        # colorbar ticks
        if ix:
            tks = [0.5, 5/6]
            cbar.set_ticks(tks)
            cbar.set_ticklabels(['row max.', 'row max.\n& correct'])
            title = 'Most frequent classification'
        else:
            cuts = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
            tks = np.array([np.linspace(a, b, 9, endpoint=False) for a, b in
                            zip(cuts[:-1], cuts[1:])]).flatten().tolist() + [1]
            cbar.set_ticks(tks)
            # change tick colors
            ticks = cax.get_xticklines()
            xy = np.array([t.get_xydata()[0] for t in ticks])
            its = np.where(xy[:, 1] == 1)[0]
            for it, t in enumerate(ticks):
                if it in its:
                    t.set_color('0.75')
                    if its.tolist().index(it) % 9 == 0:
                        t.set_color('k')
            # pretty-format the ticklabels
            ticklabs = [l.get_text() for l in cax.get_xticklabels()]
            ticklabs = format_colorbar_percent(ticklabs)
            cbar.set_ticklabels(ticklabs)
            title = ('One-versus-rest classifiers:\n'
                     'percent classified as target')
            ax.set_ylabel('Stimulus phoneme')
        ax.set_xlabel('Classifier target')
        # subplot labels
        label = ['A', 'B'][ix]
        cax.text(-0.11, 0., label, ha='left', va='baseline',
                 transform=cax.transAxes, fontsize=16, fontweight='bold')
        cbar.set_label(title, labelpad=10, size=14)
        # change ticklabel color (without changing tick color)
        _ = [l.set_color('k') for l in cax.get_xticklabels()]

if savefig:
    fig.savefig(op.join('figures', 'manuscript', 'fig-ovr.pdf'))
else:
    plt.ion()
    plt.show()
