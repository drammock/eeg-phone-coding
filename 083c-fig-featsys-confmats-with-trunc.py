#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-featsys-confmats.py'
===============================================================================

This script plots confusion probabilities from feature-based classifiers using
several feature systems.
"""
# @author: drmccloy
# Created on Fri Jan 19 14:26:44 PST 2018
# License: BSD (3-clause)

import yaml
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from aux_functions import plot_confmat, format_colorbar_percent


# FLAGS
savefig = False

# analysis params
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

# FILE NAMING VARIABLES
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data-logistic'
tdatadir = f'processed-data-logistic{trunc}'

# FEATURE SETS
feature_systems = ['phoible_redux', 'jfh_sparse', 'spe_sparse']
feature_abbrevs = ['PHOIBLE', 'PSA', 'SPE']

# binary colormap
binarycmap = LinearSegmentedColormap.from_list(name='binary', N=3,
                                               colors=['0.9', '0.7', '0.3'])
binarycmap.set_bad('1')

# SETUP
figsize = (6.75, 4.25)
gridspec_args = (2, 4)
gridspec_kwargs = dict(left=0.09, right=0.97, bottom=0.1, top=0.85,
                       wspace=0.6, hspace=0., height_ratios=[1, 19])

featsys = 'jfh_sparse'
for featsys, abbrev in zip(feature_systems, feature_abbrevs):
    # load confmat
    fname = ('row-ordered-eer-confusion-matrix-nonan-eng-cvalign-dss5-'
             f'{featsys}-average.tsv')  # individ. row ordering
    fpath = op.join(datadir, 'ordered-confusion-matrices', fname)
    tpath = op.join(tdatadir, 'ordered-confusion-matrices', fname)
    confmat = pd.read_csv(fpath, sep='\t', index_col=0)
    tconfmat = pd.read_csv(tpath, sep='\t', index_col=0)
    # force same row/col order
    tconfmat = tconfmat.loc[confmat.index, confmat.columns]
    # dynamic range change
    dynrange_diff = (np.log10(confmat.values.max() / confmat.values.min()) -
                     np.log10(tconfmat.values.max() / tconfmat.values.min()))
    print(f'{featsys:<15}{np.round(dynrange_diff, 3)}')

    # init figure
    plt.style.use([op.join(paramdir, 'matplotlib-style-confmats.yaml'),
                   op.join(paramdir, 'matplotlib-font-myriad.yaml'),
                   {'xtick.labelsize': 10, 'ytick.labelsize': 10}])
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(*gridspec_args, **gridspec_kwargs)
    # separate figure for rowmax plots (see below)
    fig2, axs2 = plt.subplots(1, 2)

    # loop over subplots
    for ix, cmat in enumerate((confmat, tconfmat)):
        this_confmat = cmat.copy()
        # plot confmat
        ax = fig.add_subplot(gs[1, (2 * ix):(2 * ix + 2)])
        ax = plot_confmat(this_confmat, ax=ax, cmap='viridis',
                          norm=LogNorm(vmin=1e-5, vmax=1))
        # subplot labels
        xpos = (0, 1)[ix]
        label = ['A  Full epochs', 'Truncated  B'][ix]
        kwargs = dict(ha='right') if ix else dict(ha='left')
        ax.text(xpos, 1.05, label, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='baseline', **kwargs)
        # axis labels
        if not ix:
            ax.set_ylabel('Stimulus phoneme')
        ax.set_xlabel('Predicted phoneme')

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # aside: rowmax plots (not saved to file; use savefig=False)
        maxes = (this_confmat.idxmax('columns').reset_index()
                 .rename(columns={'ipa': 'row', 0: 'col'}))
        # set all non-NaNs to zero
        _confmat = this_confmat.copy()
        _confmat[_confmat >= 0] = 0
        # make row maxima light gray, and diagonal row maxima dark gray
        for row, col in maxes.itertuples(index=False):
            _confmat.loc[row, col] = 0.5
            if row == col:
                this_confmat.loc[row, col] = 1
        plot_confmat(_confmat, ax=axs2[ix], cmap=binarycmap,
                     norm=Normalize(vmin=0, vmax=1))
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # colorbar
    with plt.style.context({'xtick.color': '0.5', 'xtick.major.size': 4,
                            'xtick.labelsize': 8}):
        cax = fig.add_subplot(gs[0, 1:3])
        cbar = fig.colorbar(ax.images[0], cax=cax,
                            orientation='horizontal')
        cbar.outline.set_linewidth(0.2)
        # scale on top
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
        # colorbar ticks
        cuts = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        tks = np.array([np.linspace(a, b, 9, endpoint=False)
                        for a, b in zip(cuts[:-1], cuts[1:])
                        ]).flatten().tolist() + [1]  # logspace ticks
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
        title = f'{abbrev} feature-based classifiers’ prediction scores'
        cbar.set_label(title, labelpad=10, size=14)
        # change ticklabel color (without changing tick color)
        _ = [l.set_color('k') for l in cax.get_xticklabels()]
    # save
    if savefig:
        fname = f'fig-{abbrev.lower()}.pdf'
        fig.savefig(op.join('figures', 'manuscript', fname))
if not savefig:
    plt.ion()
    plt.show()
