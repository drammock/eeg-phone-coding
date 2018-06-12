#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-featsys-confmats.py'
===============================================================================

This script plots confusion probabilities from feature-based classifiers using
the Jakobson-Fant-Halle feature set (PSA = Preliminaries to Speech Analysis).
"""
# @author: drmccloy
# Created on Fri Jan 19 14:26:44 PST 2018
# License: BSD (3-clause)

import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from aux_functions import plot_confmat, format_colorbar_percent


# FLAGS
savefig = True
show_voting = True

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data-logistic'

# FEATURE SETS
feature_systems = ['phoible_redux', 'jfh_sparse', 'spe_sparse']
feature_abbrevs = ['PHOIBLE', 'PSA', 'SPE']

# binary colormap
binarycmap = LinearSegmentedColormap.from_list(name='binary', N=3,
                                               colors=['0.9', '0.7', '0.3'])
binarycmap.set_bad('1')

if show_voting:
    figsize = (6.75, 4.5)
    gridspec_args = (2, 2)
    gridspec_kwargs = dict(left=0.09, right=0.97, bottom=0.1, top=0.83,
                           wspace=0.22, hspace=0., height_ratios=[1, 19])
    cmaps = [('viridis', dict(vmin=1e-5, vmax=1)),
             (binarycmap, dict(vmin=0, vmax=1))]
else:
    figsize = (4, 5)
    gridspec_args = (2, 1)
    gridspec_kwargs = dict(left=0.14, right=0.93, bottom=0.11, top=0.86,
                           height_ratios=[1, 19])
    cmaps = [('viridis', dict(vmin=1e-5, vmax=1))]

for featsys, abbrev in zip(feature_systems, feature_abbrevs):
    # load confmat
    fname = ('row-ordered-eer-confusion-matrix-nonan-eng-cvalign-dss5-'
             '{}-average.tsv'.format(featsys))  # individ. row ordering
    #fname = ('cross-featsys-cross-subj-row-ordered-eer-confusion-matrix-'
    #         'nonan-eng-cvalign-dss5-{}-average.tsv'.format(featsys))
    #fname = ('cross-subj-row-ordered-eer-confusion-matrix-'
    #         'nonan-eng-cvalign-dss5-{}-average.tsv'.format(featsys))
    fpath = op.join(datadir, 'ordered-confusion-matrices', fname)
    confmat = pd.read_csv(fpath, sep='\t', index_col=0)

    # init figure
    plt.style.use([op.join(paramdir, 'matplotlib-style-confmats.yaml'),
                   op.join(paramdir, 'matplotlib-font-myriad.yaml'),
                   {'xtick.labelsize': 10, 'ytick.labelsize': 10}])
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(*gridspec_args, **gridspec_kwargs)

    # loop over subplots
    for ix, (cmap, bounds) in enumerate(cmaps):
        this_confmat = confmat.copy()
        # tweaks for binary
        normfunc = Normalize if cmap == binarycmap else LogNorm
        if cmap == binarycmap:
            maxes = (confmat.idxmax('columns').reset_index()
                     .rename(columns={'ipa': 'row', 0: 'col'}))
            # set all non-NaNs to zero
            this_confmat[this_confmat >= 0] = 0
            # make row maxima light gray, and diagonal row maxima dark gray
            for row, col in maxes.itertuples(index=False):
                this_confmat.loc[row, col] = 0.5
                if row == col:
                    this_confmat.loc[row, col] = 1

        # plot confmat
        ax = fig.add_subplot(gs[1, ix])
        ax = plot_confmat(this_confmat, ax=ax, cmap=cmap,
                          norm=normfunc(**bounds))

        # colorbar
        with plt.style.context({'xtick.color': '0.5', 'xtick.major.size': 4,
                                'xtick.labelsize': 8}):
            cax = fig.add_subplot(gs[0, ix])
            cbar = fig.colorbar(ax.images[0], cax=cax,
                                orientation='horizontal')
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
                title = ('{} feature-based classifiers’\n'
                         'prediction scores').format(abbrev)
                # axis labels
                ax.set_ylabel('Stimulus phoneme')
            ax.set_xlabel('Predicted phoneme')
            # subplot labels
            label = ['A', 'B'][ix]
            cax.text(-0.11, 0., label, ha='left', va='baseline',
                     transform=cax.transAxes, fontsize=16, fontweight='bold')
        cbar.set_label(title, labelpad=10, size=14)
        # change ticklabel color (without changing tick color)
        _ = [l.set_color('k') for l in cax.get_xticklabels()]
        # save
        if savefig:
            fname = 'fig-{}.pdf'.format(abbrev.lower())
            fig.savefig(op.join('figures', 'manuscript', fname))
if not savefig:
    plt.ion()
    plt.show()
