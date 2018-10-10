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
from matplotlib.colors import LogNorm, Normalize
from aux_functions import plot_confmat, format_colorbar_percent


# FLAGS
savefig = True

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data-logistic'

# FEATURE SETS
feature_systems = ['jfh_sparse', 'spe_sparse', 'phoible_redux']
feature_abbrevs = ['PSA', 'SPE', 'PHOIBLE']

# SETUP
figsize = (6.75, 3.5)
gridspec_args = (2, 3)
gridspec_kwargs = dict(left=0.08, right=0.98, bottom=0.07, top=0.85,
                       wspace=0.22, hspace=0., height_ratios=[1, 19])

# init figure
plt.style.use([op.join(paramdir, 'matplotlib-style-confmats.yaml'),
               op.join(paramdir, 'matplotlib-font-myriad.yaml'),
               {'xtick.labelsize': 10, 'ytick.labelsize': 10}])
fig = plt.figure(figsize=figsize)
gs = GridSpec(*gridspec_args, **gridspec_kwargs)

for ix, (featsys, abbrev) in enumerate(zip(feature_systems, feature_abbrevs)):
    # load confmat
    #fname = ('row-ordered-eer-confusion-matrix-nonan-eng-cvalign-dss5-'
    #         f'{featsys}-average.tsv')  # individ. row ordering
    fname = ('cross-featsys-cross-subj-row-ordered-eer-confusion-matrix-'
             f'nonan-eng-cvalign-dss5-{featsys}-average.tsv')
    fpath = op.join(datadir, 'ordered-confusion-matrices', fname)
    confmat = pd.read_csv(fpath, sep='\t', index_col=0)
    # plot confmat
    ax = fig.add_subplot(gs[1, ix])
    ax = plot_confmat(confmat, ax=ax, cmap='viridis',
                      norm=LogNorm(vmin=1e-5, vmax=1))
    # subplot labels
    #xpos = (0, 1)[ix]
    #label = ['A', 'B'][ix]
    #kwargs = dict(ha='right') if ix else dict(ha='left')
    #ax.text(xpos, 1.05, label, transform=ax.transAxes, fontsize=16,
    #        fontweight='bold', va='baseline', **kwargs)
    # axis labels
    if not ix:
        ax.set_ylabel('Stimulus phoneme')
    ax.set_xlabel('Predicted phoneme')
    # title
    ax.set_title(abbrev)

# colorbar
with plt.style.context({'xtick.color': '0.5', 'xtick.major.size': 4,
                        'xtick.labelsize': 8}):
    cax = fig.add_subplot(gs[0, 1])
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
    title = f'Feature-based classifiers’ prediction scores (jointly ordered)'
    cbar.set_label(title, labelpad=10, size=14)
    # change ticklabel color (without changing tick color)
    _ = [l.set_color('k') for l in cax.get_xticklabels()]
# save
if savefig:
    fname = 'fig-confmat-allthree.pdf'
    fig.savefig(op.join('figures', 'supplement', fname))
else:
    plt.ion()
    plt.show()
