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

import yaml
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from aux_functions import plot_confmat, format_colorbar_percent


# FLAGS
savefig = True

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data-logistic'


paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']

# FEATURE SETS
feature_systems = ['phoible_redux', 'jfh_sparse', 'spe_sparse']
feature_abbrevs = ['PHOIBLE', 'PSA', 'SPE']

gridspec_args = (5, 3)
gridspec_kwargs = dict(left=0.1, right=0.95, bottom=0.075, top=0.925,
                       height_ratios=[1, 20, 20, 20, 20], hspace=0.5)

for featsys, abbrev in zip(feature_systems, feature_abbrevs):
    # set order per the average matrices in the main manuscript
    fname = (f'row-ordered-eer-confusion-matrix-nonan-eng-cvalign-dss5-'
             f'{featsys}-average.tsv')
    fpath = op.join(datadir, 'ordered-confusion-matrices', fname)
    avg_confmat = pd.read_csv(fpath, sep='\t', index_col=0)
    phone_order = avg_confmat.index

    # init figure
    plt.style.use([op.join(paramdir, 'matplotlib-style-confmats.yaml'),
                   {'xtick.labelsize': 7, 'ytick.labelsize': 7}])
    fig = plt.figure(figsize=(6.5, 9))
    gs = GridSpec(*gridspec_args, **gridspec_kwargs)

    for ix, subj_code in enumerate(subjects):
        # load confmat
        fname = (f'cross-subj-row-ordered-eer-confusion-matrix-'
                 f'nonan-eng-cvalign-dss5-{featsys}-{subj_code}.tsv')
        fpath = op.join(datadir, 'ordered-confusion-matrices', fname)
        confmat = pd.read_csv(fpath, sep='\t', index_col=0)
        confmat = confmat.loc[phone_order, phone_order]
        # plot confmat
        row = (ix // 3) + 1
        col = ix % 3
        ax = fig.add_subplot(gs[row, col])
        ax = plot_confmat(confmat, ax=ax, cmap='viridis',
                          norm=LogNorm(vmin=1e-5, vmax=1), title=subj_code)
        # axis labels
        if not col:
            ax.set_ylabel('Stimulus phoneme')
        if row == 4:
            ax.set_xlabel('Predicted phoneme')
    # colorbar
    with plt.style.context({'xtick.color': '0.5', 'xtick.major.size': 4,
                            'xtick.labelsize': 8}):
        cax = fig.add_subplot(gs[0, :])
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
        title = (f'{abbrev} feature-based classifiers’ prediction scores '
                 'by subject')
    cbar.set_label(title, labelpad=10, size=14)
    # change ticklabel color (without changing tick color)
    _ = [l.set_color('k') for l in cax.get_xticklabels()]
    # save
    if savefig:
        fname = f'fig-{abbrev.lower()}.pdf'
        fig.savefig(op.join('figures', 'supplement', fname))
if not savefig:
    plt.ion()
    plt.show()
