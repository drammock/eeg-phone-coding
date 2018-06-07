#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-psa.py'
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
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from aux_functions import plot_confmat


# FLAGS
savefig = True
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 30)
np.set_printoptions(linewidth=240)

# BASIC FILE I/O
paramdir = 'params'
outdir = op.join('figures', 'jobtalk')
datadir = 'processed-data-logistic'

# FEATURE SETS
feature_systems = ['phoible_redux', 'jfh_sparse', 'spe_sparse']
feature_abbrevs = ['PHOIBLE', 'PSA', 'SPE']

phone_order = list()
for featsys, abbrev in zip(feature_systems, feature_abbrevs):
    # load confmat
    #fname = ('cross-featsys-cross-subj-row-ordered-eer-confusion-matrix-'
    #         'nonan-eng-cvalign-dss5-{}-average.tsv'.format(featsys))
    fname = ('cross-subj-row-ordered-eer-confusion-matrix-'
             'nonan-eng-cvalign-dss5-{}-average.tsv'.format(featsys))
    fpath = op.join(datadir, 'ordered-confusion-matrices', fname)
    confmat = pd.read_csv(fpath, sep='\t', index_col=0)
    if featsys == 'phoible_redux':
        phone_order = confmat.index.tolist()
    confmat = confmat.loc[phone_order, phone_order]

    # init figure
    plt.style.use(['dark_background',
                   op.join(paramdir, 'matplotlib-style-confmats.yaml'),
                   {'xtick.labelsize': 10, 'ytick.labelsize': 10,
                    'xtick.color': '0.5', 'ytick.color': '0.5'}])
    fig = plt.figure(figsize=(5, 6))
    gs = GridSpec(2, 1, left=0.06, right=0.96, bottom=0.08, top=0.88,
                  hspace=0.1, height_ratios=[1, 19])

    # plot confmat
    ax = fig.add_subplot(gs[1])
    ax = plot_confmat(confmat, ax=ax, norm=LogNorm(vmin=1e-5, vmax=1),
                      cmap='viridis')

    # change tick label color without affecting tick line color
    _ = [l.set_color('w') for l in ax.get_xticklabels(which='both')]
    _ = [l.set_color('w') for l in ax.get_yticklabels(which='both')]

    # colorbar
    plt.style.use({'xtick.major.size': 3,
                   'xtick.labelsize': 12, 'font.family': 'sans-serif'})
    cax = fig.add_subplot(gs[0])
    cbar = fig.colorbar(ax.images[0], cax=cax, orientation='horizontal')
    # scale on top
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    title = 'Classification score ({} feature classifiers)'.format(abbrev)
    cax.set_xlabel(title, labelpad=12, size=14)
    # ticks
    cuts = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ticks = np.array([np.linspace(a, b, 10, endpoint=False) for a, b in
                      zip(cuts[:-1], cuts[1:])]).flatten().tolist() + [1]
    cbar.set_ticks(ticks)
    # hack a slight rightward shift of ticklabels by padding with a space
    cbar.set_ticklabels([' {}'.format(l.get_text()) for l in
                         cax.get_xticklabels()])
    _ = [l.set_color('w') for l in cax.get_xticklabels(which='both')]
    # colorbar frame
    cbar.outline.set_linewidth(0)
    # highlight boxes
    cell_zero = ('d', 'd')
    cell_one = ('dʒ', 'd')
    cell_two = ('b', 'd')
    box_zero = ('θ', 2)
    box_one = ('z', 2)
    box_two = ('ʃ', 4)
    # save
    if savefig:
        fig.savefig(op.join(outdir, 'fig-{}.pdf'.format(abbrev.lower())))
    if featsys == 'phoible_redux':
        for count, cell in enumerate((cell_zero, cell_one, cell_two)):
            ix = (confmat.index.get_loc(cell[0]),
                  confmat.columns.get_loc(cell[1]))
            xy = np.array(ix) - 0.5
            rect = Rectangle(xy, width=1, height=1, facecolor='none',
                             edgecolor='w', linewidth=2)
            ax.add_artist(rect)
            fname = 'fig-{}-highlight-{}.pdf'.format(abbrev.lower(), count)
            fig.savefig(op.join(outdir, fname))
            rect.remove()
        for count, box in enumerate((box_zero, box_one, box_two),
                                    start=(count + 1)):
            ix = (confmat.index.get_loc(box[0]),
                  confmat.columns.get_loc(box[0]))
            xy = np.array(ix) - 0.5
            rect = Rectangle(xy, width=box[1], height=box[1], facecolor='none',
                             edgecolor='w', linewidth=2, clip_on=False)
            ax.add_artist(rect)
            fname = 'fig-{}-highlight-{}.pdf'.format(abbrev.lower(), count)
            fig.savefig(op.join(outdir, fname))
            rect.remove()

if not savefig:
    plt.ion()
    plt.show()
