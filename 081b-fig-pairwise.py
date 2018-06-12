#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-avg-pairwise-error.py'
===============================================================================

This script plots error rates from pairwise consonant classifiers.  It
addresses the phoneme-level question: which consonants are best-discernable
based on the neural data?
"""
# @author: drmccloy
# Created on Fri Jan 19 14:26:44 PST 2018
# License: BSD (3-clause)

import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from aux_functions import plot_confmat, format_colorbar_percent


def plot_eers(df, ax, marker='o', legend=False, title='', ylim=(-0.05, 1.05),
              legend_bbox=(1.1, 1.), markersize=4, linewidth=0.5,
              jitter=False):
    x = np.tile(np.arange(df.shape[0]), (df.shape[1], 1))
    if jitter:
        x = x + 0.4 * (0.5 - np.random.rand(*x.shape))
    lines = ax.plot(x.T, df, marker=marker, markersize=markersize, alpha=0.6,
                    linewidth=linewidth)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(df.shape[0]))
    if legend:
        handles = lines
        labels = df.columns.tolist() + ['mean']
        ax.legend(handles, labels, loc='upper left',
                  bbox_to_anchor=legend_bbox)
    return ax


# FLAGS
savefig = True
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 30)
np.set_printoptions(linewidth=240)

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data-pairwise'

# load EERs
fname = 'eers.tsv'
eers = pd.read_csv(op.join(datadir, fname), sep='\t', index_col=0)
eers['average'] = eers.mean(axis=1)
sorted_eers = eers.sort_values('average')

# load confmat
fpath = op.join(datadir, 'confusion-matrices',
                'eer-confusion-matrix-nonan-eng-cvalign-dss5-average.tsv')
confmat = pd.read_csv(fpath, sep='\t', index_col=0)
order = confmat.sum(axis=1).sort_values().index
# lower tri
# np.fill_diagonal(confmat.values, 0.)
confmat_tri = pd.DataFrame(np.tril(confmat), index=confmat.index,
                           columns=confmat.columns)

# init figure
plt.style.use([op.join(paramdir, 'matplotlib-style-confmats.yaml'),
               {'xtick.labelsize': 10, 'ytick.labelsize': 10}])
fig = plt.figure(figsize=(3.25, 4))
gs = GridSpec(2, 1, left=0.11, right=0.92, bottom=0.1, top=0.86,
              height_ratios=[1, 19])
# plot confmat
ax = fig.add_subplot(gs[1])
ax = plot_confmat(confmat_tri, ax=ax, norm=LogNorm(vmin=1e-5, vmax=1))

# colorbar
plt.style.use([op.join(paramdir, 'matplotlib-font-myriad.yaml'),
               {'xtick.color': '0.5', 'xtick.major.size': 4}])
cax = fig.add_subplot(gs[0])
cbar = fig.colorbar(ax.images[0], cax=cax, orientation='horizontal')
cbar.outline.set_linewidth(0.2)
cbar.set_label('Pairwise classifiers’ accuracy / error', labelpad=10, size=14)
# scale on top
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
# colorbar ticks
cuts = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
ticks = np.array([np.linspace(a, b, 9, endpoint=False) for a, b in
                 zip(cuts[:-1], cuts[1:])]).flatten().tolist() + [1]
cbar.set_ticks(ticks)
# pretty-format the ticklabels
ticklabs = [l.get_text() for l in cax.get_xticklabels()]
ticklabs = format_colorbar_percent(ticklabs)
cbar.set_ticklabels(ticklabs)
# change ticklabel color (without changing tick color)
_ = [l.set_color('k') for l in cax.get_xticklabels()]
# change tick colors
ticks = cax.get_xticklines()
xy = np.array([t.get_xydata()[0] for t in ticks])
ixs = np.where(xy[:, 1] == 1)[0]
for ix, t in enumerate(ticks):
    if ix in ixs:
        t.set_color('0.75')
        if ixs.tolist().index(ix) % 9 == 0:
            t.set_color('k')

if savefig:
    fig.savefig(op.join('figures', 'manuscript', 'fig-pairwise.pdf'))
else:
    plt.ion()
    plt.show()
