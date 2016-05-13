#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-feature-based-confusion-matrices.py'
===============================================================================

This script plots confusion matrices.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import numpy as np
from os import path as op
from pandas import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.ioff()

# flags
savefig = True
negative_log = True
figsize = (4, 24)

# file I/O
figdir = 'figures'
paramdir = 'params'
outdir = 'processed-data'

# load list of languages
foreign_langs = np.load(op.join(paramdir, 'foreign-langs.npy'))
lang_names = dict(hin='Hindi', swh='Swahili', hun='Hungarian', nld='Dutch',
                  eng='English')

# set up plot
labelsize = 8
colormap = 'viridis_r' if negative_log else 'viridis'
plt.rc('font', serif='Charis SIL', family='serif', size=12)
plt.rc('axes.spines', top=False, right=False, left=False, bottom=False)
plt.rc('xtick.major', size=10, pad=2, width=0.2)
plt.rc('ytick.major', size=12, pad=2, width=0.2)
plt.rc('xtick.minor', size=0, pad=2)
plt.rc('ytick.minor', size=0, pad=2)
plt.rc('ytick', right=False)
plt.rc('xtick', top=False)
# initialize figure
fig = plt.figure(figsize=figsize)
axs = ImageGrid(fig, 111, nrows_ncols=(len(foreign_langs), 1),
                axes_pad=0.5, label_mode='all')

# iterate over languages
for ix, lang in enumerate(foreign_langs):
    fpath = op.join(outdir, 'features-confusion-matrix-{}.tsv').format(lang)
    featmatch = read_csv(fpath, sep='\t', encoding='utf-8', index_col=0)
    confmat = -np.log2(featmatch.T) if negative_log else featmatch.T
    ax = axs[ix]
    ax.imshow(confmat, cmap=plt.get_cmap(colormap))
    ax.set_ylabel(lang_names[lang])
    ax.set_xticks(np.arange(confmat.shape[1])[1::2], minor=False)
    ax.set_xticks(np.arange(confmat.shape[1])[::2], minor=True)
    ax.set_yticks(np.arange(confmat.shape[0])[1::2], minor=False)
    ax.set_yticks(np.arange(confmat.shape[0])[::2], minor=True)
    ax.set_xticklabels(confmat.columns[1::2], minor=False, size=labelsize)
    ax.set_xticklabels(confmat.columns[::2], minor=True, size=labelsize)
    ax.set_yticklabels(confmat.index[1::2], minor=False, size=labelsize)
    ax.set_yticklabels(confmat.index[::2], minor=True, size=labelsize)
    ax.tick_params(axis='both', color='0.8')
    if ix == len(foreign_langs) - 1:
        ax.set_xlabel('English')
# finish
if savefig:
    plt.savefig(op.join(figdir, 'feature-confusion-matrices.pdf'))
else:
    plt.ion()
    plt.show()
