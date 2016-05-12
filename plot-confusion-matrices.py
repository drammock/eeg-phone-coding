#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-confusion-matrices.py'
===============================================================================

This script plots confusion matrices for English vs foreign speech sounds for
four different languages. The confusion matrices are based on distinctive
feature classifiers trained on EEG responses to the speech sounds.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)


from __future__ import division, print_function
import numpy as np
import os.path as op
from pandas import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.ioff()

# flags
savefig = True

# file I/O
figdir = 'figures'
paramdir = 'params'
outdir = 'processed-data'

foreign_langs = np.load(op.join(paramdir, 'foreign-langs.npy'))
confmats = dict()
for lang in foreign_langs:
    fname = op.join(outdir, 'eeg-confusion-matrix-{}.tsv'.format(lang))
    confmats[lang] = read_csv(fname, sep='\t', index_col=0, encoding='utf-8')

lang_names = dict(hin='Hindi', swh='Swahili', hun='Hungarian', nld='Dutch',
                  eng='English')

# style setup
labelsize = 8
colormap = 'viridis'
plt.rc('font', serif='Charis SIL', family='serif', size=10)
plt.rc('axes.spines', top=False, right=False, left=False, bottom=False)
plt.rc('xtick.major', size=10, pad=2, width=0.25)
plt.rc('xtick.minor', size=0, pad=2)
plt.rc('ytick.major', size=12, pad=2, width=0.25)
plt.rc('ytick.minor', size=0, pad=2)
plt.rc('ytick', right=False)
plt.rc('xtick', top=False)

width = 2 * confmats[foreign_langs[0]].shape[1]
heights = np.array([confmats[lg].shape[0] for lg in foreign_langs])
figsize = np.array([width, heights.sum()]) * 7.5 / width

# initialize figure
fig = plt.figure(figsize=figsize)
axs = ImageGrid(fig, 111, nrows_ncols=(len(foreign_langs), 2),
                axes_pad=(0.4, 0.6), label_mode='all')

for ix, lang in enumerate(foreign_langs):
    confmat = confmats[lang]
    confmat_log = -np.log2(confmat)
    ax1 = axs[ix * 2]
    ax2 = axs[ix * 2 + 1]
    for ax, data, cmap in zip([ax1, ax2], [confmat, confmat_log],
                              ['viridis', 'viridis_r']):
        _ = ax.imshow(data, cmap=plt.get_cmap(cmap))
        ax.set_xticks(np.arange(confmat.shape[1])[1::2], minor=False)
        ax.set_xticks(np.arange(confmat.shape[1])[::2], minor=True)
        ax.set_xticklabels(confmat.columns[1::2], minor=False, size=labelsize)
        ax.set_xticklabels(confmat.columns[::2], minor=True, size=labelsize)
        ax.set_yticks(np.arange(confmat.shape[0])[1::2], minor=False)
        ax.set_yticks(np.arange(confmat.shape[0])[::2], minor=True)
        ax.set_yticklabels(confmat.index[1::2], minor=False, size=labelsize)
        ax.set_yticklabels(confmat.index[::2], minor=True, size=labelsize)
        if ix == len(foreign_langs) - 1:
            ax.set_xlabel('English')
    if ix == 0:
        ax1.set_title('prob')
        ax2.set_title('-log2(prob)')
    ax1.set_ylabel(lang_names[lang])
if savefig:
    fig.savefig(op.join(figdir, 'eeg-confusion-matrices.pdf'))
else:
    plt.ion()
    plt.show()
