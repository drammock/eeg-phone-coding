#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-weighted-confusion-matrices.py'
===============================================================================

This script plots feature-based confusion matrices, weights matrices from
EEG-trained classifiers, and the linear combination of the two.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)

from __future__ import division, print_function
from pandas import read_csv
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.ioff()

# flags
savefig = True
plot_feats = True
plot_weights = True
plot_weighted = True
plot_flags = [plot_feats, plot_weights, plot_weighted]

# figure width in inches; height auto-calculated
ncol = sum(plot_flags)
colnames = ['Equal weight to all features',
            'Features weighted by classifier EER',
            'Linear sum (equally weighted)']
colnames = np.array(colnames)[plot_flags]
figwidth = 4 * ncol

# file I/O
figdir = 'figures'
paramdir = 'params'
outdir = 'processed-data'

# style setup
labelsize = 8
colormap = 'viridis'
plt.rc('font', serif='Charis SIL', family='serif', size=10)
plt.rc('axes.spines', top=False, right=False, left=False, bottom=False)
plt.rc('xtick.major', size=10, pad=2, width=0.2)
plt.rc('ytick.major', size=12, pad=2, width=0.2)
plt.rc('xtick.minor', size=0, pad=2)
plt.rc('ytick.minor', size=0, pad=2)
plt.rc('ytick', right=False)
plt.rc('xtick', top=False)

# load list of languages, put English last
foreign_langs = np.load(op.join(paramdir, 'foreign-langs.npy'))
foreign_langs = np.append(foreign_langs[foreign_langs != 'eng'], 'eng')
# pretty names for axis labels
lang_names = dict(hin='Hindi', swh='Swahili', hun='Hungarian', nld='Dutch',
                  eng='English')

# load data
confmats = dict()
weightmats = dict()
weightedmats = dict()
for lang in foreign_langs:
    # load feature-based confusion matrices
    fpath = op.join(outdir, 'features-confusion-matrix-{}.tsv').format(lang)
    confmat = read_csv(fpath, sep='\t', encoding='utf-8', index_col=0)
    # load eeg confusion matrices
    fpath = op.join(outdir, 'eeg-confusion-matrix-{}.tsv'.format(lang))
    weightmat = read_csv(fpath, sep='\t', index_col=0, encoding='utf-8')
    # load weighted confusion matrices
    fpath = op.join(outdir, 'weighted-confusion-matrix-{}.tsv'.format(lang))
    weightedmat = read_csv(fpath, sep='\t', index_col=0, encoding='utf-8')
    # save to global dict
    confmats[lang] = confmat
    weightmats[lang] = weightmat
    weightedmats[lang] = weightedmat

# calculate figure size
matrix_width = 3 * confmat.shape[1]
heights = np.array([confmats[lg].shape[0] for lg in foreign_langs])
figsize = np.array([matrix_width, heights.sum()]) * figwidth / matrix_width

# initialize figure
fig = plt.figure(figsize=figsize)
axs = ImageGrid(fig, 111, nrows_ncols=(len(foreign_langs), ncol),
                axes_pad=0.5, label_mode='all')

# iterate over languages
for ix, lang in enumerate(foreign_langs):
    matrices = list()
    if plot_feats:
        matrices.append(confmats[lang])
    if plot_weights:
        matrices.append(weightmats[lang])
    if plot_weighted:
        matrices.append(weightedmats[lang])
    for cix, data in enumerate(matrices):
        ax = axs[ix * ncol + cix]
        ax.imshow(data, cmap=plt.get_cmap(colormap))
        if not ix:
            ax.set_title(colnames[cix])
        if not cix:
            ax.set_ylabel(lang_names[lang])
        ax.set_xticks(np.arange(data.shape[1])[1::2], minor=False)
        ax.set_xticks(np.arange(data.shape[1])[::2], minor=True)
        ax.set_yticks(np.arange(data.shape[0])[1::2], minor=False)
        ax.set_yticks(np.arange(data.shape[0])[::2], minor=True)
        ax.set_xticklabels(data.columns[1::2], minor=False, size=labelsize)
        ax.set_xticklabels(data.columns[::2], minor=True, size=labelsize)
        ax.set_yticklabels(data.index[1::2], minor=False, size=labelsize)
        ax.set_yticklabels(data.index[::2], minor=True, size=labelsize)
        ax.tick_params(axis='both', color='0.8')
        if ix == len(foreign_langs) - 1:
            ax.set_xlabel('English')
# finish
if savefig:
    plt.savefig(op.join(figdir, 'weighted-confusion-matrices.pdf'))
else:
    plt.ion()
    plt.show()
