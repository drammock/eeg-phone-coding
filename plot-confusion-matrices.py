#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:57:22 2016

@author: drmccloy
"""

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
outdir = 'processed-data'
foreign_langs = np.load(op.join(outdir, 'foreign-langs.npy'))
confmats = dict()
for lang in foreign_langs:
    fname = op.join(outdir, 'confusion-matrix-{}.tsv'.format(lang))
    confmats[lang] = read_csv(fname, sep='\t', index_col=0, encoding='utf-8')

lang_names = dict(hin='Hindi', swh='Swahili', hun='Hungarian', nld='Dutch')

# style setup
colormap = 'viridis'
plt.rc('font', serif='Charis SIL', family='serif', size=10)
plt.rc('axes.spines', top=False, right=False, left=False, bottom=False)
plt.rc('xtick.major', size=2)
plt.rc('ytick.major', size=2)
plt.rc('ytick', right=False)
plt.rc('xtick', top=False)

width = 2 * confmats[foreign_langs[0]].shape[1]
heights = np.array([confmats[lg].shape[0] for lg in foreign_langs])
figsize = np.array([width, heights.sum()]) * 7.5 / width

# initialize figure
fig = plt.figure(figsize=figsize)
axs = ImageGrid(fig, 111, nrows_ncols=(4, 2), axes_pad=(0.4, 0.6),
                label_mode='all')

for ix, lang in enumerate(foreign_langs):
    confmat = confmats[lang]
    confmat_log = -np.log2(confmat)
    ax1 = axs[ix * 2]
    ax2 = axs[ix * 2 + 1]
    for ax, data, cmap in zip([ax1, ax2], [confmat, confmat_log],
                              ['viridis', 'viridis_r']):
        _ = ax.imshow(data, cmap=plt.get_cmap(cmap))
        _ = ax.yaxis.set_ticks(range(confmat.shape[0]))
        _ = ax.xaxis.set_ticks(range(confmat.shape[1]))
        _ = ax.xaxis.set_ticklabels(confmat.columns)
        if ix == len(foreign_langs) - 1:
            _ = ax.set_xlabel('English')
    if ix == 0:
        _ = ax1.set_title('prob')
        _ = ax2.set_title('-log2(prob)')
    _ = ax1.yaxis.set_ticklabels(confmat.index)
    _ = ax1.set_ylabel(lang_names[lang])
if savefig:
    fig.savefig('confusion-matrices.pdf')
else:
    plt.ion()
    plt.show()
