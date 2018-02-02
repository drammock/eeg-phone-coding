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
from aux_functions import plot_confmat


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
np.fill_diagonal(confmat.values, 0.)
order = confmat.sum(axis=1).sort_values().index

# lower tri
confmat_tri = pd.DataFrame(np.tril(confmat), index=confmat.index,
                           columns=confmat.columns)
# plot confmat
plt.style.use(op.join(paramdir, 'matplotlib-style-confmats.yaml'))
plt.style.use({'xtick.labelsize': 10, 'ytick.labelsize': 10})
fig, ax = plt.subplots(figsize=(4, 3.5))
ax = plot_confmat(confmat_tri, ax=ax, norm=LogNorm(vmin=1e-2), cmap='plasma')
ax.set_title('Pairwise classifiers’ equal error rates\n'
             '(average across subjects)')

# delete extra ticklabels at ends
yticks = ax.get_yticklabels('minor')
yticks[0] = ''
ax.set_yticklabels(yticks, minor=True)
xticks = ax.get_xticklabels('minor')
xticks[-1] = ''
ax.set_xticklabels(xticks, minor=True)

# colorbar
plt.style.use({'ytick.color': 'k', 'ytick.major.size': 4})
cbar = fig.colorbar(ax.images[0])
ticks = np.linspace(0.01, 0.1, 10)
labels = ['0.01', '', '', '', '0.05', '', '', '', '', '0.1']
cbar.set_ticks(ticks)
cbar.set_ticklabels(labels)
cbar.outline.set_linewidth(0.2)
cbar.set_label('Confusion probability', rotation=270, labelpad=16)

# margins
fig.subplots_adjust(left=0.04, right=0.94, bottom=0.1, top=0.85)

# annotate colorbar
plt.style.use({'ytick.labelsize': 6, 'ytick.major.size': 2})
bbox = cbar.ax.figbox
cbar2 = fig.add_axes(bbox)
cbar2.xaxis.set_visible(False)
cbar2.set_yscale('log')
cbar2.set_ylim(cbar.get_clim()[::-1])
n_annot = 3
ticks = sorted_eers['average'].tail(n_annot)
labels = [' ~ '.join([x.split('_')[0], x.split('_')[-1]]) for x in
          sorted_eers.tail(n_annot).index]
cbar2.set_yticks(ticks)
cbar2.set_yticklabels(labels)
cbar2.invert_yaxis()
cbar2.patch.set_alpha(0)
for s in ['right', 'top', 'bottom']:
    cbar2.spines[s].set_visible(False)

'''
# initial sort eers
df = pd.DataFrame()
for phone in order:
    bools = (eers[['one', 'two']] == phone).any(axis=1)
    labels = bools.loc[bools].index
    labels = labels[np.logical_not(np.in1d(labels, df.index))]
    temp = eers.copy()
    temp['phone'] = phone
    temp['other'] = temp['one'].where(temp['phone'] != temp['one'],
                                      temp['two'], axis=0)
    df = pd.concat([df, temp.loc[labels, :]], axis=0)
# sort within grouping
df = df[['average', 'phone', 'other']]
df = df.groupby('phone', sort=False).apply(lambda x: x.sort_values('average'))
df = df.reset_index(drop=True).set_index(['phone', 'other'])
'''
'''
from sklearn.manifold import MDS
dissim = 1 - confmat.copy()
np.fill_diagonal(dissim.values, 0.)
plt.ion()
fig, axs = plt.subplots(1, 3)
for ax in axs:
    foo = MDS(n_components=2, metric=True, n_init=10, n_jobs=4, eps=1e-9,
              max_iter=1000, dissimilarity='precomputed')
    pos = foo.fit_transform(dissim)
    mds_df = pd.DataFrame(pos, index=confmat.index, columns=['x', 'y'])
    ax = mds_df.plot('x', 'y', ax=ax, kind='scatter', c='w')
    for label, row in mds_df.iterrows():
        ax.annotate(s=label, xy=row)
raise RuntimeError
'''
'''
# plot setup
plt.ioff()
plt.style.use(op.join(paramdir, 'matplotlib-style-lineplots.yaml'))

title = 'Pairwise logistic regression classifiers'
ylim = (0., 0.12)
ylab = 'Equal error rate'
xrot = 0
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=7)
fig, ax = plt.subplots(figsize=(16, 4))
plot_eers(df[['average']], ax, legend=False, title=title,
          ylim=ylim, legend_bbox=(0.98, 1.), marker='o',
          markersize=3, linewidth=0)
ax.set_ylabel(ylab)
ax.set_xticklabels(df.reset_index('other')['other'], rotation=xrot)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.9,
                    hspace=0., wspace=0.)
'''
if savefig:
    fname = 'fig-avg-pairwise-error.pdf'
    fig.savefig(op.join('figures', 'publication', fname))
else:
    plt.ion()
    plt.show()
