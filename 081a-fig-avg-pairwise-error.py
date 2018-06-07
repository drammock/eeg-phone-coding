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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from aux_functions import plot_confmat


# FLAGS
savefig = True
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 30)
np.set_printoptions(linewidth=240)

# BASIC FILE I/O
paramdir = 'params'
outdir = op.join('figures', 'jobtalk')
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

# lower tri
'''
np.fill_diagonal(confmat.values, 0.)
'''
confmat_tri = pd.DataFrame(np.tril(confmat), index=confmat.index,
                           columns=confmat.columns)

# style setup
plt.style.use(['dark_background',
               op.join(paramdir, 'matplotlib-style-confmats.yaml'),
               {'xtick.labelsize': 11, 'ytick.labelsize': 11,
                'xtick.color': '0.5', 'ytick.color': '0.5'}])

# init figure
fig = plt.figure(figsize=(5, 6))
gs = GridSpec(2, 1, left=0.07, right=0.94, bottom=0.08, top=0.88, hspace=0.1,
              height_ratios=[1, 19])

# plot confmat
ax = fig.add_subplot(gs[1])
ax = plot_confmat(confmat_tri, ax=ax, norm=LogNorm(vmin=1e-5, vmax=1),
                  cmap='viridis')

'''
# delete extra ticklabels at ends
yticks = ax.get_yticklabels('minor')
yticks[0] = ''
ax.set_yticklabels(yticks, minor=True)
xticks = ax.get_xticklabels('minor')
xticks[-1] = ''
ax.set_xticklabels(xticks, minor=True)
'''

# change tick label color without affecting tick line color
ticklabelcolor = 'w'
_ = [l.set_color(ticklabelcolor) for l in ax.get_xticklabels(which='both')]
_ = [l.set_color(ticklabelcolor) for l in ax.get_yticklabels(which='both')]

# colorbar
plt.style.use({'xtick.color': ticklabelcolor, 'xtick.major.size': 3,
               'xtick.labelsize': 12, 'font.family': 'sans-serif'})
cax = fig.add_subplot(gs[0])
cbar = fig.colorbar(ax.images[0], cax=cax, orientation='horizontal')
# scale on top
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
cax.set_xlabel('Accuracy / error (pairwise classifiers)',
               labelpad=12, size=14)
# ticks
cuts = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
ticks = np.array([np.linspace(a, b, 10, endpoint=False) for a, b in
                 zip(cuts[:-1], cuts[1:])]).flatten().tolist() + [1]
cbar.set_ticks(ticks)
# pretty-format the ticklabels
ticklabs = [l.get_text() for l in cax.get_xticklabels()]
for ix, lab in enumerate(ticklabs):
    if lab == '':
        continue
    lab = (lab[14:-2].replace('{', '').replace('}', '')
           .replace('10', '1').replace('^', 'e'))
    lab = 100. * float(lab)
    lab = int(lab) if lab > 0.5 else lab
    ticklabs[ix] = '{}%'.format(lab)
# hack a slight rightward shift of ticklabels by padding with a space
cbar.set_ticklabels([' {}'.format(l) for l in ticklabs])

_ = [l.set_color(ticklabelcolor) for l in cax.get_xticklabels()]
# frame
cbar.outline.set_linewidth(0)

if savefig:
    fname = 'fig-pairwise.pdf'
    fig.savefig(op.join(outdir, fname))

# highlight boxes
cell_zero = ('ɡ', 's')
cell_one = ('ɡ', 'ɡ')
for count, cell in enumerate((cell_zero, cell_one)):
    ix = (confmat_tri.index.get_loc(cell[0]),
          confmat_tri.columns.get_loc(cell[1]))
    xy = np.array(ix) - 0.5
    rect = Rectangle(xy, width=1, height=1, facecolor='none',
                     edgecolor='r', linewidth=2)
    ax.add_artist(rect)
    fname = 'fig-pairwise-highlight-{}.pdf'.format(count)
    fig.savefig(op.join(outdir, fname))
    rect.remove()

'''
# annotations
n_annot = 2
# find indices of annotations in tri matrix
confusion_values = confmat_tri.values.flatten()
nonzero_ixs = np.where(confusion_values > 0)[0]
argsort = confusion_values.argsort()
nonzero_argsort = argsort[np.in1d(argsort, nonzero_ixs)]
highest = np.unravel_index(nonzero_argsort[-n_annot:], confmat_tri.shape)

# make annotation labels
labels = ['-'.join([x.split('_')[0], x.split('_')[-1]]) for x in
          sorted_eers.tail(n_annot).index]
offsets = list(zip(*highest))

# add transparent overlay to colorbar to get coord. sys.
cbar2 = fig.add_axes(cbar.ax.figbox)
cbar2.axis('off')
cbar2.set_xscale('log')
cbar2.set_xlim(cbar.get_clim())

# transforms
t1 = cbar2.transData.inverted()
t2 = ax.transData
y = [cax.get_ylim()[0]] * n_annot  # * 2 if also doing lowest
x = sorted_eers['average'].tail(n_annot).tolist()
cbar_xy = list(zip(x, y))

# annotate
plt.style.use({'font.family': 'serif'})
ap = dict(arrowstyle='-', color='0.6')
for label, xy, cxy in zip(labels, offsets, cbar_xy):
    ax.annotate(label, xy=xy[::-1], xytext=xy, alpha=0, textcoords='data',
                arrowprops=ap)

# do secondary arrows individually, to control curves
cbar2.annotate(labels[0], xy=cbar_xy[0], textcoords='data', arrowprops=ap,
               xytext=t1.transform(t2.transform(offsets[0])))
ap.update(dict(connectionstyle='angle,angleA=0,angleB=90,rad=20'))
cbar2.annotate(labels[1], xy=cbar_xy[1], arrowprops=ap,
               xytext=t1.transform(t2.transform(offsets[1])))

if savefig:
    fig.savefig(op.join(outdir, 'fig-pairwise-annotated.pdf'))
else:
    plt.ion()
    plt.show()
'''
if not savefig:
    plt.ion()
    plt.show()
