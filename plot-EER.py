#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-EER.py'
===============================================================================

This script plots equal error rate as a grouped barplot.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)


from __future__ import division, print_function
import numpy as np
import os.path as op
from pandas import read_csv
from expyfun.analyze import barplot
import matplotlib.pyplot as plt
plt.ioff()

# flags
savefig = True

# file I/O
figdir = 'figures'
paramdir = 'params'
outdir = 'processed-data'

# styles
colors = [str(x) for x in np.linspace(0.2, 0.8, 5)]
plt.rc('font', family='sans-serif', size=8, **{'sans-serif': 'M+ 1c'})

# load data
eer = read_csv(op.join(outdir, 'equal-error-rates.tsv'), sep='\t', index_col=0)
# sort English first
eer = eer[['eng', 'hun', 'nld', 'swh', 'hin']]
# axis labels
bars = eer.values.ravel()
groups = np.arange(bars.size).reshape(-1, eer.columns.size)
group_names = dict(consonantal='conson.', sonorant='sonor.',
                   continuant='contin.',
                   delayedRelease='del. rel.', approximant='approx.',
                   nasal='nasal', lateral='lateral', labial='labial',
                   labiodental='labiodent.', coronal='coronal',
                   anterior='anterior', distributed='distrib.',
                   strident='strid.', dorsal='dorsal',
                   spreadGlottis='aspir.', periodicGlottalSource='voiced')
gn = [group_names[x] for x in eer.index]
# bar_names = dict(hin='H', swh='S', hun='U', nld='D', eng='E')
# bn = [bar_names[x] for x in eer.columns] * eer.shape[0]

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax, bar = barplot(bars, groups=groups, ax=ax, group_names=gn, bar_names=' ',
                  ylim=(0, 0.55), gap_size=0.6, bar_kwargs=dict(color=colors))
ax.axhline(0.5, color='k', linestyle=(0, (5, 7)), linewidth=1)
ax.set_ylabel('Equal error rate')
ax.legend(bar.get_children()[:5], (eer.columns), loc='upper right')
plt.tight_layout()
if savefig:
    plt.savefig(op.join(figdir, 'eer-barplot.pdf'))
else:
    plt.ion()
    plt.show()
