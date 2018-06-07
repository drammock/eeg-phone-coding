#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-pairwise-boxplot.py'
===============================================================================

This script plots the accuracies of the pairwise classifiers.
"""
# @author: drmccloy
# Created on Mon Apr 16 11:10:13 PDT 2018
# License: BSD (3-clause)

import yaml
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


# FLAGS
savefig = True
target = 'manuscript'  # presentation or manuscript

if target == 'presentation':
    figure_paramfile = 'jobtalk-figure-params.yaml'
    outdir = op.join('figures', 'jobtalk')
    plt.style.use('dark_background')
else:
    figure_paramfile = 'manuscript-figure-params.yaml'
    outdir = op.join('figures', 'manuscript')

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data-pairwise'

# figure params
with open(op.join(paramdir, figure_paramfile), 'r') as f:
    figure_params = yaml.load(f)
    col = figure_params['yel']
    bad_color = figure_params['baddatacolor']
    good_color = figure_params['gooddatacolor']
    colordict = figure_params['colordict']
    axislabelsize = figure_params['axislabelsize']
    ticklabelsize = figure_params['ticklabelsize']
    ticklabelcolor = figure_params['ticklabelcolor']
    datacolor = figure_params['datacolor']
    bgcolor = figure_params['bgcolor']

# load EERs
fname = 'eers.tsv'
eers = pd.read_csv(op.join(datadir, fname), sep='\t', index_col=0)
acc = 1 - eers

# plot params
qrtp = dict(color='none', facecolor=bad_color)                # quartile box
whsp = dict(linewidth=0)                                      # whisker
medp = dict(color=bgcolor, linewidth=2)                       # median line
sigp = dict(color=col, linewidth=2)                           # signif. bracket
ptsp = dict(size=2, color=datacolor, alpha=0.3, linewidth=0)  # data pts
boxp = dict(showcaps=False, showfliers=False, boxprops=qrtp, medianprops=medp,
            width=0.4, whiskerprops=whsp)

# init figure
height_ratios = (11, 5)
fig = plt.figure(figsize=(7, 5))
gs = GridSpec(2, 1, left=0.12, right=0.97, bottom=0.06, top=0.97, hspace=0.05,
              height_ratios=height_ratios)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0)

# plot
for ax in (ax0, ax1):
    ax = sns.violinplot(ax=ax, data=acc, cut=0, scale='area', inner=None,
                        linewidth=0.25, color='0.9')
    #ax = sns.boxplot(ax=ax, data=acc, **boxp)
    ax = sns.stripplot(ax=ax, jitter=0.2, data=acc, **ptsp)
    #ax = sns.swarmplot(ax=ax, data=acc, **ptsp)
    sns.despine(ax=ax)

# remove extra axis
sns.despine(ax=ax0, bottom=True)
ax0.xaxis.set_visible(False)

# set proper ylims
ax0.set_ylim(0.78, 1.)
ax1.set_ylim(0.42, 0.52)
ax0.set_yticks([0.8, 0.85, 0.9, 0.95, 1.])
ax1.set_yticks([0.45, 0.5])

# garnish
ax0.set_ylabel('Accuracy', size=axislabelsize, labelpad=12)
ax.tick_params(axis='y', labelsize=ticklabelsize + 2)
ax.tick_params(axis='x', length=0, labelsize=ticklabelsize + 2)

# diagonal cuts
dx = 0.01
dy = 0.15  # how big to make the diagonal lines in axes coordinates
x_left = (-dx, dx)
x = ax0.transAxes.inverted().transform(ax0.transData.transform((10, 0)))[0]
x_data = (x - dx, x + dx)
y_top = (1 - dy/height_ratios[1], 1 + dy/height_ratios[1])
y_bot = (-dy/height_ratios[0], dy/height_ratios[0])
kwargs = dict(color='k', clip_on=False, transform=ax0.transAxes)
ax0.plot(x_left, y_bot, linewidth=0.75, **kwargs)  # top-left
ax0.plot(x_data, y_bot, linewidth=0.5, alpha=0.4, **kwargs)   # top-left
kwargs.update(transform=ax1.transAxes)             # select bottom axes
ax1.plot(x_left, y_top, linewidth=0.75, **kwargs)  # bottom-left
ax1.plot(x_data, y_top, linewidth=0.5, alpha=0.4, **kwargs)   # bottom-left

fig.savefig(op.join(outdir, 'fig-pairwise-boxplot.pdf'))
