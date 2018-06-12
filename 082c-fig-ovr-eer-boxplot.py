#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-ovr.py'
===============================================================================

This script plots error rates from OVR consonant classifiers.  It
addresses the phoneme-level question: which consonants are best-discernable
based on the neural data?
"""
# @author: drmccloy
# Created on Fri Jan 19 14:26:44 PST 2018
# License: BSD (3-clause)

import yaml
import numpy as np
import pandas as pd
import os.path as op
import seaborn as sns
import matplotlib.pyplot as plt

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
datadir = 'processed-data-OVR'

# figure params
with open(op.join(paramdir, figure_paramfile), 'r') as f:
    figure_params = yaml.load(f)
    boxplot_color = figure_params['boxplot']
    median_color = figure_params['median']
    swarm_color = figure_params['swarm']
    axislabelsize = figure_params['axislabelsize']
    ticklabelsize = figure_params['ticklabelsize']

# load EERs
fname = 'eers.tsv'
eers = pd.read_csv(op.join(datadir, fname), sep='\t', index_col=0)
acc = 1 - eers

# plot params
qrtp = dict(color='none', facecolor=boxplot_color)            # quartile box
whsp = dict(linewidth=0)                                      # whisker
medp = dict(color=median_color, linewidth=2)                  # median line
ptsp = dict(size=3, color=swarm_color, linewidth=0)           # data pts
boxp = dict(showcaps=False, showfliers=False, boxprops=qrtp, medianprops=medp,
            width=0.4, whiskerprops=whsp)

# init figure
plt.style.use(op.join(paramdir, 'matplotlib-font-myriad.yaml'))
fig, ax = plt.subplots(figsize=(6.75, 3))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.95)

# plot
ax = sns.boxplot(ax=ax, data=acc, **boxp)
ax = sns.swarmplot(ax=ax, dodge=True, data=acc, **ptsp)
#ax = sns.stripplot(ax=ax, jitter=0.15, data=acc, **ptsp)
sns.despine(ax=ax)

# garnish
ax.set_ylabel('1 − equal error rate', size=axislabelsize, labelpad=12)
ax.set_xlabel('Subject code')
ax.set_ylim(-0.01, 1.01)
ax.tick_params(axis='y', labelsize=ticklabelsize + 2)
ax.tick_params(axis='x', length=0, labelsize=ticklabelsize + 2)

if savefig:
    fig.savefig(op.join(outdir, 'fig-ovr-boxplot.pdf'))
else:
    plt.ion()
    plt.show()
