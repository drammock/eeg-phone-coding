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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
datadir = 'processed-data-OVR'

# load params
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    order = analysis_params['canonical_phone_order']['eng']

# load EERs
fname = 'eers.tsv'
eers = pd.read_csv(op.join(datadir, fname), sep='\t', index_col=0)
eers['average'] = eers.mean(axis=1)
sorted_eers = eers.sort_values('average')

# load confmat
fpath = op.join(datadir, 'confusion-matrices',
                'eer-confusion-matrix-nonan-eng-cvalign-dss5-average.tsv')
confmat = pd.read_csv(fpath, sep='\t', index_col=0)
confmat = confmat.loc[order, order]

# init figure
plt.style.use(['dark_background',
               op.join(paramdir, 'matplotlib-style-confmats.yaml'),
               {'xtick.labelsize': 10, 'ytick.labelsize': 10,
                'xtick.color': '0.5', 'ytick.color': '0.5'}])

fig = plt.figure(figsize=(5, 6))
gs = GridSpec(2, 1, left=0.06, right=0.96, bottom=0.08, top=0.88, hspace=0.1,
              height_ratios=[1, 19])

# plot confmat
ax = fig.add_subplot(gs[1])
ax = plot_confmat(confmat, ax=ax, norm=LogNorm(vmin=1e-5, vmax=1),
                  cmap='viridis')

# change tick label color without affecting tick line color
_ = [l.set_color('w') for l in ax.get_xticklabels(which='both')]
_ = [l.set_color('w') for l in ax.get_yticklabels(which='both')]

# colorbar
plt.style.use({'xtick.major.size': 3,
               'xtick.labelsize': 12, 'font.family': 'sans-serif'})
cax = fig.add_subplot(gs[0])
cbar = fig.colorbar(ax.images[0], cax=cax, orientation='horizontal')
# scale on top
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
cax.set_xlabel('Classification probability (OVR classifiers)', labelpad=12,
               size=14)
# ticks
cuts = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
ticks = np.array([np.linspace(a, b, 10, endpoint=False) for a, b in
                  zip(cuts[:-1], cuts[1:])]).flatten().tolist() + [1]
cbar.set_ticks(ticks)
# hack a slight rightward shift of ticklabels by padding with a space
cbar.set_ticklabels([' {}'.format(l.get_text()) for l in
                     cax.get_xticklabels()])
_ = [l.set_color('w') for l in cax.get_xticklabels()]

# frame
cbar.outline.set_linewidth(0)

if savefig:
    fname = 'fig-ovr.pdf'
    fig.savefig(op.join(outdir, fname))
else:
    plt.ion()
    plt.show()
