#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-diagonality.py'
===============================================================================

This script plots the diagonality of the confusion matrices.
"""
# @author: drmccloy
# Created on Thu Aug 31 10:21:05 PDT 2017
# License: BSD (3-clause)

import yaml
from os import mkdir
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

np.set_printoptions(precision=6, linewidth=130)
pd.set_option('display.width', 130)
plt.ioff()

# BASIC FILE I/O
paramdir = 'params'
indir = op.join('processed-data', 'matrix-correlations')
outdir = op.join('figures', 'matrix-correlations')
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS FROM YAML
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    feature_systems = analysis_params['feature_systems']
    use_ordered = analysis_params['sort_matrices']
    methods = analysis_params['methods']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    legend_names = analysis_params['pretty_legend_names']
del analysis_params

# file naming variables
ordered = 'ordered-' if use_ordered else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# load plot style
plt.style.use(op.join(paramdir, 'matplotlib-style-lineplots.yaml'))

# pretty titles
titles = dict(phone='Empirical phone-level accuracy',
              eer='Empirical feature-level accuracy',
              theoretical='Simulated uniform accuracy across features')
xlabels = dict(phone='subject code',
               eer='subject code',
               theoretical='accuracy')
plotting_order = ['phoible_redux', 'phoible_sparse', 'jfh_dense', 'spe_dense',
                  'spe_sparse', 'jfh_sparse']
plotting_order = [legend_names[name] for name in plotting_order]

# init figure
fig, axs = plt.subplots(3, 1, figsize=(6, 12))

# loop over methods (phone-level, feature-level-eer, uniform-error-simulations)
for ax, method in zip(axs, methods[::-1]):
    fname = '{}matrix-diagonality-{}-{}.tsv'.format(ordered, sfn, method)
    df = pd.read_csv(op.join(indir, fname), sep='\t', index_col=0)
    df.rename(columns=legend_names, inplace=True)
    # sorting
    df = df[plotting_order]
    if ax == axs[0]:
        df = df.iloc[:0:-1]  # reverse row order & omit 0.999
    # plot
    df.plot(x=df.index, ax=ax, title=titles[method], legend=False)
    # garnish
    if ax == axs[0]:
        ax.set_xticks(df.index)
        ax.set_xticklabels(df.index)
    else:
        xticks = ax.xaxis.get_ticklocs()
        ax.set_xticks(np.linspace(xticks[0], xticks[-1], len(df.index)))
        ax.set_xticklabels(df.index)
    ax.set_title(titles[method])
    ax.set_xlabel(xlabels[method])
    ax.set_ylabel('matrix diagonality')
    ylim = (-1., 1.) if method == 'eer' else (-0.5, 1.)
    ax.set_ylim(*ylim)

fig.tight_layout()
fig.subplots_adjust(hspace=0.4)
# legend
bbox = axs[0].get_position()
new_xmax = bbox.xmin + 0.7 * (bbox.xmax - bbox.xmin)
new_bbox = Bbox(np.array([[bbox.xmin, bbox.ymin], [new_xmax, bbox.ymax]]))
axs[0].set_position(new_bbox)
axs[0].legend(bbox_to_anchor=(1.07, 1.), loc=2, borderaxespad=0.)
out_fname = '{}matrix-correlations-{}.pdf'.format(ordered, sfn)
fig.savefig(op.join(outdir, out_fname))
