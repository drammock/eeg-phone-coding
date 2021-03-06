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

# FLAGS
plt.ioff()

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    feature_systems = analysis_params['feature_systems']
    use_ordered = analysis_params['sort_matrices']
    methods = analysis_params['methods']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    legend_names = analysis_params['pretty_legend_names']
    scheme = analysis_params['classification_scheme']
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

# FILE NAMING VARIABLES
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''

phone_level = scheme in ['pairwise', 'OVR', 'multinomial']

# BASIC FILE I/O
datadir = f'processed-data-{scheme}{trunc}'
indir = op.join(datadir, 'matrix-correlations')
outdir = op.join('figures', 'matrix-correlations')
if not op.isdir(outdir):
    mkdir(outdir)

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

# loop over methods (phone-level, feature-level-eer, uniform-error-simulations)
order_types = ('row-', 'col-', 'feat-') if use_ordered else ('',)
for order_type in order_types:
    if phone_level:
        if order_type == 'row-':
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            axs = [ax]
            args = (order_type, ordered, sfn, 'eer')
            fname = '{}{}matrix-diagonality-{}-{}.tsv'.format(*args)
            colname_dict = dict(pairwise='Pairwise',
                                OVR='One-vs-rest',
                                multinomial='Multinomial')
            for sch in ['pairwise', 'OVR', 'multinomial']:
                this_indir = op.join('processed-data-{}'.format(sch),
                                     'matrix-correlations')
                df = pd.read_csv(op.join(this_indir, fname), sep='\t',
                                 index_col=0)
                df.rename(columns=colname_dict, inplace=True)
                df.plot(x=df.index, ax=ax, legend=False)
            ax.set_title('Phone-level logistic classifiers')
            # xticks = ax.xaxis.get_ticklocs()
            # ax.set_xticks(np.linspace(xticks[0], xticks[-1], len(df.index)))
            ax.set_xticks(np.arange(len(df.index)))
            ax.set_xticklabels(df.index)
            ax.set_xlabel('subject code')
            ax.set_ylabel('matrix diagonality')
            ax.set_ylim(-0.2, 1)
        else:
            continue
    else:
        # init figure
        fig, axs = plt.subplots(3, 1, figsize=(6, 12))
        for ax, method in zip(axs, methods[::-1]):
            args = (order_type, ordered, sfn, method)
            fname = '{}{}matrix-diagonality-{}-{}.tsv'.format(*args)
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
                ax.set_xticks(np.linspace(xticks[0], xticks[-1],
                              len(df.index)))
                ax.set_xticklabels(df.index)
            ax.set_xlabel(xlabels[method])
            ax.set_ylabel('matrix diagonality')
            ax.set_ylim(-0.2, 1)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    # legend
    bbox = axs[0].get_position()
    new_xmax = bbox.xmin + 0.75 * (bbox.xmax - bbox.xmin)
    new_bbox = Bbox(np.array([[bbox.xmin, bbox.ymin], [new_xmax, bbox.ymax]]))
    axs[0].set_position(new_bbox)
    axs[0].legend(bbox_to_anchor=(1.07, 1.), loc=2, borderaxespad=0.)
    args = (order_type, ordered, sfn, scheme, trunc)
    out_fname = '{}{}matrix-correlations-{}-{}{}.pdf'.format(*args)
    fig.savefig(op.join(outdir, out_fname))
