#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'rank-classifier-performance.py'
===============================================================================

This script generates sorted EER tables and generates plots comparing
classifier performance for each feature system.
"""
# @author: drmccloy
# Created on Fri Sep 22 15:13:27 PDT 2017
# License: BSD (3-clause)

import yaml
import numpy as np
import pandas as pd
import os.path as op
from os import makedirs
import matplotlib.pyplot as plt


def plot_eers(df, ax, legend=False, title=''):
    x = np.tile(np.arange(df.shape[0]), (df.shape[1], 1))
    x = x + 0.4 * (0.5 - np.random.rand(*x.shape))
    lines = ax.plot(x.T, df, 'o', markersize=4, alpha=0.6, linewidth=0.5)
    mean_line = ax.plot(np.arange(df.shape[0]), df.mean(axis=1), 'k',
                        alpha=0.2, linewidth=2)
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(df.shape[0]))
    if legend:
        handles = lines + mean_line
        labels = df.columns.tolist() + ['mean']
        ax.legend(handles, labels, loc='upper left',
                  bbox_to_anchor=(1.1, 1.0))
    return ax


# FLAGS
svm = False
savefig = True

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data' if svm else 'processed-data-logistic'
rankdir = op.join(datadir, 'feature-rankings')
makedirs(rankdir, exist_ok=True)

# LOAD PARAMS FROM YAML
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    feature_systems = analysis_params['feature_systems']
    feature_mappings = analysis_params['feature_mappings']
    pretty_featsys_names = analysis_params['pretty_featsys_names']
    methods = analysis_params['methods']
    skip = analysis_params['skip']
del analysis_params

# load EERs
eers = pd.read_csv(op.join(datadir, 'eers.tsv'), sep='\t', index_col=0)
valid_subjs = [s for s in subjects if s not in skip]
order = eers.mean(axis=1).sort_values().index.tolist()
sorted_eers = eers.loc[order, valid_subjs]

# plot setup
plt.ioff()
plt.style.use(op.join(paramdir, 'matplotlib-style-lineplots.yaml'))
fig = plt.figure(figsize=(12, 18))
plot_order = ['jfh_sparse', 'jfh_dense', 'spe_sparse', 'spe_dense',
              'phoible_redux', 'phoible_sparse']

# one plot with all features
ax = plt.subplot2grid((4, 6), (0, 0), colspan=5)
plot_eers(sorted_eers, ax, legend=True, title='All features')
ax.set_ylabel('Equal error rate')
ax.set_xticklabels(sorted_eers.index, rotation=90)

# one blank plot; leaves room for legend
ax = plt.subplot2grid((4, 6), (0, 5))
ax.axis('off')

for feat_sys, features in feature_systems.items():
    # subset
    this_feat_sys = eers.loc[features]
    # write out dataframe with feature names and EERs sorted within-subject
    df = pd.DataFrame()
    for subj_code in valid_subjs:
        this_subj = this_feat_sys[subj_code].copy()
        this_subj.sort_values(inplace=True)
        df[(subj_code, 'feature')] = this_subj.index.values
        df[(subj_code, 'eer')] = this_subj.values
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    fname = op.join(rankdir, '{}-feature-rankings.tsv'.format(feat_sys))
    sysname = pretty_featsys_names[feat_sys].replace('\\n', ' ')
    with open(fname, 'w') as f:
        f.write('# ranking of classifier performance (EER=equal error rate)\n')
        f.write('# {}\n\n'.format(sysname))
        df.to_csv(f, sep='\t', index=False)

    # plot EER by feature/subject
    row_order = this_feat_sys.mean(axis=1).sort_values().index.tolist()
    this_feat_sys = this_feat_sys.loc[row_order, valid_subjs]
    title = pretty_featsys_names[feat_sys].replace('\\n', ' ')
    ix = plot_order.index(feat_sys)
    row = (ix // 2) + 1
    col = (ix % 2) * 3
    ax = plt.subplot2grid((4, 6), (row, col), colspan=3)
    plot_eers(this_feat_sys, ax, legend=False, title=title)
    if col == 0:
        ax.set_ylabel('Equal error rate')
    xlabels = [feature_mappings[feat_sys][feat]
               for feat in this_feat_sys.index.tolist()]
    ax.set_xticklabels(xlabels, rotation=90)

fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95,
                    hspace=0.8, wspace=0.3)

if savefig:
    fname = 'eer-by-feat-sys.pdf' if svm else 'eer-by-feat-sys-logistic.pdf'
    fig.savefig(op.join('figures', fname))
else:
    plt.ion()
    plt.show()
