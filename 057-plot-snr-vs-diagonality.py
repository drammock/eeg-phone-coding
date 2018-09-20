#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-snr-vs-diagonality.py'
===============================================================================

This script plots matrix diagonality vs. SNR of the EEG recordings.
"""
# @author: drmccloy
# Created on Fri Sep  1 10:44:57 PDT 2017
# License: BSD (3-clause)

import yaml
import numpy as np
import pandas as pd
import os.path as op
from os import mkdir
import matplotlib.pyplot as plt
from scipy.stats import linregress

# FLAGS
plt.ioff()

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    methods = analysis_params['methods']
    feature_systems = analysis_params['feature_systems']
    use_ordered = analysis_params['sort_matrices']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    legend_names = analysis_params['pretty_legend_names']
    scheme = analysis_params['classification_scheme']
    truncate = analysis_params['eeg']['truncate']
del analysis_params

# FILE NAMING VARIABLES
trunc = '-truncated' if truncate else ''
ordered = 'ordered-' if use_ordered else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# BASIC FILE I/O
indir = f'processed-data-{scheme}{trunc}'
outdir = op.join('figures', 'snr-vs-diagonality')
if not op.isdir(outdir):
    mkdir(outdir)

# load SNR data
fname = op.join('processed-data-logistic', 'blinks-epochs-snr.tsv')
snr = pd.read_csv(fname, sep='\t', index_col=0)
snr = snr['snr']  # ignore other columns (n_trials, n_blinks, retained_epochs)

# ignore simulation data
_ = methods.pop(methods.index('theoretical'))
if scheme == 'pairwise':
    _ = methods.pop(methods.index('phone'))

# load plot style
plt.style.use(op.join(paramdir, 'matplotlib-style-lineplots.yaml'))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# pretty titles
colnames = dict(snr='SNR: 10×$\log_{10}$(evoked power / baseline power)',
                diagonality='Matrix row/column correlation')
titles = dict(phone='Phone-level', eer='Feature-level')

for ordering in ['row-', 'col-', 'feat-']:
    if scheme in ('pairwise', 'multinomial') and ordering != 'row-':
        continue
    ncol = 1 if scheme == 'pairwise' else len(feature_systems)
    nrow = 1 if scheme == 'pairwise' else len(methods)
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.5 * ncol + 3, 4.5 * nrow))
    for ax_row, method in zip(np.atleast_2d(axs), methods):
        fname = '{}{}matrix-diagonality-{}-{}.tsv'.format(ordering, ordered,
                                                          sfn, method)
        fpath = op.join(indir, 'matrix-correlations', fname)
        diag_df = pd.read_csv(fpath, sep='\t', index_col=0)
        cols = ['pairwise'] if scheme == 'pairwise' else feature_systems
        for ax, col in zip(np.atleast_1d(ax_row), cols):
            diagonality = diag_df[col]
            diagonality.name = 'diagonality'
            # merge
            df = pd.concat((snr, diagonality), axis=1, join='inner')
            df.sort_values(by='snr')
            df.rename(columns=colnames, inplace=True)
            # plot
            title = ('pairwise logistic' if scheme == 'pairwise'
                     else '{}: {}'.format(titles[method], legend_names[col]))
            # do the scatterplot just to get plot dimensions and garnishes
            df.plot.scatter(x=colnames['snr'], y=colnames['diagonality'],
                            ax=ax, title=title, legend=False, color='w')
            # recycle colors if there aren't enough
            while len(colors) < df.shape[0]:
                colors = colors * 2
            # now add the text
            new_order = ([colnames[x] for x in ['snr', 'diagonality']] +
                         ['subj'])
            row_tuples = df.reset_index()[new_order].iterrows()
            for row, c in zip(row_tuples, colors):
                ax.text(*row[1], ha='center', va='center', color=c,
                        weight='bold')
            # plot lims
            ylim = np.array([diag_df.values.min(), diag_df.values.max()])
            ylim = (np.sign(ylim) * np.ceil(np.abs(10 * ylim) +
                    np.sign(ylim) * np.array([-1, 1])) / 10.)
            ax.set_ylim(*ylim)

    if scheme == 'pairwise':
        fig.subplots_adjust(left=0.125, right=0.95)
    else:
        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.8, hspace=0.4)
    fig.suptitle('SNR vs. matrix diagonality (empirical accuracies)')
    args = [ordering, ordered, sfn, method, scheme, trunc]
    fname = 'snr-vs-matrix-diagonality-{}{}{}-{}-{}{}.pdf'.format(*args)
    fig.savefig(op.join(outdir, fname))

# supplementary figure
feature_systems = ['jfh_sparse', 'spe_sparse', 'phoible_redux']
plot_titles = dict(phoible_redux='PHOIBLE', jfh_sparse='PSA', spe_sparse='SPE')
plt.rc('font', family='serif', serif='Linux Libertine O')
plt.rc('mathtext', fontset='custom', rm='Linux Libertine O',
       it='Linux Libertine O:italic', bf='Linux Libertine O:bold')
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
for ax, featsys in zip(axs, feature_systems):
    diagonality = diag_df[featsys]
    diagonality.name = 'diagonality'
    # merge
    df = pd.concat((snr, diagonality), axis=1, join='inner')
    df.sort_values(by='snr')
    df.rename(columns=colnames, inplace=True)
    # do the scatterplot just to get plot dimensions and garnishes
    df.plot.scatter(x=colnames['snr'], y=colnames['diagonality'],
                    ax=ax, title=plot_titles[featsys], legend=False, color='w')
    # recycle colors if there aren't enough
    while len(colors) < df.shape[0]:
        colors = colors * 2
    # now add the text
    new_order = ([colnames[x] for x in ['snr', 'diagonality']] + ['subj'])
    row_tuples = df.reset_index()[new_order].iterrows()
    for row, c in zip(row_tuples, colors):
        ax.text(*row[1], ha='center', va='center', color=c, weight='bold',
                size=8)
    # plot lims
    ax.set_ylim(0.32, 0.72)
    ax.set_yticks(np.linspace(0.4, 0.7, 4))
    # xaxis
    ax.set_xticks(np.linspace(4, 7, 4))
    if ax != axs[1]:
        ax.set_xlabel('')
    # regression line
    (slope, intercept, rval, pval,
     stderr) = linregress(df.iloc[:, 0], df.iloc[:, 1])
    x = np.array(ax.get_xlim())
    y = slope * x + intercept
    ax.plot(x, y, linestyle='--', linewidth=1, color='0.5', alpha=0.5,
            zorder=1)
    ax.annotate(s=f'R²={np.round(rval ** 2, 2)}\np={np.round(pval, 2)}',
                xy=(0, 1), xytext=(2, -2), xycoords='axes fraction',
                textcoords='offset points', ha='left', va='top', color='0.5',
                size=8)

fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.8, wspace=0.4)
fig.suptitle('SNR versus matrix diagonality')
fig.savefig(op.join('figures', 'supplement',
                    f'snr-vs-matrix-diagonality{trunc}.pdf'))
