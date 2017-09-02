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

np.set_printoptions(precision=6, linewidth=130)
pd.set_option('display.width', 130)
plt.ioff()

# BASIC FILE I/O
paramdir = 'params'
indir = 'processed-data'
outdir = op.join('figures', 'snr-vs-diagonality')
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS FROM YAML
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    methods = analysis_params['methods']
    feature_systems = analysis_params['feature_systems']
del analysis_params

# load SNR data
fname = op.join(indir, 'blinks-epochs-snr.tsv')
snr = pd.read_csv(fname, sep='\t', index_col=0)
snr = snr['snr']  # ignore other columns (n_trials, n_blinks, retained_epochs)

# ignore simulation data
_ = methods.pop(methods.index('theoretical'))

# load plot style
plt.style.use(op.join(paramdir, 'matplotlib-style-lineplots.yaml'))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# pretty titles
colnames = dict(snr='SNR: 10×log(evoked/baseline power)',
                diagonality='Matrix row/col correlation')
legend_names = dict(phoible_sparse='PHOIBLE',
                    jfh_dense='JF&H (dense)',
                    jfh_sparse='JF&H (orig.)',
                    spe_dense='SPE (dense)',
                    spe_sparse='SPE (orig.)')
titles = dict(phone='Phone-level', eer='Feature-level')

fig, axs = plt.subplots(len(methods), len(feature_systems), figsize=(18, 9))
for ax_row, method in zip(axs, methods):
    fname = op.join(indir, 'matrix-correlations',
                    'matrix-diagonality-{}.tsv'.format(method))
    diag_df = pd.read_csv(fname, sep='\t', index_col=0)
    for ax, feat_sys in zip(ax_row, feature_systems):
        diagonality = diag_df[feat_sys]
        diagonality.name = 'diagonality'
        # merge
        df = pd.concat((snr, diagonality), axis=1, join='inner')
        df.sort_values(by='snr')
        df.rename(columns=colnames, inplace=True)
        # plot
        title = '{}: {}'.format(titles[method], legend_names[feat_sys])
        # do the scatterplot just to get plot dimensions and garnishes
        df.plot.scatter(x=colnames['snr'], y=colnames['diagonality'], ax=ax,
                     title=title, legend=False, color='w')
        # recycle colors if there aren't enough
        while len(colors) < df.shape[0]:
            colors = colors * 2
        # now add the text
        new_order = [colnames[x] for x in ['snr', 'diagonality']] + ['index']
        row_tuples = df.reset_index()[new_order].iterrows()
        for row, c in zip(row_tuples, colors):
            ax.text(*row[1], ha='center', va='center', color=c, weight='bold')
        # plot lims
        ylim = np.array([diag_df.values.min(), diag_df.values.max()])
        ylim = np.sign(ylim) * np.ceil(np.abs(10 * ylim) + 1) / 10.
        ax.set_ylim(*ylim)

fig.subplots_adjust(left=0.05, right=0.95, wspace=0.8, hspace=0.4)
fig.suptitle('SNR vs. matrix diagonality (empirical accuracies)')
fig.savefig(op.join(outdir, 'snr-vs-matrix-diagonality.pdf'))
fig.savefig(op.join(outdir, 'snr-vs-matrix-diagonality.png'))
