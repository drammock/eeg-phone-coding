#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-avg-confmats.py'
===============================================================================

This script plots confusion matrices.  It addresses the system-level question:
which feature system from the literature best captures the contrasts that are
recoverable with EEG?
"""
# @author: drmccloy
# Created on Thu Jan 18 15:02:09 PST 2018
# License: BSD (3-clause)

import copy
import yaml
from os import mkdir
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from aux_functions import plot_confmat, matrix_row_column_correlation

# FLAGS
savefig = True
plt.ioff()
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 30)

# BASIC FILE I/O
outdir = op.join('figures', 'manuscript')
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    feature_systems = analysis_params['feature_systems']
    canonical_phone_order = analysis_params['canonical_phone_order']
    subj_langs = analysis_params['subj_langs']
    use_eer = analysis_params['use_eer_in_plots']
    accuracies = analysis_params['theoretical_accuracies']
    methods = analysis_params['methods']
    lang_names = analysis_params['pretty_lang_names']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    skip = analysis_params['skip']
    scheme = analysis_params['classification_scheme']
del analysis_params

# only do the 3 original feature systems (not the dense ones)
del feature_systems['jfh_dense']
del feature_systems['spe_dense']
del feature_systems['phoible_sparse']
feat_sys_names = dict(jfh_sparse='PSA', spe_sparse='SPE',
                      phoible_redux='PHOIBLE', pairwise='Pairwise',
                      OVR='One-vs-rest', multinomial='Multinomial')

# init containers. Must be done as nested dict and converted afterwards;
# creating as pd.Panel converts embedded DataFrames to ndarrays.
confmats_dict = dict()

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
eer = 'eer-' if use_eer else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# load plot style; make colormap with NaN data (from log(0)) mapped as gray
plt.style.use(op.join(paramdir, 'matplotlib-style-confmats.yaml'))
cmap_copy = copy.copy(get_cmap())
# https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems
gray = cmap_copy.colors[127] @ np.array([0.299, 0.587, 0.114])
cmap_copy.set_bad(str(gray))
# cmap_copy.set_bad(cmap_copy.colors[0])

# loop over phone-level methods and feature systems
feat_sys_cols = ['jfh_sparse', 'spe_sparse', 'phoible_redux']
scheme_order = ['multinomial', 'OVR', 'pairwise'] + feat_sys_cols
for sch in scheme_order:
    phone_level = sch in ['pairwise', 'OVR', 'multinomial']

    # FILE I/O
    datadir = ('processed-data-{}'.format(sch) if phone_level else
               'processed-data-logistic')
    indir = op.join(datadir, 'ordered-confusion-matrices')

    prefix = 'cross-subj-row-ordered-eer-confusion-matrix-'
    middle_arg = cv + nc if phone_level else cv + nc + sch + '-'
    args = [sfn, middle_arg, 'average']
    fn = prefix + '{}-eng-{}{}.tsv'.format(*args)
    confmat = pd.read_csv(op.join(indir, fn), sep='\t', index_col=0)
    confmats_dict[sch] = confmat

# convert to Panel to compute color scale. axes: (sch, phones_in, phones_out)
confmats = pd.Panel(confmats_dict)
vmin = confmats.apply(lambda x: x.values.min(), axis=('major', 'minor')).min()
normalizer = LogNorm(vmin=vmin, vmax=1)

# init figure
figsize = (7.5, 5)
grid_shape = np.array([2, 3]) + np.array([0, 1])  # extra col for colorbar
gridspec_kw = dict(width_ratios=[4, 4, 4, 1])     # make colorbar narrower
fig, axs = plt.subplots(*grid_shape, figsize=figsize, gridspec_kw=gridspec_kw,
                        squeeze=False)

# make space for colorbar
scheme_order.insert(3, None)
scheme_order.append(None)

for ax, sch in zip(axs.ravel(), scheme_order):
    phone_level = sch in ['pairwise', 'OVR', 'multinomial']
    if sch is None:
        ax.axis('off')
        continue
    # get data
    data = confmats_dict[sch]
    diag = matrix_row_column_correlation(data)
    # put rows / cols in same order for all (already true for non-phone-level)
    order = confmats_dict['phoible_redux'].index
    data = data.loc[order, order]
    # titles
    title = feat_sys_names[sch]
    title = ' '.join([title, '({:.2f})'.format(diag)])
    # plot
    kwargs = dict(norm=normalizer, cmap=cmap_copy, title=title,
                  xlabel='', ylabel='')
    plot_confmat(data, ax, **kwargs)

# adjust whitespace
fig.subplots_adjust(left=0.05, right=0.9, bottom=0.05, top=0.95,
                    wspace=0.33, hspace=0.2)

# add colorbar
plt.style.use({'ytick.labelsize': 9, 'ytick.color': 'k',
               'ytick.major.size': 4})
bbox = np.array(axs[-1, -1].get_position().bounds)
bbox[-1] += axs[0, -1].get_position().bounds[1] - bbox[1]  # make taller
cax = fig.add_axes(bbox)
cbar = fig.colorbar(axs[0, 0].images[0], cax=cax)
cbar.set_label('Probability', rotation=270, labelpad=16, size=11)
cbar.outline.set_linewidth(0.2)

if savefig:
    fig.savefig(op.join(outdir, 'fig-avg-confmats.pdf'))
else:
    plt.ion()
    plt.show()
