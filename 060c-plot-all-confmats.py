#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-all-confmats.py'
===============================================================================

This script plots confusion matrices.
"""
# @author: drmccloy
# Created on Tue Jan 16 12:30:47 PST 2018
# License: BSD (3-clause)

import copy
import yaml
from os import mkdir
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm, Normalize
from aux_functions import (plot_confmat, plot_dendrogram, simulate_confmat,
                           matrix_row_column_correlation)

# FLAGS
savefig = True
logarithmic_cmap = True
plt.ioff()

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
    use_eer = analysis_params['use_eer_in_plots']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    truncate = analysis_params['eeg']['truncate']
del analysis_params

# FILE NAMING VARIABLES
trunc = '-truncated' if truncate else ''

# BASIC FILE I/O
feature_sys_fname = 'all-features.tsv'
outdir = op.join('figures', 'confusion-matrices')

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
diff_confmats_dict = dict()
subjects.update(dict(average=0))

# load feature system ground truths
ground_truth = pd.read_csv(op.join(paramdir, feature_sys_fname), sep='\t',
                           index_col=0, comment='#')
# passing dtype=float to `read_csv` doesn't work when index col. is strings
ground_truth = ground_truth.astype(float)

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
eer = 'eer-' if use_eer else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# load all dendrogram labels
dg_label_file = op.join(paramdir, 'dendrogram-labels.yaml')
with open(dg_label_file, 'r') as f:
    dg_labels = yaml.load(f)

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
    datadir = (f'processed-data-{sch}{trunc}' if phone_level else
               f'processed-data-logistic{trunc}')
    indir = op.join(datadir, 'ordered-confusion-matrices')
    dgdir = op.join(datadir, 'dendrograms')
    if not op.isdir(outdir):
        mkdir(outdir)

    # init second-level containers
    confmats_dict[sch] = dict()
    confmats_dict[sch]['simulated'] = None
    if sch in feat_sys_cols:
        diff_confmats_dict[sch] = dict()
        diff_confmats_dict[sch]['simulated'] = None

    # load accuracy info, and compute most appropriate simulation accuracy.
    if not phone_level:
        acc_info = pd.read_csv(op.join(datadir,
                               'cross-subj-matrix-maxima.tsv'),
                               sep='\t', index_col=0)
        # given there are 9, 10, or 11 features depending on which system we
        # pick, we'll use 10 as the exponent (which here becomes the divisor):
        # x^10 = y  →  x = 10^(log(y)/10)
        accuracy = 10 ** (np.log10(acc_info.loc['average'].min()) / 10.)
        accuracy = '{:.1}'.format(np.round(accuracy, 1))
        # load theoretical confmat
        prefix = 'cross-subj-row-ordered-theoretical-confusion-matrix-'
        fname = prefix + '{}-eng-{}-{}.tsv'.format(sfn, sch, accuracy)
        confmat = pd.read_csv(op.join(indir, fname), sep='\t', index_col=0)
        confmats_dict[sch]['simulated'] = confmat

    # load subject data
    for subj_code in subjects:
        prefix = 'cross-subj-row-ordered-eer-confusion-matrix-'
        middle_arg = cv + nc if phone_level else cv + nc + sch + '-'
        args = [sfn, middle_arg, subj_code]
        fn = prefix + '{}-eng-{}{}.tsv'.format(*args)
        confmat = pd.read_csv(op.join(indir, fn), sep='\t', index_col=0)
        confmats_dict[sch][subj_code] = confmat
        if sch in feat_sys_cols:
            # compute simulated confmat matched to each subject's peak accuracy
            fpath = op.join(datadir, 'cross-subj-matrix-maxima.tsv')
            acc_info = pd.read_csv(fpath, sep='\t', index_col=0)
            this_max = acc_info.loc[subj_code, sch]
            this_acc = 10 ** (np.log10(this_max) / 10.)
            this_phones = canonical_phone_order['eng']
            this_feats = feature_systems[sch]
            featmat = ground_truth.loc[this_phones, this_feats]
            sim_confmat = simulate_confmat(featmat, this_acc)
            diff_confmats_dict[sch][subj_code] = confmat - sim_confmat

# convert to DataFrame of DataFrames. axes: (subj_code, sch)
confmats = pd.DataFrame(confmats_dict, dtype=object)
diff_confmats = pd.DataFrame(diff_confmats_dict, dtype=object)

# set common color scale
vmin = confmats.applymap(lambda x: 999 if x is None else x[x > 0].min().min())
vmin = vmin.values.min()
normalizer = (LogNorm(vmin=vmin, vmax=1) if logarithmic_cmap else
              Normalize(vmin=0, vmax=1))
# set common color scale for difference matrices...
vmin = diff_confmats.applymap(lambda x: np.inf if x is None else
                              x.values.min()).values.min()
vmax = diff_confmats.applymap(lambda x: -np.inf if x is None else
                              x.values.max()).values.max()
# ...but keep it symmetrical so the diverging colorscale is obvious
vext = np.max([np.abs(vmin), vmax])
diff_normalizer = Normalize(vmin=(0 - vext), vmax=vext)

# init figure. Add extra column for dendrogram, and 3 extra rows for
# subj_data minus simulated data.
grid_shape = np.array(confmats.shape[::-1]) + np.array([3, 1])
figsize = tuple(grid_shape[::-1] * 3)
# make first column (dendrogram) different width than the rest
gridspec_kw = dict(width_ratios=([8] + [4] * (grid_shape[1] - 1)))
fig, axs = plt.subplots(*grid_shape, figsize=figsize, gridspec_kw=gridspec_kw,
                        squeeze=False)

rows = scheme_order + scheme_order[-3:]
for row, sch in enumerate(rows):
    phone_level = sch in ['pairwise', 'OVR', 'multinomial']

    # FILE I/O
    datadir = ('processed-data-{}'.format(sch) if phone_level else
               'processed-data-logistic')
    indir = op.join(datadir, 'ordered-confusion-matrices')
    dgdir = op.join(datadir, 'dendrograms')

    # load dendrogram
    args = [sfn] + ([cv + nc[:-1]] if phone_level else
                    [cv + nc + sch])
    prefix = 'cross-subj-row-ordered-dendrogram-'
    dgfn = prefix + '{}-eng-{}.yaml'.format(*args)
    with open(op.join(dgdir, dgfn), 'r') as dgf:
        dg = yaml.load(dgf)
        # suppress dendrogram colors
        dg['color_list'] = ['0.8'] * len(dg['color_list'])

    # select dendrogram label set
    this_dg_labels = (dg_labels['phone_level'][sch] if phone_level else
                      dg_labels['logistic'][sch])

    # plot dendrogram
    ax = axs[row, 0]
    plot_dendrogram(dg, orientation='left', ax=ax, linewidth=0.5,
                    leaf_rotation=0)
    ax.invert_yaxis()
    ax.yaxis.set_visible(False)  # comment this out to confirm correct ordering
    # ax.axis('off')  # enable this to suppress xscale
    if sch == 'multinomial':
        xlim = (0.025, 0.01)
    elif sch == 'OVR':
        xlim = (10, 0)
    elif sch == 'pairwise':
        xlim = (3.9, 2.9)
    else:
        xlim = (7, 0)
    ax.set_xlim(*xlim)
    # annotate dendrogram
    order = np.argsort([y[1] for y in dg['dcoord']])[::-1]
    for i, d, n in zip(np.array(dg['icoord'])[order],
                       np.array(dg['dcoord'])[order], this_dg_labels):
        xy = (d[1], sum(i[1:3]) / 2.)
        kwargs = dict(xytext=(1, 0), textcoords='offset points',
                      va='center', ha='left', fontsize=4)
        ax.annotate(n, xy, **kwargs)
        ax.set_title(feat_sys_names[sch])

    # plot subject-specific confmats first, then average, then simulated
    cols = [x for x in confmats.index if x not in ['average', 'simulated']]
    cols += ['average', 'simulated']
    for col, subj_code in enumerate(cols, start=1):
        ax = axs[row, col]
        # last 3 rows are the difference confmats (neural data - simulated)
        cm = confmats if row < (len(rows) - 3) else diff_confmats
        data = cm.loc[subj_code, sch]
        if data is None:  # i.e., the 'simulated' column for some rows
            ax.axis('off')
        else:
            if row < (len(rows) - 3):
                diag = matrix_row_column_correlation(data)
            # column titles
            title = ('Simulated ({} accuracy)'.format(accuracy) if
                     subj_code == 'simulated' else subj_code)
            if row < (len(rows) - 3):
                title = ' '.join([title, '({:.2f})'.format(diag)])
            # plot
            kwargs = dict(norm=normalizer, cmap=cmap_copy, title=title,
                          xlabel='', ylabel='')
            if row > (len(rows) - 4):
                kwargs.update(dict(cmap='RdYlGn', norm=diff_normalizer))
            plot_confmat(data, ax, **kwargs)

# adjust whitespace
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.025, top=0.975,
                    wspace=0.1, hspace=0.33)

plt.style.use({'ytick.labelsize': 14, 'ytick.color': 'k',
               'ytick.major.size': 8})
# add colorbar
bbox = np.array(axs[2, -1].get_position().bounds)
bbox[-1] += axs[0, -1].get_position().bounds[1] - bbox[1]  # make taller
bbox[-2] /= 4.                                             # make narrower
bbox[0] += 1.5 * bbox[-2]                                  # shift rightward
cax = fig.add_axes(bbox)
cbar = fig.colorbar(axs[0, 1].images[0], cax=cax)

# add difference colorbar
bbox = np.array(axs[-1, -1].get_position().bounds)
bbox[-1] += axs[-3, -1].get_position().bounds[1] - bbox[1]  # make taller
bbox[-2] /= 4.                                              # make narrower
bbox[0] += 1.5 * bbox[-2]                                   # shift rightward
cax = fig.add_axes(bbox)
cbar = fig.colorbar(axs[-1, -2].images[0], cax=cax)

if savefig:
    fname = 'cross-subj-row-ordered-confusion-matrices-'
    suffix = f'{sfn}-eng-all{trunc}.pdf'
    fig.savefig(op.join(outdir, fname + suffix))
else:
    plt.ion()
    plt.show()
