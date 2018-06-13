#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-phone-by-feature-matrices.py'
===============================================================================

This script plots binary feature matrices for each system.
"""
# @author: drmccloy
# Created on Mon Feb 19 14:44:19 PST 2018
# License: BSD (3-clause)

import yaml
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
from aux_functions import plot_featmat

# FLAGS
savefig = True

# BASIC FILE I/O
paramdir = 'params'
outdir = op.join('figures', 'manuscript')
feature_sys_fname = 'all-features.tsv'

# LOAD PARAMS FROM YAML
figure_param_file = 'jobtalk-figure-params.yaml'
with open(op.join(paramdir, figure_param_file), 'r') as f:
    figure_params = yaml.load(f)
    feature_order = figure_params['feat_order']

analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    feature_fnames = analysis_params['feature_fnames']
    feature_systems = analysis_params['feature_systems']
    canonical_phone_order = analysis_params['canonical_phone_order']['eng']

# only do the 3 original feature systems
del feature_systems['jfh_dense']
del feature_systems['spe_dense']
del feature_systems['phoible_sparse']

feat_sys_names = dict(jfh_sparse='PSA', spe_sparse='SPE',
                      phoible_redux='PHOIBLE')

# load plot style
plt.style.use([op.join(paramdir, 'matplotlib-style-confmats.yaml'),
               op.join(paramdir, 'matplotlib-font-myriad.yaml'),
               {'xtick.major.size': 0, 'xtick.minor.size': 0,
                'ytick.major.size': 0, 'ytick.minor.size': 0,
                'xtick.labelsize': 11, 'ytick.labelsize': 11,
                'xtick.color': '0.3', 'ytick.color': '0.3'}])
cmap = LinearSegmentedColormap.from_list(name='tol', N=2,
                                         colors=['0.8', '0.5'])
cmap.set_bad('1')

# init figure
fig = plt.figure(figsize=(6, 8))
axs = ImageGrid(fig, 111, nrows_ncols=(3, 1), axes_pad=0.7, label_mode='all')
fig.subplots_adjust(hspace=0.5, left=0.1, right=1, bottom=0.04, top=0.96)

# loop over feature systems
for ax, featsys in zip(axs, feature_systems):
    featmat = pd.read_csv(op.join(paramdir, feature_fnames[featsys]), sep='\t',
                          index_col=0, comment="#")
    if featsys == 'jfh_sparse':
        featmat.columns = list(map(lambda x: x.split('-')[0], featmat.columns))
    # put features in quasi-consistent order
    featmat = featmat.loc[canonical_phone_order]  # remove engma
    featmat = featmat[feature_order[featsys]]     # quasi-consistent order
    # title
    title = '{}  ({} features)'.format(feat_sys_names[featsys],
                                       len(featmat.columns))
    # plot
    plot_featmat(featmat.T, ax=ax, cmap=cmap, title=title)
    # grid
    ax.grid(which='minor', axis='y', color='0.9', linewidth=1.35)
    ax.grid(which='minor', axis='x', color='0.9', linewidth=1.35)

if savefig:
    fig.savefig(op.join(outdir, 'fig-featsys-matrices.pdf'))
else:
    plt.ion()
    plt.show()
