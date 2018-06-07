#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-feature-system-matrices.py'
===============================================================================

This script plots binary feature matrices for each system.
"""
# @author: drmccloy
# Created on Tue Mar  6 10:00:49 PST 2018
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from aux_functions import plot_featmat

# FLAGS
savefig = True
plt.ioff()
target = 'manuscript'  # manuscript, jobtalk
box_around = 'ð'  # IPA or None
no_title = True
ftype = 'svg'

# BASIC FILE I/O
paramdir = 'params'
outdir = op.join('figures', target)
feature_sys_fname = 'all-features.tsv'

# LOAD PARAMS FROM YAML
figure_param_file = 'jobtalk-figure-params.yaml'
with open(op.join(paramdir, figure_param_file), 'r') as f:
    figure_params = yaml.load(f)
    feature_order = figure_params['feat_order']
    highlight_color = figure_params['red']

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

# load plot styles
styles = [op.join(paramdir, 'matplotlib-style-confmats.yaml'),
          {'xtick.major.size': 0, 'xtick.minor.size': 0,
           'ytick.major.size': 0, 'ytick.minor.size': 0,
           'xtick.labelsize': 10, 'ytick.labelsize': 11}]
if target == 'jobtalk':
    styles = ['dark_background'] + styles + [{'xtick.color': '0.3',
                                              'ytick.color': '0.3'}]
    # colors
    cmap = LinearSegmentedColormap.from_list(name='tol', N=2,
                                             colors=['0.5', '0.9'])
    cmap.set_bad('0')
    gridcol = '0.2'
    gridlwd = 1.5
    ticklabcol = 'w'
else:
    # colors
    cmap = LinearSegmentedColormap.from_list(name='tol', N=2,
                                             colors=['0.85', '0.55'])
    cmap.set_bad('1')
    gridcol = '0.9'
    gridlwd = 1.35
    ticklabcol = 'k'

plt.style.use(styles)

for featsys in feature_systems:
    # init figure
    fig, ax = plt.subplots(figsize=(3, 6))
    left = 0.1
    fig.subplots_adjust(hspace=0.5, left=left, right=1., bottom=0.15,
                        top=0.95)
    # load data
    featmat = pd.read_csv(op.join(paramdir, feature_fnames[featsys]), sep='\t',
                          index_col=0, comment="#")
    # fix column names for PSA
    if featsys == 'jfh_sparse':
        featmat.columns = list(map(lambda x: x.split('-')[0], featmat.columns))
    # remove engma
    featmat = featmat.loc[canonical_phone_order]
    # put features in quasi-consistent order
    featmat = featmat[feature_order[featsys]]
    # title
    if no_title:
        title = ''
    else:
        title = '{}  ({} features)'.format(feat_sys_names[featsys],
                                           len(featmat.columns))
    # plot
    plot_featmat(featmat, ax=ax, cmap=cmap, title=title)
    # grid
    ax.grid(which='minor', axis='y', color=gridcol, linewidth=gridlwd)
    ax.grid(which='minor', axis='x', color=gridcol, linewidth=gridlwd)
    # garnishes
    _ = [l.set_rotation(90) for l in ax.get_xticklabels(which='both')]
    _ = [l.set_color(ticklabcol) for l in ax.get_xticklabels(which='both')]
    _ = [l.set_color(ticklabcol) for l in ax.get_yticklabels(which='both')]
    # enforce equal left margin across figures (without this, axes with fixed
    # aspect ratio get centered in the available space)
    fig.canvas.draw()
    pos = ax.get_position()
    fig.subplots_adjust(right=left + pos.width)
    # box
    if box_around is not None:
        xy = np.array([0, featmat.index.tolist().index(box_around)]) - 0.5
        rect = Rectangle(xy=xy, width=featmat.shape[1], height=1,
                         facecolor='none', zorder=10,
                         edgecolor=highlight_color, linewidth=3, clip_on=False)
        ax.add_artist(rect)
    # save
    if savefig:
        fname = 'fig-{}-vertical.{}'.format(feat_sys_names[featsys].lower(),
                                            ftype)
        fig.savefig(op.join(outdir, fname))

if not savefig:
    plt.ion()
    plt.show()
