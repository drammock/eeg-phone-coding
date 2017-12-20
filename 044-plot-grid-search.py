#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-grid-search.py'
===============================================================================

This script plots a heatmap of the grid-searched params for a classifier. Much
was borrowed from here:
http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
"""
# @author: drmccloy
# Created on Thu Oct 19 17:51:11 PDT 2017
# License: BSD (3-clause)

import yaml
import os.path as op
from os import mkdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import ImageGrid

# flags
svm = False
unified_color_scale = True

# basic file I/O
indir = 'processed-data' if svm else 'processed-data-logistic'
outdir = op.join('figures', 'grid-search')
paramdir = 'params'
if not op.isdir(outdir):
    mkdir(outdir)

# load params
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    features = analysis_params['features']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    align_on_cv = analysis_params['align_on_cv']
    skip = analysis_params['skip']

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
if svm:
    suffix = ('unified-colorscale' if unified_color_scale else
              'inividual-colorscales')
else:
    suffix = 'logistic'

# init container
classifiers = dict()
scores = dict()
_subjects = {k: v for k, v in subjects.items() if k not in skip}

# load the classifiers (so we can set an appropriate global color scale)
for i, subj_code in enumerate(_subjects):
    subj_indir = op.join(indir, 'classifiers', subj_code)
    for j, feat in enumerate(features):
        # load the classifier
        fname = 'classifier-{}{}{}-{}.npz'.format(cv, nc, feat, subj_code)
        obj = np.load(op.join(subj_indir, fname))
        key = obj.keys()[0]
        classifiers[key] = obj[key].item()
        scores[key] = classifiers[key].cv_results_['mean_test_score']

# set global color scale
scores = pd.DataFrame(scores).values
vmin = scores.min()
vmax = scores.max()
norm = Normalize(vmin, vmax)

# initialize figure
dims = (len(_subjects), len(features))
figsize = (dims[1] * 3, dims[0] * 4)
if svm:
    fig = plt.figure(figsize=figsize)
    axs = ImageGrid(fig, rect=111, nrows_ncols=dims, axes_pad=0.2)
else:
    fig, axs = plt.subplots(*dims, figsize=figsize)

# plot
for i, subj_code in enumerate(_subjects):
    for j, feat in enumerate(features):
        if svm:
            index = i * len(features) + j
            ax = axs[index]
        else:
            ax = axs[i, j]
        key = '{}-{}'.format(subj_code, feat)
        clf = classifiers[key]
        # get grid search scores
        c_range = clf.param_grid[0]['C']
        scores = clf.cv_results_['mean_test_score']
        if svm:
            gamma_range = clf.param_grid[0]['gamma']
            scores = scores.reshape(len(c_range), len(gamma_range))
            ax.imshow(scores, interpolation='nearest', cmap=plt.cm.inferno,
                      norm=norm)
            # ax.colorbar()
            ax.xaxis.set_ticks(np.arange(len(gamma_range)))
            ax.yaxis.set_ticks(np.arange(len(c_range)))
            ax.yaxis.set_ticklabels(c_range)
            ax.xaxis.set_ticklabels(gamma_range, rotation=90)
            ax.set_xlabel('gamma')
        else:
            ax.plot(scores)
            ax.xaxis.set_ticks(np.arange(len(c_range)))
            ax.xaxis.set_ticklabels(c_range, rotation=90)
            ax.set_ylim(0, 1)
        if not j:
            ax.set_ylabel(subj_code, fontsize=24)
        if not i:
            ax.set_title(feat, fontsize=24)
fig.suptitle('validation accuracy')
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.98)
fig.savefig(op.join(outdir, 'grid-search-params-{}.pdf'.format(suffix)))
