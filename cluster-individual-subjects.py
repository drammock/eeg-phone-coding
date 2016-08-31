#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'cluster-individual-subjs.py'
===============================================================================

This script tries to learn optimal phonological features based on EEG data,
one subject at a time.
"""
# @author: drmccloy
# Created on Fri Aug 26 12:09:55 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op
from time import time
from numpy import logical_not as negate
# from sklearn.metrics import adjusted_rand_score
from aux_functions import time_domain_pca, print_elapsed, split_and_resid

np.set_printoptions(precision=6, linewidth=160)
pd.set_option('display.width', 160)
rand = np.random.RandomState(seed=0)

# flags
use_dss = True
pca_time_domain = True
# chosen by visual inspection of plot_erp_dss.py (`None` uses all timepts):
truncate_pca_to_timepts = 20
conserve_memory = True
# kmeans, spectral, precomputed, tsne, sparse_graph, dense_graph, sparse_ncut
# dense_ncut
cluster_method = 'tsne'
use_n_dss_channels = 32
n_iterations = 1  # num. times to subtract centers and re-cluster on residuals
n_clusters = 2    # num. clusters to split into at each iteration
n_neighbors = 20  # used to make dist_mat in sparse_graph and spectral methods
n_jobs = 6        # sparse_ncut: 6
gamma = 100
which_subjs = ['FV', 'IQ', 'IP', 'IT', 'IV', 'FA', 'IJ', 'IM', 'IL', 'IA',
               'CQ', 'ID']

# file i/o
paramdir = 'params'
outdir = 'processed-data'
infile = 'merged-dss-data.npz' if use_dss else 'merged-eeg-data.npz'

_st = time()
print('loading data:', end=' ')
# load feature table
feat_ref = pd.read_csv(op.join(paramdir, 'reference-feature-table-cons.tsv'),
                       sep='\t', index_col=0, encoding='utf-8')
featcol = feat_ref.columns.tolist()
# load merged EEG data & other params
subj_dict = np.load(op.join(paramdir, 'subjects.npz'))
invars = np.load(op.join(outdir, infile))
epochs = invars['epochs']
feats = invars['feats']
test_mask = invars['test_mask']
train_mask = invars['train_mask']
validation_mask = invars['validation_mask']
# organize
full_df = pd.DataFrame(feats)
full_df['segment'] = invars['cons']
full_df['lang'] = invars['langs']
full_df['subj'] = invars['subjs']
print_elapsed(_st)
if conserve_memory:
    del feats, invars

# reduce dimensionality of time domain with PCA
if pca_time_domain:
    print('running PCA on time domain:', end=' ')
    _st = time()
    epochs = time_domain_pca(epochs)
    if truncate_pca_to_timepts is not None:
        epochs = epochs[:, :, :truncate_pca_to_timepts]
    print_elapsed(_st)
# if normalize_variance:
#     print('normalizing signal variance:', end=' ')
#     _st = time()
#     do_something_here()
#     print_elapsed(_st)
epochs_cat = epochs[:, :use_n_dss_channels, :].reshape(epochs.shape[0], -1)
training_data = epochs_cat[train_mask].copy()
training_df = full_df.loc[train_mask].copy()

if conserve_memory:
    del (epochs, epochs_cat, train_mask, test_mask, validation_mask, full_df)

cluster_type = (' spectral' if cluster_method in
                ('precomputed', 'sparse_graph', 'dense_graph',
                 'sparse_ncut', 'dense_ncut') else '')

# iterate over subjects
for subj_id in which_subjs:
    print('processing subject {}'.format(subj_id))
    subset = np.in1d(training_df['subj'], subj_dict[subj_id])
    data = training_data[subset]
    df = training_df[subset]
    # do the splits
    cluster_centers = np.zeros((n_iterations, n_clusters, data.shape[-1]))
    eigvals = dict()
    eigvecs = dict()
    print(' running {}{} clustering'.format(cluster_method, cluster_type))
    for _iter in range(n_iterations):
        print('  iteration {}:'.format(_iter))
        colnum = 'split_{}'.format(_iter)
        split_kwargs = dict(n_jobs=n_jobs, n_clusters=n_clusters,
                            method=cluster_method, gamma=gamma,
                            random_state=rand)
        if (cluster_method.startswith('sparse') or
                cluster_method.startswith('dense')):
            (predictions, residuals, centers, evals,
             evecs) = split_and_resid(data, **split_kwargs)
            eigvals[colnum] = evals
            eigvecs[colnum] = evecs
        elif cluster_method == 'tsne':
            embedding = split_and_resid(data, **split_kwargs)
            for _ix in range(n_clusters):
                df[colnum + str(_ix)] = embedding[:, _ix]
        else:
            predictions, residuals, centers = split_and_resid(data,
                                                              **split_kwargs)
        if cluster_method != 'tsne':
            df[colnum] = predictions
            cluster_centers[_iter, :, :] = centers
            data = residuals
    # calc correspondence between feature and prediction
    raise RuntimeError
    print('calculating feature correspondences:', end=' ')
    _st = time()
    results = pd.DataFrame()
    for _iter in range(n_iterations):
        colnum = 'split_{}'.format(_iter)
        class_one = df[colnum].astype(bool)
        class_zero = negate(class_one)
        truepos = df.loc[class_one].sum()
        falseneg = df.loc[class_zero].sum()
        trueneg = negate(df.loc[class_zero]).sum()
        falsepos = negate(df.loc[class_one]).sum()
        results['{}_match'.format(colnum)] = (truepos[featcol] +
                                              trueneg[featcol]) / df.shape[0]
        results['{}_mismatch'.format(colnum)] = (falsepos[featcol] +
                                                 falseneg[featcol]) / df.shape[0]
    print_elapsed(_st)
    # save results
    outvars = dict(df=df, centers=centers, results=results)
    if (cluster_method.startswith('sparse') or
            cluster_method.startswith('dense')):
        outvars.update(dict(eigvals=eigvals, eigvecs=eigvecs))

    suffix = ('-{}_neighbors'.format(n_neighbors) if cluster_method in
              ('spectral', 'sparse_graph') else '')
    out_fname = 'cluster-results-{}-dss_{}-{}_{}{}.npz'.format(subj_id,
                                                               use_n_dss_channels,
                                                               cluster_method,
                                                               n_clusters,
                                                               suffix)
    np.savez(out_fname, **outvars)

raise RuntimeError('RESULTS SAVED')
# adjusted_rand_score(df['lateral'], df['split_0'])

# plot eigvecs 1 and 2 color-coded by class
fig, axs = plt.subplots(2, 1, sharey=True, sharex=True)
for ix, _iter in enumerate(('split_0', 'split_1')):
    axs[ix].plot(eigvecs[_iter][df[_iter].values.astype(bool), 0],
                 eigvecs[_iter][df[_iter].values.astype(bool), 1], 'r.')
    axs[ix].plot(eigvecs[_iter][np.logical_not(df[_iter].values), 0],
                 eigvecs[_iter][np.logical_not(df[_iter].values), 1], 'b.')

# plot eigvecs 1 and 2 color-coded by feature
fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
for ix, feat in enumerate(results.index):
    ax = axs.ravel()[ix]
    subset = df[feat].values.astype(bool)
    if feat == 'consonantal':
        ax.plot(eigvecs['split_0'][subset, 0], eigvecs['split_0'][subset, 1],
                'r.', markersize=2, alpha=0.1)
    ax.plot(eigvecs['split_0'][np.logical_not(subset), 0],
            eigvecs['split_0'][np.logical_not(subset), 1],
            'b.', markersize=2, alpha=0.1)
    if feat != 'consonantal':
        ax.plot(eigvecs['split_0'][subset, 0], eigvecs['split_0'][subset, 1],
                'r.', markersize=2, alpha=0.1)
    ax.set_title(feat)

# TSNE: plot features as colors
fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
for ix, feat in enumerate(featcol):
    ax = axs.ravel()[ix]
    subset = df[feat].values.astype(bool)
    ax.plot(df['split_00'].loc[negate(subset)],
            df['split_01'].loc[negate(subset)], '.', alpha=0.1)
    ax.plot(df['split_00'].loc[subset], df['split_01'].loc[subset], '.',
            alpha=0.1)
    ax.set_title(feat)

# TSNE: plot segments as colors
foo = df['segment'].astype('category')
