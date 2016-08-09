#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'learn-features-from-eeg.py'
===============================================================================

This script tries to learn optimal phonological features based on EEG data.
"""
# @author: drmccloy
# Created on Wed Aug  3 14:33:16 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import numpy as np
import pandas as pd
from os import path as op
from time import time
from numpy import logical_not as negate
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from aux_functions import time_domain_pca

pd.set_option('display.width', 160)
rand = np.random.RandomState(seed=0)

# flags
use_dss = True
pca_time_domain = True
cluster_method = 'spectral'  # kmeans, spectral, tsne
use_n_dss_channels = 16
n_iterations = 4
n_jobs = 10

# file i/o
paramdir = 'params'
outdir = 'processed-data'
infile = 'merged-dss-data.npz' if use_dss else 'merged-eeg-data.npz'


def split_and_resid(data, n_clusters=2, n_jobs=10, random_state=rand,
                    method='kmeans'):
    if method == 'kmeans':
        km = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                    random_state=random_state)
        predictions = km.fit_predict(data)
        centers = km.cluster_centers_[predictions]
        assert data.size == centers.size
        residuals = data - centers
    elif method == 'tsne':
        clust = TSNE(n_components=n_clusters, random_state=random_state)
        predictions = clust.fit_transform(data)
        residuals = None
    elif method == 'spectral':
        clust = SpectralClustering(n_clusters=n_clusters, n_init=25,
                                   affinity='nearest_neighbors',
                                   n_neighbors=10,
                                   eigen_solver='amg',
                                   random_state=random_state)
        predictions = clust.fit_predict(data)
        residuals = None
        # predictions = clust.labels_
    else:
        raise ValueError()
    return predictions, residuals


print('loading data')
# load feature table
feat_ref = pd.read_csv(op.join(paramdir, 'reference-feature-table-cons.tsv'),
                       sep='\t', index_col=0, encoding='utf-8')
featcol = feat_ref.columns.tolist()

# load merged EEG data & other params
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

# reduce dimensionality of time domain with PCA
if pca_time_domain:
    print('running PCA on time domain')
    epochs = time_domain_pca(epochs)
epochs_cat = epochs[:, :use_n_dss_channels, :].reshape(epochs.shape[0], -1)
training_data = epochs_cat[train_mask]
df = full_df.loc[train_mask].copy()

# do the splits
data = training_data.copy()
for _iter in range(n_iterations):
    print('running {} clustering (iteration {})'.format(cluster_method, _iter),
          end='  duration: ')
    colnum = 'split_{}'.format(_iter)
    _s = time()
    predictions, residuals = split_and_resid(data, method=cluster_method)
    _n = time()
    df[colnum] = predictions
    data = residuals
    print(np.round(_n - _s, 2))

# calc correspondence between feature and prediction
print('calculating feature correspondences')
results = pd.DataFrame()
for _iter in range(n_iterations):
    colnum = 'split_{}'.format(_iter)
    class_one = df[colnum].astype(bool)
    class_zero = negate(class_one)
    truepos = df.loc[class_one].sum()
    falseneg = df.loc[class_zero].sum()
    trueneg = negate(df.loc[class_zero]).sum()
    falsepos = negate(df.loc[class_one]).sum()
    results['{}_match'.format(colnum)] = (truepos[featcol] + trueneg[featcol]) / df.shape[0]
    results['{}_mismatch'.format(colnum)] = (falsepos[featcol] + falseneg[featcol]) / df.shape[0]

raise RuntimeError
results.sort_values(by='split_0_match', ascending=False)
adjusted_rand_score(df['consonantal'], df['split_0'])
