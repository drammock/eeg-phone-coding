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
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from aux_functions import time_domain_pca

pd.set_option('display.width', 160)
rand = np.random.RandomState(seed=0)

# flags
use_dss = True
pca_time_domain = True
cluster_method = 'spectral'  # kmeans, spectral, tsne
use_n_dss_channels = 16
n_iterations = 4
n_clusters = 2
n_jobs = 4

# file i/o
paramdir = 'params'
outdir = 'processed-data'
infile = 'merged-dss-data.npz' if use_dss else 'merged-eeg-data.npz'


def print_elapsed(start_time, end=' sec.\n'):
    print(np.round(time() - start_time, 1), end=end)


def compute_medioids(data, cluster_ids, n_jobs=1):
    # don't pass n_jobs > 1 (memory issues); pairwise_distances fast w/ 1 CPU
    print('    computing medioids:', end=' ')
    _st = time()
    _ids = np.sort(np.unique(cluster_ids))
    medioids = np.zeros((len(_ids), data.shape[1]))
    for _id in _ids:
        this_data = data[cluster_ids == _id]
        dists = pairwise_distances(this_data, n_jobs=n_jobs)
        rowsums = dists.sum(axis=1)
        medioids[_id] = this_data[rowsums.argmin()]
    print_elapsed(_st)
    return medioids


def split_and_resid(data, n_clusters=n_clusters, n_jobs=-2, random_state=rand,
                    method='kmeans'):
    print('    clustering:', end=' ')
    _st = time()
    if method == 'kmeans':
        km = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                    random_state=random_state)
        predictions = km.fit_predict(data)
        centers = km.cluster_centers_
        print_elapsed(_st)
    else:
        if method == 'tsne':
            clust = TSNE(n_components=n_clusters, random_state=random_state)
            predictions = clust.fit_transform(data)
        elif method == 'spectral':
            clust = SpectralClustering(n_clusters=n_clusters, n_init=10,
                                       affinity='nearest_neighbors',
                                       n_neighbors=10, eigen_solver='amg',
                                       random_state=random_state)
            predictions = clust.fit_predict(data)
        else:
            raise ValueError()
        print_elapsed(_st)
        centers = compute_medioids(data, predictions)  # don't pass n_jobs here
    residuals = data - centers[predictions]
    return predictions, residuals, centers


_st = time()
print('loading data:', end=' ')
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
print_elapsed(_st)

# reduce dimensionality of time domain with PCA
if pca_time_domain:
    print('running PCA on time domain:', end=' ')
    _st = time()
    epochs = time_domain_pca(epochs)
    print_elapsed(_st)
epochs_cat = epochs[:, :use_n_dss_channels, :].reshape(epochs.shape[0], -1)
training_data = epochs_cat[train_mask]
df = full_df.loc[train_mask].copy()

# do the splits
data = training_data.copy()
cluster_centers = np.zeros((n_iterations, n_clusters, data.shape[-1]))
print('running {} clustering'.format(cluster_method))
for _iter in range(n_iterations):
    print('  iteration {}:'.format(_iter))
    colnum = 'split_{}'.format(_iter)
    predictions, residuals, centers = split_and_resid(data, n_jobs=n_jobs,
                                                      n_clusters=n_clusters,
                                                      method=cluster_method)
    df[colnum] = predictions
    cluster_centers[_iter, :, :] = centers
    data = residuals

# calc correspondence between feature and prediction
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
    results['{}_match'.format(colnum)] = (truepos[featcol] + trueneg[featcol]) / df.shape[0]
    results['{}_mismatch'.format(colnum)] = (falsepos[featcol] + falseneg[featcol]) / df.shape[0]
print_elapsed(_st)

outvars = dict(df=df, centers=centers, results=results)
np.savez('cluster-results.npz', **outvars)

raise RuntimeError
results.sort_values(by='split_0_match', ascending=False)
adjusted_rand_score(df['lateral'], df['split_0'])


np.where(df.split_0 == 0)[0]
