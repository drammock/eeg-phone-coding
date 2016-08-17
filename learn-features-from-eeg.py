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
import networkx as nx
from os import path as op
from time import time
from numpy import logical_not as negate
from scipy.sparse import linalg, csr_matrix
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_laplacian
from aux_functions import time_domain_pca

np.set_printoptions(precision=6, linewidth=160)
pd.set_option('display.width', 160)
rand = np.random.RandomState(seed=0)

# flags
use_dss = True
pca_time_domain = True
# kmeans, spectral, precomputed, tsne, sparse_graph, full_graph
cluster_method = 'full_graph'
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
                    method='kmeans', gamma=100):
    print('    clustering:', end=' ')
    _st = time()
    if method == 'kmeans':
        km = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                    random_state=random_state)
        predictions = km.fit_predict(data)
        centers = km.cluster_centers_
        print_elapsed(_st)
    elif method in ('tsne', 'spectral', 'precomputed'):
        if method == 'tsne':
            clust = TSNE(n_components=n_clusters, random_state=random_state)
            predictions = clust.fit_transform(data)
        elif method == 'spectral':
            clust = SpectralClustering(n_clusters=n_clusters, n_init=10,
                                       affinity='nearest_neighbors',
                                       n_neighbors=10, eigen_solver='amg',
                                       random_state=random_state)
            predictions = clust.fit_predict(data)
        else:  # spectral clustering with precomputed adjacency mat
            dist_mat = pairwise_distances(data)  # don't pass n_jobs here
            adjacency_mat = np.exp(-gamma * dist_mat)
            clust = SpectralClustering(adjacency_mat, n_clusters=n_clusters,
                                       n_init=10, affinity='precomputed')
            predictions = clust.fit_predict(data)
        print_elapsed(_st)
        centers = compute_medioids(data, predictions)  # don't pass n_jobs here
    elif method in ('sparse_graph', 'full_graph'):
        if method == 'full_graph':
            # dist_mat = pairwise_distances(data)  # don't pass n_jobs here
            # adjacency_mat = np.exp(-gamma * dist_mat)
            # # do this in-place for memory reasons
            adjacency_mat = pairwise_distances(data)
            adjacency_mat *= -gamma
            adjacency_mat = np.exp(adjacency_mat, out=adjacency_mat)
            k = n_clusters
        else:  # sparse_graph
            knn = NearestNeighbors(n_neighbors=50, algorithm='brute',
                                   n_jobs=n_jobs).fit(data)
            dist_mat = knn.kneighbors_graph(data, mode='distance')
            dist_mat.eliminate_zeros()
            rows, cols = dist_mat.nonzero()
            # make (symmetric) adjacency matrix
            adjacencies = np.exp(-gamma * dist_mat.data)
            adjacency_mat = csr_matrix((np.r_[adjacencies, adjacencies],
                                        (np.r_[rows, cols],
                                         np.r_[cols, rows])),
                                       shape=dist_mat.shape)
            # find number of connected graph components and
            # set number of eigenvectors as connected components + 1
            adjacency_graph = nx.Graph(adjacency_mat)
            k = nx.number_connected_components(adjacency_graph) + 1
            del adjacency_graph
        # eigendecomposition of graph laplacian
        laplacian = graph_laplacian(adjacency_mat, normed=True)
        del adjacency_mat
        solver = np.linalg.eigh if method == 'full_graph' else linalg.eigsh
        solver_kwargs = (dict() if method == 'full_graph' else
                         dict(k=k, which='LM', sigma=0, mode='normal'))
        eigval, eigvec = solver(laplacian, **solver_kwargs)
        # compute clusters
        if method == 'full_graph':  # pos/neg eigenvalues give clusters
            predictions = np.zeros(data.shape[0]).astype(int)
            for kk in range(k):
                if kk:  # skip first eigenvector
                    # TODO: this won't work for larger values of k
                    predictions[np.where(eigvec[:, kk] > 0)] = kk
        else:  # sparse_graph, cluster based on eigenvalues
            km = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                        random_state=random_state)
            predictions = km.fit_predict(eigvec[:, 1:])
            # centers = km.cluster_centers_  # no good, clustering eigenvectors
        print_elapsed(_st)
        centers = compute_medioids(data, predictions)
    else:
        raise ValueError('unknown clustering method "{}"'.format(method))
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
del feats, invars

# reduce dimensionality of time domain with PCA
if pca_time_domain:
    print('running PCA on time domain:', end=' ')
    _st = time()
    epochs = time_domain_pca(epochs)
    print_elapsed(_st)
# if normalize_variance:
#     print('normalizing signal variance:', end=' ')
#     _st = time()
#     do_something_here()
#     print_elapsed(_st)
epochs_cat = epochs[:, :use_n_dss_channels, :].reshape(epochs.shape[0], -1)
training_data = epochs_cat[train_mask]
df = full_df.loc[train_mask].copy()
data = training_data.copy()
# TODO: delete me (shrinking dataset for testing)
#data = training_data[df['subj'] == 1].copy()
#df = df[df['subj'] == 1]
del epochs, epochs_cat, training_data
del train_mask, test_mask, validation_mask
del full_df

# do the splits
cluster_centers = np.zeros((n_iterations, n_clusters, data.shape[-1]))
cluster_type = (' spectral' if cluster_method in
                ('precomputed', 'sparse_graph', 'full_graph') else '')
print('running {}{} clustering'.format(cluster_method, cluster_type))
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
