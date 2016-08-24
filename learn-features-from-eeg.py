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
import matplotlib.pyplot as plt
import os.path as op
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
# chosen by visual inspection of plot_erp_dss.py (`None` uses all timepts):
truncate_pca_to_timepts = 20
conserve_memory = True
single_subject = False
# kmeans, spectral, precomputed, tsne, sparse_graph, dense_graph, sparse_ncut
# dense_ncut
cluster_method = 'sparse_ncut'
use_n_dss_channels = 32
n_iterations = 3  # num. times to subtract centers and re-cluster on residuals
n_clusters = 2    # num. clusters to split into at each iteration
n_neighbors = 20  # used to make dist_mat in sparse_graph and spectral methods
n_jobs = 6        # sparse_ncut: 6
gamma = 100

# file i/o
paramdir = 'params'
outdir = 'processed-data'
infile = 'merged-dss-data.npz' if use_dss else 'merged-eeg-data.npz'


def print_elapsed(start_time, end=' sec.\n'):
    print(np.round(time() - start_time, 1), end=end)


def compute_medioids(data, cluster_ids, n_jobs=1):
    # pairwise_distances fast w/ 1 CPU; don't pass n_jobs > 1 (memory issues)
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
                    method='kmeans', gamma=100, n_neighbors=10):
    print('    clustering:', end=' ')
    _st = time()
    if method == 'kmeans':
        km = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                    random_state=random_state)
        predictions = km.fit_predict(data)
        centers = km.cluster_centers_
        print_elapsed(_st)
        residuals = data - centers[predictions]
        return predictions, residuals, centers
    elif method in ('tsne', 'spectral', 'precomputed'):
        if method == 'tsne':
            clust = TSNE(n_components=n_clusters, random_state=random_state)
            predictions = clust.fit_transform(data)
        elif method == 'spectral':
            clust = SpectralClustering(n_clusters=n_clusters, n_init=10,
                                       affinity='nearest_neighbors',
                                       n_neighbors=n_neighbors,
                                       eigen_solver='amg',
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
        residuals = data - centers[predictions]
        return predictions, residuals, centers
    elif method.startswith('sparse') or method.startswith('dense'):
        print('\n      adjacency matrix:', end=' ')
        if method.startswith('dense'):
            # dist_mat = pairwise_distances(data)  # don't pass n_jobs here
            # adjacency_mat = np.exp(-gamma * dist_mat)
            # # do above 2 lines in-place, for memory reasons
            adjacency_mat = pairwise_distances(data)
            adjacency_mat *= -gamma
            adjacency_mat = np.exp(adjacency_mat, out=adjacency_mat)
            n_components = 1
        else:  # sparse_graph or sparse_ncut
            knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto',
                                   n_jobs=n_jobs).fit(data)
            dist_mat = knn.kneighbors_graph(data, mode='distance')
            dist_mat.eliminate_zeros()
            rows, cols = dist_mat.nonzero()
            # make (symmetric) adjacency matrix
            adjacencies = np.exp(-gamma * dist_mat.sqrt().data)
            adjacency_mat = csr_matrix((np.r_[adjacencies, adjacencies],
                                        (np.r_[rows, cols],
                                         np.r_[cols, rows])),
                                       shape=dist_mat.shape)
            # find number of connected graph components and
            # set number of eigenvectors as connected components + 1
            adjacency_graph = nx.Graph(adjacency_mat)
            n_components = nx.number_connected_components(adjacency_graph)
            del adjacency_graph
        # eigendecomposition of graph laplacian
        print_elapsed(_st)
        _st = time()
        print('      graph laplacian:', end=' ')
        laplacian = graph_laplacian(adjacency_mat, normed=True).tocsc()
        del adjacency_mat
        print_elapsed(_st)
        _st = time()
        print('      eigendecomposition:', end=' ')
        solver = np.linalg.eigh if method.startswith('dense') else linalg.eigsh
        solver_kwargs = (dict() if method.startswith('dense') else
                         dict(k=n_components + 1, which='LM', sigma=0,
                              mode='normal'))
        eigvals, eigvecs = solver(laplacian, **solver_kwargs)
        # compute clusters
        print_elapsed(_st)
        _st = time()
        print('      cluster predictions:', end=' ')
        if method.endswith('ncut'):
            predictions = (eigvecs[:, n_components] > 0).astype(int)
        else:
            km = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                        random_state=random_state)
            last_column = n_components + n_clusters
            predictions = km.fit_predict(eigvecs[:, n_components:last_column])
        print_elapsed(_st)
        centers = compute_medioids(data, predictions)
        residuals = data - centers[predictions]
        return (predictions, residuals, centers, eigvals, eigvecs)
    else:
        raise ValueError('unknown clustering method "{}"'.format(method))


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
training_data = epochs_cat[train_mask]
df = full_df.loc[train_mask].copy()
if single_subject:
    subj_one = df['subj'] == 1
    data = training_data[subj_one].copy()
    df = df[subj_one]
else:
    data = training_data.copy()
if conserve_memory:
    del (epochs, epochs_cat, training_data, train_mask, test_mask,
         validation_mask, full_df)

# do the splits
cluster_type = (' spectral' if cluster_method in
                ('precomputed', 'sparse_graph', 'dense_graph',
                 'sparse_ncut', 'dense_ncut') else '')
cluster_centers = np.zeros((n_iterations, n_clusters, data.shape[-1]))
eigvals = dict()
eigvecs = dict()
print('running {}{} clustering'.format(cluster_method, cluster_type))
for _iter in range(n_iterations):
    print('  iteration {}:'.format(_iter))
    colnum = 'split_{}'.format(_iter)
    split_kwargs = dict(n_jobs=n_jobs, n_clusters=n_clusters,
                        method=cluster_method, gamma=gamma)
    if (cluster_method.startswith('sparse') or
            cluster_method.startswith('dense')):
        (predictions, residuals, centers, evals,
         evecs) = split_and_resid(data, **split_kwargs)
        eigvals[colnum] = evals
        eigvecs[colnum] = evecs
    else:
        predictions, residuals, centers = split_and_resid(data, **split_kwargs)
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
    results['{}_match'.format(colnum)] = (truepos[featcol] +
                                          trueneg[featcol]) / df.shape[0]
    results['{}_mismatch'.format(colnum)] = (falsepos[featcol] +
                                             falseneg[featcol]) / df.shape[0]
print_elapsed(_st)

outvars = dict(df=df, centers=centers, results=results)
if cluster_method.startswith('sparse') or cluster_method.startswith('dense'):
    outvars.update(dict(eigvals=eigvals, eigvecs=eigvecs))

suffix = ('-{}_neighbors'.format(n_neighbors) if cluster_method in
          ('spectral', 'sparse_graph') else '')
out_fname = 'cluster-results-dss_{}-{}_{}{}.npz'.format(use_n_dss_channels,
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
