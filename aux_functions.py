#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:58:22 2016

@author: drmccloy
"""

from __future__ import division, print_function
import numpy as np
import networkx as nx
from time import time
from scipy.sparse import linalg, csr_matrix
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_laplacian
from mne_sandbox.preprocessing._dss import _pca


def time_domain_pca(epochs):
    time_cov = np.sum([np.dot(trial.T, trial) for trial in epochs], axis=0)
    eigval, eigvec = _pca(time_cov, max_components=None, thresh=1e-6)
    W = np.sqrt(1 / eigval)  # whitening diagonal
    epochs = np.array([np.dot(trial, eigvec) * W[np.newaxis, :]
                       for trial in epochs])
    return epochs


def print_elapsed(start_time, end=' sec.\n'):
    print(np.round(time() - start_time, 1), end=end)


def find_EER_threshold(pred_class_one_prob, labels, string=''):
    steps = np.linspace(0, 1, 11)
    converged = False
    iteration = 0
    threshold = -1
    print('Finding EER thresholds{}: iteration'.format(string), end=' ')
    while not converged:
        old_threshold = threshold
        iteration += 1
        print(str(iteration), end=' ')
        preds = np.array([pred_class_one_prob >= thresh for thresh in steps])
        trues = np.tile(labels.astype(bool), (steps.size, 1))
        false_pos_rate = ((np.logical_not(trues) & preds).sum(axis=1) /
                          np.logical_not(trues).sum(axis=1))
        false_neg_rate = ((trues & np.logical_not(preds)).sum(axis=1) /
                          trues.sum(axis=1))
        ratios = (false_pos_rate / false_neg_rate)
        if np.isinf(ratios[0]) or ratios[0] > ratios[1]:
            ratios = ratios[::-1]
            steps = steps[::-1]
        ix = np.searchsorted(ratios, v=1)
        threshold = steps[ix]
        converged = (np.isclose(ratios[ix], 1.) or
                     np.isclose(threshold, old_threshold))
        steps = np.linspace(steps[ix - 1], steps[ix], 11)
    print()
    eer = false_pos_rate[ix]
    return threshold, eer


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


def split_and_resid(data, n_clusters=2, n_jobs=-2, random_state=0,
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
    elif method == 'tsne':
        clust = TSNE(n_components=n_clusters, random_state=random_state,
                     perplexity=n_neighbors)
        embedding = clust.fit_transform(data)
        return embedding
    elif method in ('spectral', 'precomputed'):
        if method == 'spectral':
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
