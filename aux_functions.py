#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:58:22 2016

@author: drmccloy
"""

import warnings
import numpy as np
import networkx as nx
from time import time
from scipy.sparse import linalg, csr_matrix
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_laplacian
from mne import BaseEpochs

def pca(cov, max_components=None, thresh=0):
    """Perform PCA decomposition from a covariance matrix

    Parameters
    ----------
    cov : array-like
        Covariance matrix
    max_components : int | None
        Maximum number of components to retain after decomposition. ``None``
        (the default) keeps all suprathreshold components (see ``thresh``).
    thresh : float | None
        Threshold (relative to the largest component) above which components
        will be kept. The default keeps all non-zero values; to keep all
        values, specify ``thresh=None`` and ``max_components=None``.

    Returns
    -------
    eigval : array
        1-dimensional array of eigenvalues.
    eigvec : array
        2-dimensional array of eigenvectors.
    """

    if thresh is not None and (thresh > 1 or thresh < 0):
        raise ValueError('Threshold must be between 0 and 1 (or None).')
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.abs(eigval)
    sort_ix = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, sort_ix]
    eigval = eigval[sort_ix]
    if max_components is not None:
        eigval = eigval[:max_components]
        eigvec = eigvec[:, :max_components]
    if thresh is not None:
        suprathresh = np.where(eigval / eigval.max() > thresh)[0]
        eigval = eigval[suprathresh]
        eigvec = eigvec[:, suprathresh]
    return eigval, eigvec


def dss(data, trial_types=None, pca_max_components=None, pca_thresh=0,
        bias_max_components=None, bias_thresh=0, norm=False,
        return_data=False, return_power=False):
    if isinstance(data, BaseEpochs):
        if trial_types is None:
            trial_types = data.events[:, -1]
            trial_dict = {v: k for k, v in data.event_id.items()}
        data = data.get_data()
    # norm each channel's time series
    if norm:
        channel_norms = np.linalg.norm(data, ord=2, axis=-1)
        data = data / channel_norms[:, :, np.newaxis]
    # PCA across channels
    data_cov = np.einsum('hij,hkj->ik', data, data)
    data_eigval, data_eigvec = pca(data_cov, max_components=pca_max_components,
                                   thresh=pca_thresh)
    # diagonal data-PCA whitening matrix:
    W = np.diag(np.sqrt(1 / data_eigval))
    """
    # make sure whitening works
    white_data = W @ data_eigvec.T @ data
    white_data_cov = np.einsum('hij,hkj->ik', white_data, white_data)
    assert np.allclose(white_data_cov, np.eye(white_data_cov.shape[0]))
    """
    # code path for computing separate bias for each condition
    if trial_types is not None:
        raise NotImplementedError
        bias_cov_dict = dict()
        for tt in trial_dict.keys():
            indices = np.where(trial_types == tt)
            evoked = data[indices].mean(axis=0)
            bias_cov_dict[tt] = np.einsum('hj,ij->hi', evoked, evoked)
        # compute bias rotation matrix
        bias_cov = np.array([bias_cov_dict[tt] for tt in trial_types])
        biased_white_eigvec = np.einsum('ii,hi,jhk,km,mm->jim', W,
                                        data_eigvec, bias_cov, data_eigvec, W)
        eigs = [pca(bwe, max_components=bias_max_components,
                    thresh=bias_thresh) for bwe in biased_white_eigvec]
        bias_eigval = np.array([x[0] for x in eigs])
        bias_eigvec = np.array([x[1] for x in eigs])
        # compute DSS operator
        """ THE NORMAL WAY
        dss_mat = np.array([data_eigvec @ W @ bias for bias in bias_eigvec])
        dss_normalizer = 1 / np.sqrt([np.diag(dss.T @ data_cov @ dss)
                                      for dss in dss_mat])
        dss_operator = np.array([dss @ np.diag(norm) for dss, norm
                                 in zip(dss_mat, dss_normalizer)])
        """
        dss_mat = np.einsum('ij,jj,hjk->hik', data_eigvec, W, bias_eigvec)
        dss_normalizer = 1 / np.sqrt(np.einsum('hij,ik,hkj->hj',
                                               dss_mat, data_cov, dss_mat))
        dss_operator = np.einsum('hij,hj->hij', dss_mat, dss_normalizer)
        results = [dss_operator]
        # apply DSS to data
        if return_data:
            data_dss = np.einsum('hij,hik->hkj', data, dss_operator)
            results.append(data_dss)
        if return_power:
            unbiased_power = np.array([_power(data_cov, dss) for dss in dss_operator])
            biased_power = np.array([_power(bias, dss) for bias, dss in zip(bias_cov, dss_operator)])
            results.extend([unbiased_power, biased_power])
    else:
        # compute bias rotation matrix
        evoked = data.mean(axis=0)
        bias_cov = np.einsum('hj,ij->hi', evoked, evoked)
        biased_white_eigvec = W @ data_eigvec.T @ bias_cov @ data_eigvec @ W
        bias_eigval, bias_eigvec = pca(biased_white_eigvec,
                                       max_components=bias_max_components,
                                       thresh=bias_thresh)
        # compute DSS operator
        dss_mat = data_eigvec @ W @ bias_eigvec
        dss_normalizer = 1 / np.sqrt(np.diag(dss_mat.T @ data_cov @ dss_mat))
        dss_operator = dss_mat @ np.diag(dss_normalizer)
        results = [dss_operator]
        # apply DSS to data
        if return_data:
            data_dss = np.einsum('hij,ik->hkj', data, dss_operator)
            results.append(data_dss)
        if return_power:
            unbiased_power = _power(data_cov, dss_operator)
            biased_power = _power(bias_cov, dss_operator)
            results.extend([unbiased_power, biased_power])
    return tuple(results)


def _power(cov, dss):
    return np.sqrt(((cov @ dss) ** 2).sum(axis=0))


def time_domain_pca(epochs):
    time_cov = np.sum([np.dot(trial.T, trial) for trial in epochs], axis=0)
    eigval, eigvec = pca(time_cov, max_components=None, thresh=1e-6)
    W = np.sqrt(1 / eigval)  # whitening diagonal
    epochs = np.array([np.dot(trial, eigvec) * W[np.newaxis, :]
                       for trial in epochs])
    return epochs


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


def new_classifier(clf_type, **kwargs):
    if clf_type == 'lda':
        clf = LDA(solver='svd', **kwargs)
    elif clf_type == 'svm':
        clf = SVC(**kwargs)
    else:
        raise ValueError('unrecognized value for classifier type (clf_type)')
    return clf


def train_classifier(classifier, data, labels, msg):
    # train classifier
    print('  {}'.format(msg), end=': ')
    _st = time()
    classifier.fit(X=data, y=labels)
    # handle class names and dtypes for structured array
    dtype_names = ['{}{}'.format(['-', '+'][val], msg)
                   for val in np.unique(labels)]
    dtype_formats = [float] * np.unique(labels).size
    model_dtype_dict = dict(names=dtype_names, formats=dtype_formats)
    print_elapsed(_st)
    return classifier, model_dtype_dict


def test_classifier(classifier, data, dtypes):
    prob = classifier.predict_proba(data)
    return np.array([tuple(x) for x in prob], dtype=dtypes)


def _eer(steps, threshold, converged, iteration, probs, truth):
    from numpy import logical_and as bool_and
    from numpy import logical_not as bool_not
    old_threshold = threshold
    iteration += 1
    preds = np.array([probs >= thresh for thresh in steps])
    trues = np.tile(truth.astype(bool), (steps.size, 1))
    falses = bool_not(trues)
    # false pos / false neg rates
    fpr = bool_and(falses, preds).sum(axis=1) / falses.sum(axis=1)
    fnr = bool_and(trues, bool_not(preds)).sum(axis=1) / trues.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress divide by zero warnings
        ratios = fpr / fnr
        if np.any(np.diff(ratios) < 0):  # also suppress inf-minus-inf warnings
            ratios = ratios[::-1]
            steps = steps[::-1]
    ix = np.searchsorted(ratios, v=1.)
    threshold = steps[ix]
    converged = (np.isclose(ratios[ix], 1.) or
                 np.isclose(threshold, old_threshold))
    steps = np.linspace(steps[ix - 1], steps[ix], 11)
    return steps, threshold, converged, iteration, fpr[ix]

def score_EER(estimator, X, y):
    probs = estimator.predict_proba(X)[:, 1]
    steps = np.linspace(0, 1, 11)
    converged = False
    iteration = 0
    threshold = -1
    while not converged:
        (steps, threshold, converged, iteration,
         eer) = _eer(steps, threshold, converged, iteration, probs, y)
    return 1 - eer


def find_EER_threshold(probs, truth):
    steps = np.linspace(0, 1, 11)
    converged = False
    iteration = 0
    threshold = -1
    while not converged:
        (steps, threshold, converged, iteration,
         eer) = _eer(steps, threshold, converged, iteration, probs, truth)
    return threshold, eer


def merge_features_into_df(df, paramdir, features_file):
    import json
    import os.path as op
    import pandas as pd
    # separate language from talker ID
    df['lang'] = df['talker'].map(lambda x: x[:3])
    eng_talkers = df['lang'] == 'eng'
    # LOAD FEATURES
    feats = pd.read_csv(op.join(paramdir, features_file), sep='\t',
                        index_col=0, comment='#')
    feats = feats.astype(float)  # passing dtype=float in reader doesn't work
    feats.columns = [cn.split('-')[0] for cn in feats.columns]  # flat-plain -> flat
    # abstract away from individual tokens
    df['ascii'] = df['syll'].transform(lambda x: x[:-2].replace('-', '_')
                                       if x.split('-')[-1] in ('0', '1', '2')
                                       else x.replace('-', '_'))
    # add IPA column
    with open(op.join(paramdir, 'ascii-to-ipa.json'), 'r') as _file:
        ipadict = json.load(_file)
    df['ipa'] = df['ascii'].map(ipadict)
    # revise IPA column for English (undo the strict phonetic coding of English
    # phones created during stimulus recording). The undoing is built into the
    # mapping in english-ascii-to-ipa.json
    with open(op.join(paramdir, 'english-ascii-to-ipa.json'), 'r') as _file:
        ipadict = json.load(_file)
    df.loc[eng_talkers, 'ipa'] = df.loc[eng_talkers, 'ascii'].map(ipadict)
    # ensure we have feature values for all the phonemes (at least for English)
    assert np.all(np.in1d(df.loc[eng_talkers, 'ipa'].unique(), feats.index))
    # merge feature columns. depending on feature set, this may yield some rows
    # that are all NaN (e.g., foreign phonemes with English-only feature sets)
    # or some feature values that are NaN (in cases of sparse feature matrices)
    feat_cols = feats.loc[df['ipa']].copy()
    feat_cols.reset_index(inplace=True, drop=True)
    assert np.allclose(df.index, feat_cols.index)
    df = pd.concat([df, feat_cols], axis=1)
    return df
