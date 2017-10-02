#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions.
"""
# @author: drmccloy
# Created on Tue Aug  9 10:58:22 2016
# License: BSD (3-clause)

import numpy as np
import scipy.cluster.hierarchy as hy


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
    from mne import BaseEpochs
    if isinstance(data, BaseEpochs):
        """
        if trial_types is None:
            trial_types = data.events[:, -1]
            trial_dict = {v: k for k, v in data.event_id.items()}
        """
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
        '''
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
            unbiased_power = np.array([_power(data_cov, dss)
                                       for dss in dss_operator])
            biased_power = np.array([_power(bias, dss) for bias, dss in
                                     zip(bias_cov, dss_operator)])
            results.extend([unbiased_power, biased_power])
        '''
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


def time_domain_pca(epochs, max_components=None):
    """ NORMAL WAY
    time_cov = np.sum([trial.T @ trial for trial in epochs], axis=0)
    eigval, eigvec = pca(time_cov, max_components=max_components, thresh=1e-6)
    W = np.sqrt(1 / eigval)  # whitening diagonal
    epochs = np.array([trial @ eigvec * W[np.newaxis, :] for trial in epochs])
    """
    time_cov = np.einsum('hij,hik->jk', epochs, epochs)
    eigval, eigvec = pca(time_cov, max_components=max_components, thresh=1e-6)
    W = np.diag(np.sqrt(1 / eigval))  # diagonal time-PCA whitening matrix
    epochs = np.einsum('hij,jk,kk->hik', epochs, eigvec, W)
    return epochs


def print_elapsed(start_time, end=' sec.\n'):
    from time import time
    print(np.round(time() - start_time, 1), end=end)


def _eer(probs, thresholds, pos, neg):
    # predictions
    guess_pos = probs[np.newaxis, :] >= thresholds[:, np.newaxis]
    guess_neg = np.logical_not(guess_pos)
    # false pos / false neg
    false_pos = np.logical_and(guess_pos, neg)
    false_neg = np.logical_and(guess_neg, pos)
    # false pos/neg rates for each threshold step (ignore div-by-zero warnings)
    err_state = np.seterr(divide='ignore')
    false_pos_rate = false_pos.sum(axis=1) / neg.sum()
    false_neg_rate = false_neg.sum(axis=1) / pos.sum()
    np.seterr(**err_state)
    # get rid of infs and zeros
    false_pos_rate[false_pos_rate == np.inf] = 1e9
    false_neg_rate[false_neg_rate == np.inf] = 1e9
    false_pos_rate[false_pos_rate == -np.inf] = -1e9
    false_neg_rate[false_neg_rate == -np.inf] = -1e9
    false_pos_rate[false_pos_rate == 0.] = 1e-9
    false_neg_rate[false_neg_rate == 0.] = 1e-9
    # FPR / FNR ratio
    ratios = false_pos_rate / false_neg_rate
    reverser = -1 if np.any(np.diff(ratios) < 0) else 1
    # find crossover point
    ix = np.searchsorted(ratios[::reverser], v=1.)
    closest_threshold_index = len(ratios) - ix if reverser < 0 else ix
    # check for convergence
    converged = np.isclose(ratios[closest_threshold_index], 1.)
    # return EER estimate
    eer = np.max([false_pos_rate[closest_threshold_index],
                  false_neg_rate[closest_threshold_index]])
    return closest_threshold_index, converged, eer


def _EER_score_threshold(estimator, X, y):
    # higher return values are better than lower return values
    probs = estimator.predict_proba(X)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    converged = False
    threshold = -1
    # ground truth
    pos = np.array(y, dtype=bool)
    neg = np.logical_not(pos)
    while not converged:
        ix, converged, eer = _eer(probs, thresholds, pos, neg)
        old_threshold = threshold
        threshold = thresholds[ix]
        converged = converged or np.isclose(threshold, old_threshold)
        low = (ix - 1) if ix > 0 else ix
        high = low + 1
        thresholds = np.linspace(thresholds[low], thresholds[high], 101)
    return (1 - eer), threshold


def EER_score(estimator, X, y):
    """EER scoring function
    estimator: sklearn.SVC
        sklearn estimator
    X: np.ndarray
        training data
    y: np.ndarray
        training labels
    """
    return _EER_score_threshold(estimator, X, y)[0]


def EER_threshold(clf, X, y, return_eer=False):
    """Get EER threshold

    clf: sklearn.GridSearchCV
        the fitted crossval object
    X: np.ndarray
        training data
    y: np.ndarray
        training labels
    """
    estimator = clf.estimator
    estimator.C = clf.best_params_['C']
    estimator.gamma = clf.best_params_['gamma']
    estimator.fit(X, y)
    score, threshold = _EER_score_threshold(estimator, X, y)
    result = (threshold, (1 - score)) if return_eer else threshold
    return result


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
    feats.columns = [cn.split('-')[0] for cn in feats.columns]  # flat-plain -> flat # noqa
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


def matrix_row_column_correlation(matrix):
    '''compute correlation between rows and columns of a matrix. Yields a
    measure of diagonality that ranges from 1 for diagonal matrix, through 0
    for a uniform matrix, to -1 for a matrix that is non-zero only at the
    off-diagonal corners. See https://math.stackexchange.com/a/1393907 and
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
    '''
    A = np.asarray(matrix)
    d = A.shape[0]
    assert (A.ndim == 2) and (d == A.shape[1])
    ones = np.ones(d)
    ix = np.arange(1, d + 1)  # row/column indices
    mass = A.sum()            # total mass of matrix
    rw = np.outer(ix, ones)   # row weights
    cw = np.outer(ones, ix)   # column weights
    rcw = np.outer(ix, ix)    # row * column weights

    # BROADCASTING METHOD                          # LINALG ALTERNATIVE
    sum_x = np.sum(rw * A)                         # ix @ A @ ones
    sum_y = np.sum(cw * A)                         # ones @ A @ ix
    sum_xy = np.sum(rw * cw * A)                   # ix @ A @ ix
    sum_xsq = np.sum(np.outer(ix ** 2, ones) * A)  # (ix ** 2) @ A @ ones
    sum_ysq = np.sum(np.outer(ones, ix ** 2) * A)  # ones @ A @ (ix ** 2)
    numerator = mass * sum_xy - sum_x * sum_y
    denominator = (np.sqrt(mass * sum_xsq - (sum_x ** 2)) *
                   np.sqrt(mass * sum_ysq - (sum_y ** 2)))
    return numerator / denominator


def _symmetric_kl_divergence(u, v, base=np.e):
    from scipy.stats import entropy
    return entropy(u, v, base=base) + entropy(v, u, base=base)


def _dist(matrix, metric=_symmetric_kl_divergence, fixup=True):
    '''pdist wrapper'''
    from scipy.spatial.distance import pdist
    dists = pdist(matrix, metric=metric)
    if fixup:
        valid = np.where(np.isfinite(dists))
        eps = np.finfo(dists.dtype).eps
        # make near-0 values actually 0 (or else tiny negatives may sneak in)
        dists[np.abs(dists) < eps] = 0.
        # make ∞ finite, but at least 1 order of magnitude bigger than the
        # largest finite value
        exponent = np.ceil(np.log10(dists[valid].max() + eps)) + 1
        dists[np.where(np.isinf(dists))] = 10 ** exponent
    return dists


def optimal_leaf_ordering(matrix, metric=_symmetric_kl_divergence):
    '''performs optimal leaf ordering on the rows and columns of a matrix'''
    results = dict(dendrograms=dict(), linkages=dict())
    for key, mat in dict(row=matrix, col=matrix.T).items():
        dists = _dist(mat, metric)
        z = hy.linkage(dists, optimal_ordering=True)
        dg = hy.dendrogram(z, no_plot=True)
        # make dendrogram output play nice with YAML
        for coord in ['icoord', 'dcoord']:
            dg[coord] = np.array(dg[coord], dtype=float).tolist()
        results['dendrograms'][key] = dg
        results['linkages'][key] = z
    return results


def optimal_matrix_diag(matrix, metric=_symmetric_kl_divergence):
    '''optimizes diagonality of a matrix'''
    raise NotImplementedError
    '''
    from scipy.spatial.distance import pdist, squareform
    matrix = np.asarray(joint_prob)
    dists = squareform(_dist(matrix, metric))
    dists[np.diag_indices_from(dists)] = 1e9
    ranked = np.argsort(dists, axis=None)
    neighbors = np.argsort(dists, axis=1)
    '''


def plot_dendrogram(dg, orientation='top', ax=None, no_labels=False,
                    leaf_font_size=None, leaf_rotation=None,
                    contraction_marks=None, above_threshold_color='b'):
    '''wrapper for scipy's (private) dendrogram plotting function'''
    from scipy.cluster.hierarchy import _plot_dendrogram
    defaults = dict(p=30, n=len(dg['ivl']), mh=z[:, 2].max(),
                    orientation=orientation, ax=ax, no_labels=no_labels,
                    leaf_font_size=leaf_font_size, leaf_rotation=leaf_rotation,
                    contraction_marks=contraction_marks,
                    above_threshold_color=above_threshold_color)
    dg.update(defaults)
    del dg['leaves']
    _plot_dendrogram(**dg)


def plot_confmat(df, ax=None, origin='upper', norm=None, cmap=None, title='',
                 xlabel='', ylabel='', **kwargs):
    from os.path import join
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    with plt.style.context(join('params', 'matplotlib-style-confmats.yaml')):
        ax.imshow(df, origin=origin, norm=norm, cmap=cmap)
        # garnish
        ax.set_xticks(np.arange(df.shape[1])[1::2], minor=False)
        ax.set_xticks(np.arange(df.shape[1])[::2], minor=True)
        ax.set_yticks(np.arange(df.shape[0])[1::2], minor=False)
        ax.set_yticks(np.arange(df.shape[0])[::2], minor=True)
        ax.set_xticklabels(df.columns[1::2], minor=False)
        ax.set_xticklabels(df.columns[::2], minor=True)
        ax.set_yticklabels(df.index[1::2], minor=False)
        ax.set_yticklabels(df.index[::2], minor=True)
        # annotate
        if len(title):
            ax.set_title(title)
        if len(xlabel):
            ax.set_xlabel(xlabel)
        if len(ylabel):
            ax.set_ylabel(ylabel)
    return ax


def plot_consonant_shape(df, ax=None, title='', xlabel='', ylabel='',
                         **kwargs):
    from os.path import join
    import matplotlib.pyplot as plt
    import pandas as pd
    rowsort = pd.DataFrame(np.sort(df, axis=1)[:, ::-1], index=df.index)
    sorted_df = rowsort.sort_values(by=0, axis=0)[::-1].T
    style_context = join('params', 'matplotlib-style-numerous-lineplots.yaml')
    with plt.style.context(style_context):
        if ax is None:
            fig, ax = plt.subplots(**kwargs)
        sorted_df.plot(ax=ax, legend=False)
        ax.set_xticks(np.arange(23))
        ax.set_xticklabels(np.arange(1, 24))
        ax.set_yticks([])
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        last_y = 1e3
        x = -0.1
        for ix, label in enumerate(sorted_df.columns):
            y = sorted_df.iloc[0, ix]
            far_enough = last_y - y > 0.003
            x = (x - 0.5) if not far_enough else -0.1
            if far_enough:
                last_y = y
            c = cycle[ix % len(cycle)]
            ax.text(x, y, label, ha='right', va='center', color=c)
        # annotate
        if len(title):
            ax.set_title(title)
        if len(xlabel):
            ax.set_xlabel(xlabel)
        if len(ylabel):
            ax.set_ylabel(ylabel)
    return ax
