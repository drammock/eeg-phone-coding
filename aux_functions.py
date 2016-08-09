#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:58:22 2016

@author: drmccloy
"""


def time_domain_pca(epochs):
    import numpy as np
    from mne_sandbox.preprocessing._dss import _pca
    time_cov = np.sum([np.dot(trial.T, trial) for trial in epochs], axis=0)
    eigval, eigvec = _pca(time_cov, max_components=None, thresh=1e-6)
    W = np.sqrt(1 / eigval)  # whitening diagonal
    epochs = np.array([np.dot(trial, eigvec) * W[np.newaxis, :]
                       for trial in epochs])
    return epochs


