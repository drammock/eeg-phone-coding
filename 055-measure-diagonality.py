#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'measure-diagonality.py'
===============================================================================

This script computes the diagonality of the confusion matrices, using a
generalization of the Pearson product-moment correlation formula for samples.
"""
# @author: drmccloy
# Created on Mon Aug 28 11:17:01 PDT 2017
# License: BSD (3-clause)

import yaml
from os import mkdir
import os.path as op
import numpy as np
import pandas as pd

np.set_printoptions(precision=6, linewidth=130)
pd.set_option('display.width', 130)


def matrix_row_column_correlation(mat):
    '''compute correlation between rows and columns of a matrix. Yields a
    measure of diagonality that ranges from 1 for diagonal matrix, through 0
    for a uniform matrix, to -1 for a matrix that is non-zero only at the
    off-diagonal corners. See https://math.stackexchange.com/a/1393907 and
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
    '''
    A = np.asarray(mat)
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


# BASIC FILE I/O
paramdir = 'params'
indir = op.join('processed-data', 'confusion-matrices')
outdir = op.join('processed-data', 'matrix-correlations')
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS FROM YAML
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    feature_systems = analysis_params['feature_systems']
    subj_langs = analysis_params['subj_langs']
    accuracies = analysis_params['theoretical_accuracies']
    methods = analysis_params['methods']
    skip = analysis_params['skip']
del analysis_params

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''

# init container
matrix_diagonality = {m: None for m in methods}

# loop over methods (phone-level, feature-level-eer, uniform-error-simulations)
for method in methods:
    simulating = (method == 'theoretical')
    if simulating:
        subjects = {str(acc): acc for acc in accuracies}
    # init dataframe
    matrix_diagonality[method] = pd.DataFrame(data=np.nan, index=subjects,
                                              columns=feature_systems)
    # loop over subjects
    for subj_code in subjects:
        if subj_code in skip:
            matrix_diagonality[method].drop(subj_code, inplace=True)
            continue
        # loop over feature systems
        for feat_sys in feature_systems:
            # load the data
            middle_arg = feat_sys if simulating else cv + nc + feat_sys
            args = [method, 'eng', middle_arg, subj_code]
            fname = '{}-confusion-matrix-{}-{}-{}.tsv'.format(*args)
            confmat = pd.read_csv(op.join(indir, fname), sep='\t', index_col=0)
            # compute diagonality
            matrix_diagonality[method].loc[subj_code, feat_sys] = \
                matrix_row_column_correlation(confmat)
    # save
    fname = op.join(outdir, 'matrix-diagonality-{}.tsv'.format(method))
    matrix_diagonality[method].to_csv(fname, sep='\t')
