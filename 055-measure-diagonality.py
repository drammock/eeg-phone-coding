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
from aux_functions import matrix_row_column_correlation


# BASIC FILE I/O
paramdir = 'params'
# indir defined below, after loading YAML parameters
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
    use_ordered = analysis_params['sort_matrices']
    methods = analysis_params['methods']
    skip = analysis_params['skip']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
del analysis_params

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
ordered = 'ordered-' if use_ordered else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'
indir = op.join('processed-data', '{}confusion-matrices'.format(ordered))

# init container
matrix_diagonality = {m: None for m in methods}

# loop over methods (phone-level, feature-level-eer, uniform-error-simulations)
for method in methods:
    simulating = (method == 'theoretical')
    if simulating:
        subjects = {str(acc): acc for acc in accuracies}
    # init dataframe
    if use_ordered:
        matrix_diagonality[method] = dict()
    else:
        matrix_diagonality[method] = pd.DataFrame(data=np.nan, index=subjects,
                                                  columns=feature_systems)
    # loop over ordering types
    order_types = ('row-', 'col-', 'feat-') if use_ordered else ('',)
    for order_type in order_types:
        df = matrix_diagonality[method]
        if use_ordered:
            kwargs = dict(data=np.nan, index=subjects, columns=feature_systems)
            matrix_diagonality[method][order_type] = pd.DataFrame(**kwargs)
            df = df[order_type]
        # loop over subjects
        for subj_code in subjects:
            if subj_code in skip:
                df.drop(subj_code, inplace=True)
                continue
            # loop over feature systems
            for feat_sys in feature_systems:
                # load the data
                middle_arg = feat_sys if simulating else cv + nc + feat_sys
                args = [order_type, ordered, method, sfn, 'eng', middle_arg,
                        subj_code]
                fname = '{}{}{}-confusion-matrix-{}-{}-{}-{}.tsv'.format(*args)
                confmat = pd.read_csv(op.join(indir, fname), sep='\t',
                                      index_col=0)
                # compute diagonality
                df.loc[subj_code, feat_sys] = matrix_row_column_correlation(confmat)
        # save
        args = [order_type, ordered, sfn, method]
        fname = '{}{}matrix-diagonality-{}-{}.tsv'.format(*args)
        df.to_csv(op.join(outdir, fname), sep='\t')
