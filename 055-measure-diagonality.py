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


# LOAD PARAMS FROM YAML
paramdir = 'params'
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
    scheme = analysis_params['classification_scheme']
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

phone_level = scheme in ['pairwise', 'OVR', 'multinomial']

# FILE NAMING VARIABLES
cv = 'cvalign-' if align_on_cv else ''
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
ordered = 'ordered-' if use_ordered else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# BASIC FILE I/O
datadir = f'processed-data-{scheme}{trunc}'
outdir = op.join(datadir, 'matrix-correlations')
indir = op.join(datadir, '{}confusion-matrices'.format(ordered))
if not op.isdir(outdir):
    mkdir(outdir)

# init container
matrix_diagonality = {m: None for m in methods}

# loop over methods (phone-level, feature-level-eer, uniform-error-simulations)
for method in methods:
    if phone_level and method != 'eer':
        continue
    simulating = (method == 'theoretical')
    if simulating:
        subjects = {str(acc): acc for acc in accuracies}
    # init dataframe
    if use_ordered:
        matrix_diagonality[method] = dict()
    else:
        matrix_diagonality[method] = pd.DataFrame(data=np.nan,
                                                  index=subjects,
                                                  columns=feature_systems)
    # loop over ordering types
    order_types = ('row-', 'col-', 'feat-') if use_ordered else ('',)
    for order_type in order_types:
        df = matrix_diagonality[method]
        if phone_level and order_type != 'row-':
            continue
        if use_ordered:
            kwargs = dict(data=np.nan, index=subjects, columns=[])
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
                fsys = '' if phone_level else feat_sys
                middle_arg = fsys if simulating else cv + nc + fsys
                args = [order_type, ordered, method, sfn, 'eng', middle_arg,
                        subj_code]
                fname = '{}{}{}-confusion-matrix-{}-{}-{}{}.tsv'.format(*args)
                confmat = pd.read_csv(op.join(indir, fname), sep='\t',
                                      index_col=0)
                # compute diagonality
                column = scheme if phone_level else feat_sys
                df.loc[subj_code,
                       column] = matrix_row_column_correlation(confmat)
        # save
        args = [order_type, ordered, sfn, method]
        fname = '{}{}matrix-diagonality-{}-{}.tsv'.format(*args)
        df.to_csv(op.join(outdir, fname), sep='\t')
