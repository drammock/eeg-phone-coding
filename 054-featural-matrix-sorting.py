#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'featural-matrix-sorting.py'
===============================================================================

This script uses classifier EERs to sort the rows and columns of the
confusion matrices.
"""
# @author: drmccloy
# Created on Fri Sep 22 15:13:27 PDT 2017
# License: BSD (3-clause)

import yaml
from os import mkdir
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, linewidth=160)
pd.set_option('display.width', 250)
plt.ion()

# BASIC FILE I/O
paramdir = 'params'
datadir = 'processed-data'
indir = op.join(datadir, 'confusion-matrices')
dgdir = op.join(datadir, 'dendrograms')
outdir = op.join(datadir, 'ordered-confusion-matrices')
for _dir in [outdir, dgdir]:
    if not op.isdir(_dir):
        mkdir(_dir)

# LOAD PARAMS FROM YAML
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    feature_systems = analysis_params['feature_systems']
    accuracies = analysis_params['theoretical_accuracies']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    canonical_phone_order = analysis_params['canonical_phone_order']
    feature_fnames = analysis_params['feature_fnames']
    subj_langs = analysis_params['subj_langs']
    methods = analysis_params['methods']
    skip = analysis_params['skip']
del analysis_params

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

# load EERs
eers = pd.read_csv(op.join(datadir, 'eers.tsv'), sep='\t', index_col=0)
eng_phones = canonical_phone_order['eng']

# load phone-feature matrix
featmat_fname = 'all-features.tsv'
featmat = pd.read_csv(op.join(paramdir, featmat_fname), sep='\t', index_col=0,
                      comment='#')

# loop over methods
for method in methods:
    _eers = eers.copy()
    _subjects = subjects.copy()
    simulating = (method == 'theoretical')
    if simulating:
        _subjects = {str(accuracy): accuracy for accuracy in accuracies}
        for acc in accuracies:
            _eers[str(acc)] = acc
        _eers = _eers[[str(acc) for acc in accuracies]]
    # loop over subjects
    for subj_code in _subjects:
        if subj_code in skip:
            continue
        key = 'theoretical' if simulating else subj_code
        # loop over languages
        for lang in subj_langs[key]:
            # loop over feature systems
            for feat_sys, feats in feature_systems.items():
                # load the data
                middle_arg = '' if simulating else cv + nc
                args = [method, sfn, lang, middle_arg + feat_sys, subj_code]
                fname = '{}-confusion-matrix-{}-{}-{}-{}.tsv'.format(*args)
                fpath = op.join(indir, fname)
                joint_prob = pd.read_csv(fpath, index_col=0, sep='\t')
                # subset the EER dataframe
                this_eers = _eers.loc[feats, subj_code]
                this_eers.sort_values(inplace=True)
                # subset the feature system
                this_featmat = featmat.loc[eng_phones, this_eers.index]
                this_featmat.sort_values(this_eers.index.tolist(),
                                         inplace=True)
                # perform optimal ordering of rows/columns
                col_ord = this_featmat.index.tolist()
                if lang == 'eng':
                    ordered_prob = joint_prob.loc[col_ord, col_ord]
                else:
                    ordered_prob = joint_prob.loc[:, col_ord]
                # save ordered matrix
                out = op.join(outdir, 'feat-ordered-' + fname)
                ordered_prob.to_csv(out, sep='\t')
