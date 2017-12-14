#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-single-feature-confmats.py'
===============================================================================

This script converts a feature-level error rate (from a binary phonological
feature classifier) into a matrix of phone confusion probabilities.
"""
# @author: drmccloy
# Created on Wed Dec 13 15:27:52 PST 2017
# License: BSD (3-clause)

import yaml
from os import mkdir
import numpy as np
import pandas as pd
import os.path as op
from aux_functions import merge_features_into_df

# BASIC FILE I/O
paramdir = 'params'
indir = 'processed-data'
outdir = op.join(indir, 'single-feat-confmats')
feature_sys_fname = 'all-features.tsv'
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
    features = analysis_params['features']
    feature_systems = analysis_params['feature_systems']
    canonical_phone_order = analysis_params['canonical_phone_order']
    subj_langs = analysis_params['subj_langs']
    skip = analysis_params['skip']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
del analysis_params

# file naming variables
sfn = 'nan' if sparse_feature_nan else 'nonan'

# load the trial params
df_cols = ['subj', 'talker', 'syll', 'train']
df_types = dict(subj=int, talker=str, syll=str, train=bool)
df = pd.read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
                 usecols=df_cols, dtype=df_types)
df = merge_features_into_df(df, paramdir, feature_sys_fname)
df = df[(df_cols + ['lang', 'ascii', 'ipa'] + features)]

# load features (already merged into df, but useful to have separately)
ground_truth = pd.read_csv(op.join(paramdir, feature_sys_fname), sep='\t',
                           index_col=0, comment='#')
# passing dtype=float to `read_csv` doesn't work when index col. is strings
ground_truth = ground_truth.astype(float)
ground_truth.index.name = 'ipa_out'
ground_truth.columns.name = 'features'

# load equal error rates (EERs)
eers = pd.read_csv(op.join(indir, 'eers.tsv'), sep='\t', index_col=0)
eng_phones = canonical_phone_order['eng']

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''

# loop over subjects
for subj_code in subjects:
    if subj_code in skip:
        continue

    # loop over features
    for feat in features:
        this_eer = eers.loc[feat, subj_code]
        this_ground_truth = ground_truth[feat].loc[eng_phones]

        # (following is a glorified XOR that preserves NaNs)
        # add 1 to feature presence/absence values; outer product of
        # 1 or 4 -> matching feat vals
        # 2 -> mismatched feat vals
        # NaN -> unspecified (sparse) feat val
        match = np.outer(this_ground_truth + 1, this_ground_truth + 1)
        match = pd.DataFrame(match, index=this_ground_truth.index,
                             columns=this_ground_truth.index)

        # sparse_value = np.nan if sparse_feature_nan else 0.5
        # perform the XOR and map False to eer; True to 1 - eer
        def map_eer(x):
            if np.isnan(x):
                return np.nan  # don't put sparse_value yet; do when plotting
            elif x == 2:
                return this_eer
            else:
                return 1. - this_eer
        confmat = match.applymap(map_eer)
        # (end of glorified XOR that preserves NaNs)

        # save unordered confusion matrix
        args = [sfn, cv + nc + feat, subj_code]
        fname = 'eer-confusion-matrix-{}-eng-{}-{}.tsv'.format(*args)
        confmat.to_csv(op.join(outdir, fname), sep='\t')
