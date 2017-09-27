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
import os.path as op
import numpy as np
import pandas as pd

np.set_printoptions(precision=4, linewidth=160)
pd.set_option('display.width', 250)

# BASIC FILE I/O
paramdir = 'params'

# LOAD PARAMS FROM YAML
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    features = analysis_params['features']
    feature_fnames = analysis_params['feature_fnames']
    feature_mappings = analysis_params['feature_mappings']
    canonical_phone_order = analysis_params['canonical_phone_order']

master_feat_sys = pd.DataFrame()
for feat_sys, fname in feature_fnames.items():
    table = pd.read_csv(op.join(paramdir, fname), sep='\t', comment='#',
                        index_col=0)
    colnames = [colname.split('-')[0] for colname in table.columns]
    inverse_mapping = {v: k for k, v in feature_mappings[feat_sys].items()}
    table.columns = [inverse_mapping[colname] for colname in colnames]
    master_feat_sys = pd.concat([master_feat_sys, table], axis=1)

errors = list()
for feat in features:
    df = master_feat_sys[feat]
    if df.ndim == 1:  # no duplicates
        continue
    if not df.T.duplicated(keep=False).all():
        print(feat)
        errors.append(feat)

if not len(errors):
    cols = np.logical_not(master_feat_sys.columns.duplicated(keep='first'))
    rows = canonical_phone_order['eng']
    master_feat_sys = master_feat_sys.loc[rows, cols].astype(float)
    master_feat_sys.to_csv(op.join(paramdir, 'all-features.tsv'), sep='\t')
