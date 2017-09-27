#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'find-redundant-features.py'
===============================================================================

This script finds features from different feature systems that have equivalent
definitions.
"""
# @author: drmccloy
# Created on Tue Aug 15 09:31:37 PDT 2017
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd

np.set_printoptions(precision=6, linewidth=130)
pd.set_option('display.width', 130)

paramdir = 'params'
analysis_param_file = op.join(paramdir, 'current-analysis-settings.yaml')
with open(analysis_param_file, 'r') as f:
    analysis_params = yaml.load(f)
    feature_fnames = analysis_params['feature_fnames']
    feature_systems = analysis_params['feature_systems']

dfs = list()
for feat_sys, fname in feature_fnames.items():
    df = pd.read_csv(op.join(paramdir, fname), sep='\t', comment='#',
                     index_col=0, skip_blank_lines=True)
    df = pd.concat([df], axis=1, keys=[feat_sys], names=['feature_system'])
    dfs.append(df)

dfs = pd.concat(dfs, axis=1)

results = dict()
for ix, col in enumerate(dfs.columns.tolist()):
    key = ','.join(col)
    result = np.where(np.all(dfs.values == dfs[col].values[:, np.newaxis], axis=0))
    # ignore equality to self
    result = (np.array(list(set(result[0]) - set([ix]))),)
    if len(result) and len(result[0]):
        results[key] = [','.join(x) for x in dfs.columns[result].tolist()]

for k, v in results.items():
    print('{}: {}'.format(k, v))
