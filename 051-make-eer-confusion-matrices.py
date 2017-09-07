#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-eer-confusion-matrices.py'
===============================================================================

This script converts feature-level error rates (from a bank of binary
phonological feature classifiers) into a matrix of phone confusion
probabilities.
"""
# @author: drmccloy
# Created on Tue Jun 06 13:13:09 2017
# License: BSD (3-clause)

import yaml
from os import mkdir
import numpy as np
import pandas as pd
import os.path as op
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import entropy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from aux_functions import merge_features_into_df

np.set_printoptions(precision=6, linewidth=160)
pd.set_option('display.width', 250)
plt.ion()

# BASIC FILE I/O
paramdir = 'params'
indir = 'processed-data'
outdir = op.join(indir, 'confusion-matrices')
feature_sys_fname = 'consonant-features-transposed-all-reduced.tsv'
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
del analysis_params

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
# add theoretical "subject" where all features are equipotent
eers['theory'] = 0.01
subjects['theory'] = -1  # dummy value
eng_phones = canonical_phone_order['eng']

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''

# loop over subjects
for subj_code in subjects:
    if subj_code in skip:
        continue

    # loop over languages
    for lang in subj_langs[subj_code]:
        this_phones = canonical_phone_order[lang]

        # loop over feature systems
        for feat_sys, feats in feature_systems.items():
            # ground truth for just the features in this feature system
            this_gr_truth = ground_truth[feats]
            # get rid of engma
            this_gr_truth = this_gr_truth.loc[eng_phones]

            # make 3d array of EERs. Each feature plane of shape (ipa_in,
            # ipa_out) has a uniform value corresponding to the EER for that
            # feature.
            eer = eers.loc[feats, subj_code]
            eer_df = pd.DataFrame({p: eer for p in this_gr_truth.index})
            eer_df = eer_df.T.loc[this_gr_truth.index, feats]
            eer_df.index.name, eer_df.columns.name = 'ipa_out', 'features'
            eer_3d = pd.Panel({p: eer_df for p in this_phones},
                              items=pd.Index(this_phones, name='ipa_in'))
            # make 3d arrays of feature values where true feature values are
            # repeated along orthogonal planes (i.e., feats_in.loc['p'] looks
            # like feats_out.loc[:, 'p'].T)
            feats_out = pd.Panel({p: this_gr_truth for p in this_phones},
                                 items=pd.Index(this_phones, name='ipa_in'))
            feats_in = pd.Panel({p: this_gr_truth.loc[this_phones]
                                 for p in this_gr_truth.index},
                                items=this_gr_truth.index).swapaxes(0, 1)
            feats_in.items.name = 'ipa_in'

            # intersect feats_in with feats_out to get boolean feature_match array
            feat_match = np.logical_not(np.logical_xor(feats_in, feats_out))
            """
            axis = [x.name for x in feat_match.axes].index('features')
            n_feat_match = match.sum(axis=axis)
            """

            # where features mismatch, insert the EER for that feature.
            # where they match, insert 1. - EER.
            prob_3d = eer_3d.copy()
            prob_3d.values[np.where(feat_match)] = 1. - prob_3d.values[np.where(feat_match)]

            # restore NaN values where features are undefined
            nan_mask = np.where(np.isnan(feats_out))
            prob_3d.values[nan_mask] = np.nan

            # collapse across features to compute joint probabilities
            axis = [x.name for x in prob_3d.axes].index('features')
            joint_prob = prob_3d.prod(axis=axis, skipna=True).swapaxes(0, 1)
            """ ALTERNATE WAY
            log_prob_3d = (-1. * prob_3d.apply(np.log))
            joint_log_prob = (-1. * log_prob_3d.sum(axis=axis)).swapaxes(0, 1)
            joint_prob = joint_log_prob.apply(np.exp)
            """

            # save unordered confusion matrix
            args = [lang, cv + nc + feat_sys, subj_code]
            out_fname = 'eer-confusion-matrix-{}-{}-{}.tsv'.format(*args)
            joint_prob.to_csv(op.join(outdir, out_fname), sep='\t')
