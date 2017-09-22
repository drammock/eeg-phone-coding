#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-confusion-matrices.py'
===============================================================================

This script converts phone-level error rates (from a bank of binary
phonological feature classifiers) into a matrix of phone confusion
probabilities.
"""
# @author: drmccloy
# Created on Tue Jun 06 13:13:09 2017
# License: BSD (3-clause)

import yaml
import numpy as np
import pandas as pd
import os.path as op
from os import mkdir
from aux_functions import merge_features_into_df

np.set_printoptions(precision=6, linewidth=160)
pd.set_option('display.width', 140)

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
            classifications = list()
            this_gr_truth = ground_truth[feats]
            # get rid of engma
            this_gr_truth = this_gr_truth.loc[canonical_phone_order['eng']]

            # loop over features
            for feat in feats:
                args = [lang, cv + nc + feat, subj_code]
                fname = 'classifier-probabilities-{}-{}-{}.tsv'.format(*args)
                # NB: columns are [-feat, +feat, feat, lang], where the first
                # two are classifier probabilities, and "feat" is the binary
                # classification based on the EER threshold for that feature
                fpath = op.join(indir, subj_code, fname)
                kwargs = dict(sep='\t', index_col=0, usecols=['ipa', feat],
                              dtype={'ipa': str, feat: int})
                classifications.append(pd.read_csv(fpath, **kwargs))
            # convert to DF; put cols in same order as in this_gr_truth
            classifications = pd.concat(classifications, axis='columns')
            classif_by_phone = classifications.groupby(classifications.index)
            # taking mean of 0s/1s yields (empirical) probability that a
            # classifier thought its feature was present for a given phone
            prob_by_phone = classif_by_phone.mean()
            prob_by_phone = prob_by_phone.loc[this_phones, feats]  # sort
            prob_by_phone.index.name = 'ipa_in'
            prob_by_phone.columns.name = 'features'

            # expand to 3D (classif_prob x ground_truth x features)
            prob_3d = pd.Panel({p: prob_by_phone for p in this_gr_truth.index},
                               items=this_gr_truth.index).swapaxes(0, 1)
            truth_3d = pd.Panel({p: this_gr_truth for p in prob_by_phone.index
                                 }, items=prob_by_phone.index)
            # make sure we did the swapaxes correctly
            assert np.array_equal(prob_by_phone.T, prob_3d.iloc[:, 0])
            assert np.allclose(this_gr_truth, truth_3d.iloc[0],
                               equal_nan=True)

            # invert probabilities where feature shouldn't be active. This
            # converts prob_3d from an array of probabilities that the
            # classifier **thought the feature was present** into an array of
            # probabilities that the classifier **was correct**
            mask = np.where(np.logical_not(truth_3d))
            prob_3d.values[mask] = 1. - prob_3d.values[mask]

            # handle feature values that are "sparse" in this feature system
            sparse_mask = np.where(np.isnan(truth_3d))
            sparse_value = np.nan if sparse_feature_nan else 0.5
            prob_3d.values[sparse_mask] = sparse_value

            # collapse across features to compute joint probabilities
            axis = [x.name for x in prob_3d.axes].index('features')
            ''' this one-liner can be numerically unstable, use three-liner
            joint_prob = prob_3d.prod(axis=axis, skipna=True).swapaxes(0, 1)
            '''
            log_prob_3d = (-1. * prob_3d.apply(np.log))
            joint_log_prob = (-1. * log_prob_3d.sum(axis=axis)).swapaxes(0, 1)
            joint_prob = joint_log_prob.apply(np.exp)
            # save
            args = [sfn, lang, cv + nc + feat_sys, subj_code]
            out_fname = 'phone-confusion-matrix-{}-{}-{}-{}.tsv'.format(*args)
            joint_prob.to_csv(op.join(outdir, out_fname), sep='\t')
