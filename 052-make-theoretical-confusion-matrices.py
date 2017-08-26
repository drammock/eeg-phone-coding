#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-theoretical-confusion-matrices.py'
===============================================================================

This script assumes constant phonological feature-level error rate, and makes
matrices of resulting phone confusion probabilities.  The error rates that are
simulated are read from the config file (actually, the config file specifies
accuracies, which are 1. - error rates).
"""
# @author: drmccloy
# Created on Thu Aug 24 16:00:58 PDT 2017
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
    feature_systems = analysis_params['feature_systems']
    canonical_phone_order = analysis_params['canonical_phone_order']
    subj_langs = analysis_params['subj_langs']
    accuracies = analysis_params['theoretical_accuracies']
del analysis_params

# load features
ground_truth = pd.read_csv(op.join(paramdir, feature_sys_fname), sep='\t',
                           index_col=0, comment='#')
# passing dtype=float to `read_csv` doesn't work when index col. is strings
ground_truth = ground_truth.astype(float)
ground_truth.index.name = 'ipa_out'
ground_truth.columns.name = 'features'

# make theoretical "subjects" where all features are equipotent
subjects = {str(accuracy): accuracy for accuracy in accuracies}
eng_phones = canonical_phone_order['eng']

# loop over "subjects"
for subj_code, acc in subjects.items():

    # loop over languages (key "theory" lists all 5 langs)
    for lang in subj_langs['theory']:
        this_phones = canonical_phone_order[lang]

        # loop over feature systems
        for feat_sys, feats in feature_systems.items():
            # ground truth for just the features in this feature system
            this_gr_truth = ground_truth[feats]
            # get rid of engma
            this_gr_truth = this_gr_truth.loc[eng_phones]

            # generate uniform accuracy matrices
            index = pd.Index(eng_phones, name='ipa_out')
            cols = pd.Index(feats, name='features')
            acc_df = pd.DataFrame(data=acc, index=index, columns=cols)

            # make 3d array of accuracy. Each feature plane of shape (ipa_in,
            # ipa_out) has a uniform value corresponding to the accuracy for
            # that feature (and in this case, accuracy for all features is the
            # same).
            acc_3d = pd.Panel({p: acc_df for p in this_phones},
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

            # intersect feats_in with feats_out to get boolean feature_match
            # array. Where features match, insert the accuracy for that
            # feature. Where they mismatch, insert 1. - accuracy.
            feat_mismatch = np.logical_xor(feats_in, feats_out)
            indices = np.where(feat_mismatch)
            prob_3d = acc_3d.copy()
            prob_3d.values[indices] = 1. - prob_3d.values[indices]

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
            args = [lang, feat_sys, subj_code]
            out_fname = 'theoretical-confusion-matrix-{}-{}-{}.tsv'.format(*args)
            joint_prob.to_csv(op.join(outdir, out_fname), sep='\t')
