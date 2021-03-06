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
import numpy as np
import pandas as pd
import os.path as op
from os import mkdir
from aux_functions import merge_features_into_df

# LOAD PARAMS FROM YAML
paramdir = 'params'
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
    scheme = analysis_params['classification_scheme']
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

# FILE NAMING VARIABLES
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

phone_level = scheme in ['pairwise', 'OVR', 'multinomial']

# BASIC FILE I/O
indir = f'processed-data-{scheme}{trunc}'
outdir = op.join(indir, 'confusion-matrices')
feature_sys_fname = 'all-features.tsv'
if not op.isdir(outdir):
    mkdir(outdir)

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
fname = 'error-rates.tsv' if scheme == 'multinomial' else 'eers.tsv'
eers = pd.read_csv(op.join(indir, fname), sep='\t', index_col=0)
eng_phones = canonical_phone_order['eng']

# init container
confmats = dict()

# loop over subjects
for subj_code in subjects:
    if subj_code in skip:
        continue
    confmats[subj_code] = dict()

    # loop over languages
    for lang in subj_langs[subj_code]:

        # handle pairwise classifier data differently
        if phone_level:
            if lang != 'eng':
                continue
            # subset EERs to this subject
            this_eers = eers[subj_code]
            # make empty confmat
            confmat = pd.DataFrame(index=pd.Index(eng_phones, name='ipa_in'),
                                   columns=pd.Index(eng_phones,
                                                    name='ipa_out'),
                                   dtype=float, data=np.nan)
            if scheme == 'pairwise':
                # fill in off-diagonal values
                for contrast, eer in this_eers.iteritems():
                    phone_one, _, phone_two = contrast.split('_')
                    confmat.loc[phone_one, phone_two] = eer
                    confmat.loc[phone_two, phone_one] = eer
                # compute diagonal entries (mean accuracy of row; subtract
                # from 1 because other entries are error rate, not accuracy)
                means = np.nanmean(1. - confmat, axis=1)
                confmat.values[np.diag_indices(confmat.shape[0])] = means
            elif scheme == 'multinomial':
                fname = ('classifier-probabilities-{}-{}{}.tsv'
                         .format(lang, cv + nc, subj_code))
                fpath = op.join(indir, 'classifiers', subj_code, fname)
                probs = pd.read_csv(fpath, sep='\t', index_col=0)
                # clfs = probs[confmat.columns].apply(lambda x: x == x.max(),
                #                                     axis=1).astype(int)
                # assert np.all(clfs.sum(axis=1) == 1)
                confmat = probs.groupby(probs.index).mean()
            else:  # OVR
                probs = pd.DataFrame()
                clfs = pd.DataFrame()
                for phone in eng_phones:
                    fname = ('classifier-probabilities-{}-{}{}-{}.tsv'
                             .format(lang, cv + nc, phone, subj_code))
                    this_clf = pd.read_csv(op.join(indir, 'classifiers',
                                                   subj_code, fname),
                                           sep='\t', index_col='ipa')
                    probs[phone] = this_clf[phone]
                    clfs[phone] = this_clf['prediction']
                n_present = clfs.index.value_counts(sort=False)
                n_classif = clfs.groupby(clfs.index).sum()
                n_classif = n_classif[n_classif.index]  # order cols like rows
                confmat = n_classif / np.tile(n_present[:, None],
                                              (1, n_classif.shape[1]))
            assert np.all(np.isfinite(confmat))
            confmat = confmat[confmat.index]   # order cols like rows
            # put in dict
            confmats[subj_code] = confmat
            # save unordered confusion matrix
            args = [sfn, lang, cv + nc, subj_code]
            out_fname = 'eer-confusion-matrix-{}-{}-{}{}.tsv'.format(*args)
            confmat.to_csv(op.join(outdir, out_fname), sep='\t')
            continue

        this_phones = canonical_phone_order[lang]
        confmats[subj_code][lang] = dict()

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

            # intersect feats_in with feats_out -> boolean feature_match array
            feat_match = np.logical_not(np.logical_xor(feats_in, feats_out))

            # where features mismatch, insert the EER for that feature.
            # where they match, insert 1. - EER.
            prob_3d = eer_3d.copy()
            match_ix = np.where(feat_match)
            prob_3d.values[match_ix] = 1. - prob_3d.values[match_ix]

            # handle feature values that are "sparse" in this feature system
            sparse_mask = np.where(np.isnan(feats_out))
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
            confmats[subj_code][lang][feat_sys] = joint_prob

            # save unordered confusion matrix
            args = [sfn, lang, cv + nc + feat_sys, subj_code]
            out_fname = 'eer-confusion-matrix-{}-{}-{}-{}.tsv'.format(*args)
            joint_prob.to_csv(op.join(outdir, out_fname), sep='\t')

# compute across-subject averages
for feat_sys in feature_systems:
    if phone_level:
        these_confmats = confmats
    else:
        these_confmats = dict()
        for subj_code in subjects:
            if subj_code in skip:
                continue
            these_confmats[subj_code] = confmats[subj_code]['eng'][feat_sys]
    average_confmat = pd.Panel(these_confmats).mean(axis=0)
    middle_arg = '' if phone_level else '{}-'.format(feat_sys)
    args = [sfn, 'eng', cv + nc + middle_arg, 'average']
    out_fname = 'eer-confusion-matrix-{}-{}-{}{}.tsv'.format(*args)
    average_confmat.to_csv(op.join(outdir, out_fname), sep='\t')
    # don't need to loop over feature_systems if pairwise/OVR
    if phone_level:
        break
