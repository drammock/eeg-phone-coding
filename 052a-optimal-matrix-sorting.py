#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'optimal-matrix-sorting.py'
===============================================================================

This script uses "optimal leaf ordering" to sort the rows and columns of the
confusion matrices.
"""
# @author: drmccloy
# Created on Mon Sep  4 11:13:58 PDT 2017
# License: BSD (3-clause)

import yaml
from os import mkdir
import pandas as pd
import os.path as op
from aux_functions import optimal_leaf_ordering

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
    methods = analysis_params['methods']
    skip = analysis_params['skip']
    sparse_feature_nan = analysis_params['sparse_feature_nan']
    scheme = analysis_params['classification_scheme']
del analysis_params

phone_level = scheme in ['pairwise', 'OVR', 'multinomial']

# BASIC FILE I/O
datadir = 'processed-data-{}'.format(scheme)
indir = op.join(datadir, 'confusion-matrices')
dgdir = op.join(datadir, 'dendrograms')
outdir = op.join(datadir, 'ordered-confusion-matrices')
for _dir in [outdir, dgdir]:
    if not op.isdir(_dir):
        mkdir(_dir)

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''
sfn = 'nan' if sparse_feature_nan else 'nonan'

if phone_level:
    subjects.update(dict(average=0))
    for subj_code in subjects:
        # load the data
        args = ['eer', sfn, 'eng', cv + nc, subj_code]
        fname = '{}-confusion-matrix-{}-{}-{}{}.tsv'.format(*args)
        fpath = op.join(indir, fname)
        joint_prob = pd.read_csv(fpath, index_col=0, sep='\t')
        # TODO: refactor; rest of this block is duplicated below
        # perform optimal ordering of rows/columns
        olo = optimal_leaf_ordering(joint_prob)
        dendrograms, linkages = olo['dendrograms'], olo['linkages']
        row_ord = dendrograms['row']['leaves']
        col_ord = dendrograms['col']['leaves']
        ordered_prob = joint_prob.iloc[row_ord, col_ord]
        # save ordered matrix
        out = op.join(outdir, 'ordered-' + fname)
        ordered_prob.to_csv(out, sep='\t')
        row_ordered_prob = joint_prob.iloc[row_ord, row_ord]
        col_ordered_prob = joint_prob.iloc[col_ord, col_ord]
        row_out = op.join(outdir, 'row-ordered-' + fname)
        col_out = op.join(outdir, 'col-ordered-' + fname)
        row_ordered_prob.to_csv(row_out, sep='\t')
        col_ordered_prob.to_csv(col_out, sep='\t')
        # save dendrogram objects
        for ordering in ['row', 'col']:
            prefix = '{}-ordered-'.format(ordering)
            dg_fname = '{}-dendrogram-{}-{}-{}{}.yaml'.format(*args)
            with open(op.join(dgdir, prefix + dg_fname), 'w') as dgf:
                yaml.dump(dendrograms[ordering], dgf,
                          default_flow_style=True)
else:
    # loop over methods
    for method in methods:
        simulating = (method == 'theoretical')
        _subjects = ({str(accuracy): accuracy for accuracy in accuracies}
                     if simulating else subjects)
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
                    args = [method, sfn, lang, middle_arg + feat_sys,
                            subj_code]
                    fname = '{}-confusion-matrix-{}-{}-{}-{}.tsv'.format(*args)
                    fpath = op.join(indir, fname)
                    joint_prob = pd.read_csv(fpath, index_col=0, sep='\t')
                    # perform optimal ordering of rows/columns
                    olo = optimal_leaf_ordering(joint_prob)
                    dendrograms, linkages = olo['dendrograms'], olo['linkages']
                    row_ord = dendrograms['row']['leaves']
                    col_ord = dendrograms['col']['leaves']
                    ordered_prob = joint_prob.iloc[row_ord, col_ord]
                    # save ordered matrix
                    out = op.join(outdir, 'ordered-' + fname)
                    ordered_prob.to_csv(out, sep='\t')
                    if lang == 'eng':
                        row_ordered_prob = joint_prob.iloc[row_ord, row_ord]
                        col_ordered_prob = joint_prob.iloc[col_ord, col_ord]
                        row_out = op.join(outdir, 'row-ordered-' + fname)
                        col_out = op.join(outdir, 'col-ordered-' + fname)
                        row_ordered_prob.to_csv(row_out, sep='\t')
                        col_ordered_prob.to_csv(col_out, sep='\t')
                    # save dendrogram objects
                    for ordering in ['row', 'col']:
                        prefix = '{}-ordered-'.format(ordering)
                        dg_fname = ('{}-dendrogram-{}-{}-{}-{}.yaml'
                                    .format(*args))
                        dg_fpath = op.join(dgdir, prefix + dg_fname)
                        with open(dg_fpath, 'w') as dgf:
                            yaml.dump(dendrograms[ordering], dgf,
                                      default_flow_style=True)
