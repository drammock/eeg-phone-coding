#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os.path as op
import pandas as pd
from aux_functions import matrix_row_column_correlation


# LOAD SUBJECT LIST
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    scheme = analysis_params['classification_scheme']
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

subjects.update(average=0)

# FILE NAMING VARIABLES
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''

phone_level = scheme in ['pairwise', 'OVR', 'multinomial']

# BASIC FILE I/O
indir = f'processed-data-{scheme}{trunc}'

# FEATURE SETS
feature_systems = ['phoible_redux', 'jfh_sparse', 'spe_sparse']
feature_abbrevs = ['PHOIBLE', 'PSA', 'SPE']

# init container
diags = dict()

for featsys, abbrev in zip(feature_systems, feature_abbrevs):
    diags[featsys] = dict()

    for subj in subjects:
        # load confmat
        prefix = '' if phone_level else 'cross-featsys-'
        infix = '' if phone_level else f'{featsys}-'
        fname = (f'{prefix}cross-subj-row-ordered-eer-confusion-matrix-nonan-'
                 f'eng-cvalign-dss5-{infix}{subj}.tsv')
        confmat = pd.read_csv(op.join(indir, 'ordered-confusion-matrices',
                                      fname), sep='\t', index_col=0)
        # compute diagonality
        diag = matrix_row_column_correlation(confmat)
        diags[featsys][subj] = diag
        print('{:^8}({:^7}): {:.3}'.format(abbrev, subj, diag))

fname = 'cross-featsys-cross-subj-row-ordered-matrix-diagonality-nonan-eer.tsv'
df = pd.DataFrame.from_dict(diags)
df.to_csv(op.join(indir, 'matrix-correlations', fname), sep='\t')


# do it again with separate clustering for each matrix
diags = dict()
for featsys, abbrev in zip(feature_systems, feature_abbrevs):
    diags[featsys] = dict()

    for subj in subjects:
        # load confmat
        infix = '' if phone_level else f'{featsys}-'
        fname = ('row-ordered-eer-confusion-matrix-nonan-eng-cvalign-dss5-'
                 f'{infix}{subj}.tsv')
        confmat = pd.read_csv(op.join(indir, 'ordered-confusion-matrices',
                                      fname), sep='\t', index_col=0)
        # compute diagonality
        diag = matrix_row_column_correlation(confmat)
        diags[featsys][subj] = diag
        print('{:^8}({:^7}): {:.3}'.format(abbrev, subj, diag))

fname = 'individually-row-ordered-matrix-diagonality-nonan-eer.tsv'
df = pd.DataFrame.from_dict(diags)
df.to_csv(op.join(indir, 'matrix-correlations', fname), sep='\t')
