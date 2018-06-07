#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os.path as op
import pandas as pd
from aux_functions import matrix_row_column_correlation


# BASIC FILE I/O
indir = op.join('processed-data-logistic', 'ordered-confusion-matrices')
paramdir = 'params'

# LOAD SUBJECT LIST
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']

subjects.update(average=0)

# FEATURE SETS
feature_systems = ['phoible_redux', 'jfh_sparse', 'spe_sparse']
feature_abbrevs = ['PHOIBLE', 'PSA', 'SPE']

# init container
diags = dict()

for featsys, abbrev in zip(feature_systems, feature_abbrevs):
    diags[featsys] = dict()

    for subj in subjects:
        # load confmat
        fname = ('cross-featsys-cross-subj-row-ordered-eer-confusion-matrix-'
                 'nonan-eng-cvalign-dss5-{}-{}.tsv'.format(featsys, subj))
        confmat = pd.read_csv(op.join(indir, fname), sep='\t', index_col=0)
        # compute diagonality
        diag = matrix_row_column_correlation(confmat)
        diags[featsys][subj] = diag
        print('{:^8}({:^7}): {:.3}'.format(abbrev, subj, diag))

fname = 'cross-featsys-cross-subj-row-ordered-matrix-diagonality-nonan-eer.tsv'
df = pd.DataFrame.from_dict(diags)
df.to_csv(op.join('processed-data-logistic', 'matrix-correlations', fname),
          sep='\t')


# do it again with separate clustering for each matrix
diags = dict()
for featsys, abbrev in zip(feature_systems, feature_abbrevs):
    diags[featsys] = dict()

    for subj in subjects:
        # load confmat
        fname = ('row-ordered-eer-confusion-matrix-'
                 'nonan-eng-cvalign-dss5-{}-{}.tsv'.format(featsys, subj))
        confmat = pd.read_csv(op.join(indir, fname), sep='\t', index_col=0)
        # compute diagonality
        diag = matrix_row_column_correlation(confmat)
        diags[featsys][subj] = diag
        print('{:^8}({:^7}): {:.3}'.format(abbrev, subj, diag))

fname = 'individually-row-ordered-matrix-diagonality-nonan-eer.tsv'
df = pd.DataFrame.from_dict(diags)
df.to_csv(op.join('processed-data-logistic', 'matrix-correlations', fname),
          sep='\t')
