#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os.path as op
import pandas as pd
from aux_functions import optimal_leaf_ordering

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    scheme = analysis_params['classification_scheme']
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

# FILE NAMING VARIABLES
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''

phone_level = ['pairwise', 'OVR', 'multinomial']

# BASIC FILE I/O
indir = f'processed-data-{scheme}{trunc}'
outdir = op.join(indir, 'ordered-confusion-matrices')
paramdir = 'params'

# FEATURE SETS
feature_systems = ['phoible_redux', 'jfh_sparse', 'spe_sparse']

for featsys in feature_systems:
    # load the data
    infix = '' if phone_level else (featsys + '-')
    fname = f'eer-confusion-matrix-nonan-eng-cvalign-dss5-{infix}average.tsv'
    fpath = op.join(indir, 'confusion-matrices', fname)
    confmat = pd.read_csv(fpath, sep='\t', index_col=0)
    # compute the optimal ordering
    olo = optimal_leaf_ordering(confmat)
    dendrograms, linkages = olo['dendrograms'], olo['linkages']
    row_ord = dendrograms['row']['leaves']
    ordered_confmat = confmat.iloc[row_ord, row_ord]
    # save ordered matrix
    out = op.join(outdir, 'row-ordered-' + fname)
    ordered_confmat.to_csv(out, sep='\t')
