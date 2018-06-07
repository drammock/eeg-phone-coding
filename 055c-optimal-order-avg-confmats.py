#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as op
import pandas as pd
from aux_functions import optimal_leaf_ordering

# BASIC FILE I/O
indir = op.join('processed-data-logistic', 'confusion-matrices')
outdir = op.join('processed-data-logistic', 'ordered-confusion-matrices')
paramdir = 'params'

# FEATURE SETS
feature_systems = ['phoible_redux', 'jfh_sparse', 'spe_sparse']

for featsys in feature_systems:
    # load the data
    fname = ('eer-confusion-matrix-nonan-eng-cvalign-dss5-'
             '{}-average.tsv'.format(featsys))
    fpath = op.join(indir, fname)
    confmat = pd.read_csv(fpath, sep='\t', index_col=0)
    # compute the optimal ordering
    olo = optimal_leaf_ordering(confmat)
    dendrograms, linkages = olo['dendrograms'], olo['linkages']
    row_ord = dendrograms['row']['leaves']
    ordered_confmat = confmat.iloc[row_ord, row_ord]
    # save ordered matrix
    out = op.join(outdir, 'row-ordered-' + fname)
    ordered_confmat.to_csv(out, sep='\t')
