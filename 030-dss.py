#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'dss.py'
===============================================================================

This script loads EEG data in mne.Epochs format and applies denoising source
separation (DSS).
"""
# @author: drmccloy
# Created on Tue Nov 15 12:32:11 2016
# License: BSD (3-clause)

import yaml
import mne
import numpy as np
from aux_functions import dss
from os import mkdir
from os import path as op


# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    truncate = analysis_params['eeg']['truncate']
del analysis_params

# FILE NAMING VARIABLES
trunc = '-truncated' if truncate else ''

# BASIC FILE I/O
indir = op.join('eeg-data-clean', f'epochs{trunc}')
outdir = op.join('eeg-data-clean', f'dss{trunc}')
if not op.isdir(outdir):
    mkdir(outdir)

# file naming variables
cv = 'cvalign-' if align_on_cv else ''

# iterate over subjects
for subj_code, subj_num in subjects.items():
    # read epochs
    basename = '{0:03}-{1}-{2}'.format(subj_num, subj_code, cv)
    epochs = mne.read_epochs(op.join(indir, basename + 'epo.fif.gz'))
    epochs.apply_proj()
    data = epochs.get_data()
    # compute DSS matrix
    dss_mat, dss_data = dss(data, pca_thresh=0.01, return_data=True)
    # save
    fname = op.join(outdir, basename)
    np.save(fname + 'dss-mat.npy', dss_mat, allow_pickle=False)
    np.save(fname + 'dss-data.npy', dss_data, allow_pickle=False)
