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

from __future__ import division, print_function
import yaml
import mne
import numpy as np
from mne_sandbox.preprocessing import dss
from os import mkdir
from os import path as op


# BASIC FILE I/O
indir = op.join('eeg-data-clean', 'epochs')
outdir = op.join('eeg-data-clean', 'dss')
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
subjects = analysis_params['subjects']
align_on_cv = analysis_params['align_on_cv']
do_dss = analysis_params['dss']['use']
save_dss_mat = analysis_params['dss']['save_mat']
save_dss_data = analysis_params['dss']['save_data']
del analysis_params

# file naming variables
cv = 'cvalign-' if align_on_cv else ''

if do_dss:
    # iterate over subjects
    for subj_code, subj_num in subjects.items():
        # read epochs
        basename = '{0:03}-{1}-{2}'.format(subj_num, subj_code, cv)
        epochs = mne.read_epochs(op.join(indir, basename + 'epo.fif.gz'))
        epochs.apply_proj()
        # compute DSS matrix. Keep all non-zero components for now.
        dss_mat, dss_data = dss(epochs, data_thresh=0.01, bias_thresh=0.01)
        fname = op.join(outdir, basename)
        if save_dss_mat:
            np.save(fname + 'dss-mat.npy', dss_mat, allow_pickle=False)
        if save_dss_data:
            np.save(fname + 'dss-data.npy', dss_data, allow_pickle=False)
