# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'autoreject-and-dss.py'
===============================================================================

This script loads EEG data in mne.Epochs format and runs autorejection to
detect bad channels, reject epochs with too many bad channels, and interpolate
bad channels if there are only a few that are bad.  It also applies denoising
source separation (DSS).
"""
# @author: drmccloy
# Created on Tue Nov 15 12:32:11 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import yaml
import mne
import numpy as np
from mne_sandbox.preprocessing import dss
from autoreject import LocalAutoRejectCV, compute_thresholds
from os import mkdir
from os import path as op
from functools import partial

raise RuntimeError

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
do_autorej = analysis_params['eeg']['autoreject']
do_dss = analysis_params['dss']['use']
align_on_cv = analysis_params['align_on_cv']
save_dss_mat = analysis_params['eeg']['save_dss_mat']
save_dss_data = analysis_params['eeg']['save_dss_data']
del analysis_params

# file naming variables
cv = 'cvalign-' if align_on_cv else ''
ar = 'autorej-' if do_autorej else ''

# set up autoreject
rs = mne.utils.check_random_state(12345)
# what percentage of bad sensors should trigger dropping an epoch?
consensus_percents = np.linspace(0., 1., 11)
# for retained epochs, how many of worst channels should we interpolate?
n_interpolates = np.array([2, 4, 8, 12])
thresh_func = partial(compute_thresholds, method='random_search',
                      random_state=rs)
# instantiate autoreject object
autorej = LocalAutoRejectCV(n_interpolates, consensus_percents,
                            thresh_func=thresh_func)

# iterate over subjects
for subj_code, subj in subjects.items():
    # read epochs
    basename = '{0:03}-{1}-{2}'.format(subj, subj_code, cv)
    epochs = mne.read_epochs(op.join(indir, basename + 'epo.fif.gz'))
    # autoreject
    if do_autorej:
        epochs = autorej.fit_transform(epochs)
    # save epochs
    epochs.save(op.join(indir, basename + ar + 'epo.fif.gz'))
    if do_dss:
        # compute DSS matrix. Keep all non-zero components for now.
        dss_mat, dss_data = dss(epochs, data_thresh=None, bias_thresh=None)
        fname = op.join(outdir, basename + ar)
        if save_dss_mat:
            np.save(fname + 'dss-mat.npy', dss_mat, allow_pickle=False)
        if save_dss_data:
            np.save(fname + 'dss-data.npy', dss_data, allow_pickle=False)
