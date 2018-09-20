#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'time-domain-redux.py'
===============================================================================

This script runs PCA on the time domain of epoched data, to reduce colinearity
of "features" (AKA time points) prior to classification, and then unrolls the
EEG channels (or DSS components, if using DSS) before saving.
"""
# @author: drmccloy
# Created on Tue Aug  8 13:33:19 PDT 2017
# License: BSD (3-clause)

import yaml
import mne
import numpy as np
import os.path as op
from os import mkdir
from time import time
from aux_functions import time_domain_pca, print_elapsed

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    pca_truncate = analysis_params['pca']['truncate_to_n_timepts']
    skip = analysis_params['skip']
    truncate = analysis_params['eeg']['truncate']

# FILE NAMING VARIABLES
cv = 'cvalign-' if align_on_cv else ''
trunc = '-truncated' if truncate else ''

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = op.join(indir, f'time-domain-redux{trunc}')
if not op.isdir(outdir):
    mkdir(outdir)

# iterate over subjects
for subj_code, subj in subjects.items():
    if subj_code in skip:
        continue
    basename = '{0:03}-{1}-{2}'.format(subj, subj_code, cv)
    # load data
    print('loading data for subject {}'.format(subj_code), end=': ')
    _st = time()
    epochs = mne.read_epochs(op.join(indir, f'epochs{trunc}',
                                     basename + 'epo.fif.gz'), verbose=False)
    if do_dss:
        data = np.load(op.join(indir, f'dss{trunc}',
                               basename + 'dss-data.npy'))
        data = data[:, :n_comp, :]
    else:
        data = epochs.get_data()
    print_elapsed(_st)

    # reduce time-domain dimensionality
    print('running PCA on time domain', end=': ')
    _st = time()
    data = time_domain_pca(data, max_components=pca_truncate)
    print_elapsed(_st)

    # unroll data / concatenate channels
    data = data.reshape(data.shape[0], -1)  # (n_epochs, n_chans * n_times)

    # save
    file_ext = 'dss{}-data.npy'.format(n_comp) if do_dss else 'epoch-data.npy'
    out_fname = basename + 'redux-' + file_ext
    np.save(op.join(outdir, out_fname), data, allow_pickle=False)
