#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'validate-dss.py'
===============================================================================

This script loads EEG data in mne.Epochs format, and DSS'ed versions of the
same data, for interactive inspection and plotting.
"""
# @author: drmccloy
# Created on Mon Apr 10 13:57:27 2017
# License: BSD (3-clause)

from os import path as op
from os import mkdir
import yaml
import numpy as np
import matplotlib.pyplot as plt
import mne
from aux_functions import dss

np.set_printoptions(linewidth=130)

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    truncate = analysis_params['eeg']['truncate']
del analysis_params

# FILE NAMING VARIABLES
cv = 'cvalign-' if align_on_cv else ''
trunc = '-truncated' if truncate else ''

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = op.join('figures', f'dss{trunc}')
if not op.isdir(outdir):
    mkdir(outdir)


# iterate over subjects
fig, axs = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
fig.suptitle('Relative power of DSS components')
fig.text(0.5, 0.04, 'DSS component number', ha='center')
fig.text(0.04, 0.5, 'relative power', va='center', rotation='vertical')
for ix, (subj_code, subj_num) in enumerate(subjects.items()):
    basename = '{0:03}-{1}-{2}'.format(subj_num, subj_code, cv)
    ax = axs.ravel()[ix]
    # read epochs
    epochs = mne.read_epochs(op.join(indir, 'epochs', basename + 'epo.fif.gz'))
    data = epochs.get_data()
    # load DSS matrix
    dss_mat = np.load(op.join(indir, 'dss', basename + 'dss-mat.npy'))
    #dss_mat = dss_mat[:, :n_comp]  # for this plot we want to see all of them
    # compute power
    evoked = data.mean(axis=0)
    bias_cov = np.einsum('hj,ij->hi', evoked, evoked)
    biased_power = np.sqrt(((bias_cov @ dss_mat) ** 2).sum(axis=0))
    # plot powers
    dss_pow = biased_power / biased_power.max()
    dss_line = ax.plot(dss_pow, label='DSS component power')
    ax.axvline(n_comp-1, color='k', linewidth=0.5, linestyle='dashed')
    ax.set_title(subj_code)
# finish plots
#axs.ravel()[-1].legend(handles=dss_line)
ax.annotate('smallest\nretained\ncomponent', xy=(n_comp-1, dss_pow[n_comp-1]),
            xytext=(15, 15), textcoords='offset points', ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->'))
fig.savefig(op.join(outdir, 'dss-component-power.pdf'))
