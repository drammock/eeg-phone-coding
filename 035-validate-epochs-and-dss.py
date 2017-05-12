#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'validate-epochs-and-dss.py'
===============================================================================

This script loads EEG data in mne.Epochs format, and DSS'ed versions of the
same data, for interactive inspection and plotting.
"""
# @author: drmccloy
# Created on Mon Apr 10 13:57:27 2017
# License: BSD (3-clause)

from __future__ import division, print_function
from os import path as op
import yaml
import numpy as np
import matplotlib.pyplot as plt
import mne

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = op.join('figures', 'dss')

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)

subjects = analysis_params['subjects']
align_on_cv = analysis_params['align_on_cv']
do_dss = analysis_params['dss']['use']
del analysis_params

# file naming variables
cv = 'cvalign-' if align_on_cv else ''

# layout
# lout = mne.channels.read_layout(kind='EEG1005')

# containers
epoch_counts = dict()

# iterate over subjects
for subj_code, subj_num in subjects.items():
    """
    # raws (testing)
    basename = '{0:03}-{1}-'.format(subj_num, subj_code)
    blinks = mne.read_events(op.join(indir, 'blinks',
                                     basename + 'blink-eve.txt'), mask=None)
    raw = mne.io.read_raw_fif(op.join(indir, 'raws', basename + 'raw.fif.gz'),
                              preload=True)
    print('USE THE "PROJ" BUTTON ON THE BOTTOM RIGHT OF THE PLOT TO ENABLE'
          'OR DISABLE BLINK PROJECTORS (to make sure they work passably well)')
    raw.plot(n_channels=33, duration=30, events=blinks)
    """
    # read epochs
    basename = '{0:03}-{1}-{2}'.format(subj_num, subj_code, cv)
    epochs = mne.read_epochs(op.join(indir, 'epochs', basename + 'epo.fif.gz'),
                             proj=False)
    epoch_counts[subj_code] = len(epochs)
    # read events
    events = mne.read_events(op.join(indir, 'events', basename + 'eve.txt'),
                             mask=None)
    if do_dss:
        dss_mat = np.load(op.join(indir, 'dss', basename + 'dss-mat.npy'))
        # make into projectors
        for n, component in enumerate(dss_mat):
            desc = 'dss-{}'.format(n)
            data = dict(data=component, nrow=1, ncol=len(component),
                        row_names=desc, col_names=epochs.ch_names)
            dss_proj = mne.io.Projection(data=data, active=False, kind=1,
                                         desc=desc, explained_var=None)
            epochs.add_proj(dss_proj)
        lout = mne.channels.find_layout(epochs.info)
        fig = mne.viz.plot_projs_topomap(epochs.info['projs'], layout=lout,
                                         show=False)
        fig.canvas.set_window_title(subj_code)
        plt.savefig(op.join(outdir, '{}.png'.format(subj_code)))
        plt.close()
print('EPOCH COUNTS:')
print('\n'.join(['{}: {}'.format(s, n) for s, n in epoch_counts.items()]))
