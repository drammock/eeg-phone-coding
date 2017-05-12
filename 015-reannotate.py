#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'reannotate-raws.py'
===============================================================================

This script opens and plots MNE raw files for (re)annotation.
"""
# @author: drmccloy
# Created on Fri Apr  7 09:43:03 2017
# License: BSD (3-clause)

from __future__ import division, print_function
import yaml
import os.path as op
import mne
from mne.preprocessing import find_eog_events

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = op.join(indir, 'raws')

# LOAD PARAMS
paramdir = 'params'
analysis_paramfile = 'current-analysis-settings.yaml'
# analysis params
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
subjects = analysis_params['subjects']
n_jobs = analysis_params['n_jobs']
skip = analysis_params['skip']
blink_chan = analysis_params['blink_channel']
del analysis_params

# iterate over subjects
for subj_code, subj_num in subjects.items():
    if subj_code in skip:
        continue
    # read Raws and events
    basename = '{0:03}-{1}-'.format(subj_num, subj_code)
    fname = op.join(indir, 'raws', basename + 'raw.fif.gz')
    raw = mne.io.read_raw_fif(fname, preload=True)
    mne.io.set_eeg_reference(raw, ['Ch17'], copy=False)
    # filter
    picks = mne.pick_types(raw.info, meg=False, eeg=True)
    raw.filter(l_freq=0.1, h_freq=40., picks=picks, n_jobs=n_jobs)
    # blinks
    blink_events = find_eog_events(raw, ch_name=blink_chan[subj_code],
                                   reject_by_annotation=True)
    raw.plot(n_channels=33, duration=30, events=blink_events, block=True)
    new_blink_events = find_eog_events(raw, ch_name=blink_chan[subj_code],
                                       reject_by_annotation=True)
    print('#############################')
    print('old: {}'.format(blink_events.shape[0]))
    print('new: {}'.format(new_blink_events.shape[0]))
    print('#############################')
    # re-save annotated Raw object
    raw.save(fname, overwrite=True)
