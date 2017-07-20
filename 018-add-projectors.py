#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'add-projectors.py'
===============================================================================

This script reads EEG data from mne.io.Raw format, runs blink detection, and
re-saves in mne.io.Raw format with an SSP projector added that removes blink
artifacts. Also filters the data, and sets the reference channel and then drops
it after subtraction.
"""
# @author: drmccloy
# Created on Tue Jul 11 15:07:09 2017
# License: BSD (3-clause)

import yaml
import mne
from mne.preprocessing import find_eog_events, create_eog_epochs
import numpy as np
from os import mkdir
import os.path as op
from pandas import read_csv

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = op.join(indir, 'raws-with-projs')
if not op.isdir(outdir):
    mkdir(outdir)

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
ref_chan = analysis_params['eeg']['ref_channel']
del analysis_params

# containers
n_blinks_detected = dict()

# iterate over subjects
for subj_code, subj in subjects.items():
    if subj_code in skip:
        continue
    # read Raws and events
    basename = '{0:03}-{1}-'.format(subj, subj_code)
    events = mne.read_events(op.join(indir, 'events', basename + 'eve.txt'),
                             mask=None)
    raw_fname = op.join(indir, 'raws-annotated', basename + 'raw.fif.gz')
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    sfreq = raw.info['sfreq']

    # set EEG reference
    mne.io.set_eeg_reference(raw, ref_chan, copy=False)
    raw.drop_channels(ref_chan)

    # filter
    picks = mne.pick_types(raw.info, meg=False, eeg=True)
    raw.filter(l_freq=0.1, h_freq=40., picks=picks, n_jobs='cuda')

    # add blink projector to raw
    blink_events = find_eog_events(raw, ch_name=blink_chan[subj_code],
                                   reject_by_annotation=True)
    blink_epochs = mne.Epochs(raw, blink_events, event_id=998, tmin=-0.5,
                              tmax=0.5, proj=False, reject=None, flat=None,
                              baseline=None, picks=picks, preload=True,
                              reject_by_annotation=True)
    ssp_blink_proj = mne.compute_proj_epochs(blink_epochs, n_grad=0, n_mag=0,
                                             n_eeg=2, n_jobs=n_jobs,
                                             desc_prefix=None, verbose=None)
    raw = raw.add_proj(ssp_blink_proj)

    # save raw with projectors added; save blinks for later plotting / QA
    raw.save(op.join(outdir, basename + 'raw.fif.gz'), overwrite=True)
    mne.write_events(op.join(indir, 'blinks', basename + 'blink-eve.txt'),
                     blink_events)
    blink_epochs.save(op.join(indir, 'blinks', basename + 'blink-epo.fif.gz'))
    n_blinks_detected[subj_code] = len(blink_epochs)
    del (raw, blink_epochs, blink_events, ssp_blink_proj)

with open(op.join(indir, 'blinks', 'blink-summary.tsv'), 'w') as outfile:
    outfile.write('subj\tn_blinks\n')
    for subj, n_blinks in n_blinks_detected.items():
        outfile.write('{}\t{}\n'.format(subj, n_blinks))
