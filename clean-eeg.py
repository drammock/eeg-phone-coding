# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'clean-eeg.py'
===============================================================================

This script analyzes the expyfun log to make sure there were no anomalies.
"""
# @author: drmccloy
# Created on Mon Feb 29 17:18:25 2016
# License: BSD (3-clause)


import mne
import numpy as np
from os import mkdir
from os import path as op
from expyfun import binary_to_decimals

# file i/o
paramdir = 'params'
paramfile = 'global-params.npz'
eegdir = 'eeg-data-raw'
outdir = 'eeg-data-clean'
if not op.isdir(outdir):
    mkdir(outdir)

# params
subjects = dict(IJ=1)
eeg_fs = 1000.
tmin, tmax = (-0.02, 0.7)
params = np.load(op.join(paramdir, paramfile))
wav_names = params['wav_names']
#wav_array = params['wav_array']
#stim_fs = params['fs']
del params

# create event dict
master_event_id = dict()
for _id, name in enumerate(wav_names):
    master_event_id[name] = _id

for subj_code, subj in subjects.iteritems():
    # read raws
    header = 'jsalt_binaural_cortical_{0}_{1:03}.vhdr'.format(subj_code, subj)
    raw = mne.io.read_raw_brainvision(op.join(eegdir, header),
                                      preload=True, response_trig_shift=None)
    mne.io.set_eeg_reference(raw, ref_channels=['Ch17'], copy=False)
    raw_events = mne.find_events(raw)
    mne.write_events(op.join(outdir, header[:-5] + '-raw-eve.bak'), raw_events)
    # decode triggers
    stim_start_indices = np.where(raw_events[:, -1] == 1)[0]
    stim_start_indices = stim_start_indices[1:]  # 1st trigger is block number
    id_lims = np.c_[np.r_[stim_start_indices - 9], stim_start_indices]
    events = raw_events[stim_start_indices]
    for ix, (st, nd) in enumerate(id_lims):
        events[ix, -1] = binary_to_decimals(raw_events[st:nd, -1] // 4 - 1, 9)
    event_id = {k: v for k, v in master_event_id.items() if v in events[:, -1]}
    mne.write_events(op.join(outdir, header[:-5] + '-eve.txt'), events)
    # TODO: shift timepoint of stim-start trigger to C-V boundary
    # TODO: calc tmin (max duration of pre-boundary Cs)
    # TODO: calc tmax (max duration of post-boundary Vs)
    # generate epochs
    epochs = mne.Epochs(raw, events, event_id)
