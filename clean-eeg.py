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
from glob import glob
from os import path as op
from os import mkdir
from expyfun import binary_to_decimals

# file i/o
paramdir = 'params'
paramfile = 'master-dataframe.tsv'
eegdir = 'eeg-data-raw'
outdir = 'eeg-data-clean'
if not op.isdir(outdir):
    mkdir(outdir)

# params
subjects = [1]
fs = 1000.
event_id = dict()

for subj in subjects:
    # read raws
    header = 'jsalt_binaural_cortical_{:03}.vhdr'.format(subj)
    raw = mne.io.read_raw_brainvision(op.join(eegdir, header),
                                      preload=True, response_trig_shift=None)
    raw_events = mne.find_events(raw)
    mne.write_events(op.join(outdir, header[:-5] + '-raw-eve.bak'), raw_events)
    # decode triggers
    stim_start_indices = np.where(raw_events[:, -1] == 1)[0]
    id_lims = np.c_[np.r_[0, stim_start_indices + 1][:-1], stim_start_indices]
    assert np.unique(np.diff(id_lims)) == 9  # number of bits in trigger IDs
    events = raw_events[stim_start_indices]
    for ix, (st, nd) in enumerate(id_lims):
        events[ix, -1] = binary_to_decimals(raw_events[st:nd, -1] // 4 - 1, 9)
    mne.write_events(op.join(outdir, header[:-5] + '-eve.txt'), events)
