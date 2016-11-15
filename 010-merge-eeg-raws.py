# -*- coding: utf-8 -*-
"""
merge-eeg-raws.py

This script preprocesses raw EEG data files to:
- convert raw data from BrainVision format to mne-python Raw format
- combine files for subjects where the EEG system was stopped and restarted
- add montage and reference information to the Raw files
- generate event file with stimulus ID values in place of 1-triggers
"""
# @author: drmccloy
# Created on Mon Nov 14 17:18:25 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import yaml
import mne
import numpy as np
from os import mkdir, getcwd
from os import path as op
from expyfun import binary_to_decimals
from pandas import read_csv
from ast import literal_eval

# BASIC FILE I/O
eegdir = 'eeg-data-raw'
outdir = 'eeg-data-clean'
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS...
paramdir = 'params'
paramfile = 'global-params.npz'
analysis_param_file = 'current-analysis-settings.yaml'
# ...from NPZ
params = np.load(op.join(paramdir, paramfile))
wav_names = params['wav_names']
# ...from YAML
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
subjects = analysis_params['subjects']
del params, analysis_params
# load montage
montage = mne.channels.read_montage(op.join(getcwd(), 'montage',
                                            'easycap-M1-LABSN.txt'))
# load long-form dataframe of all trials across all subjects
master_df = read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
# and because pandas.read_csv(... dtype) argument doesn't work:
for col in ('subj', 'block', 'onset', 'offset', 'wav_idx', 'wav_nsamp'):
    master_df[col] = master_df[col].apply(int)
master_df['ttl_id'] = master_df['ttl_id'].apply(literal_eval)

# create event dict
master_ev_id = dict()
for _id, name in enumerate(wav_names):
    master_ev_id[name] = _id

# iterate over subjects
for subj_code, subj in subjects.items():
    # read raws
    header = 'jsalt_binaural_cortical_{0}_{1:03}.vhdr'.format(subj_code, subj)
    basename = op.join(outdir, '{0:03}-{1}-'.format(subj, subj_code))
    raw = mne.io.read_raw_brainvision(op.join(eegdir, header),
                                      preload=True, response_trig_shift=None)
    # deal with subjects who had hardware failure and had to restart a block.
    # this is a bit convoluted due to (foolishly) stamping the start of each
    # block with integers (confounded with the 1, 4, and 8 triggers used for
    # stim start and trialID 1/0 bits)
    try:
        h = 'jsalt_binaural_cortical_{0}_{1:03}-2.vhdr'.format(subj_code, subj)
        raw2 = mne.io.read_raw_brainvision(op.join(eegdir, h), preload=True,
                                           response_trig_shift=None)
        two_runs = True
    except IOError:
        raw_events = mne.find_events(raw)
        two_runs = False
    if two_runs:
        raw1_events = mne.find_events(raw)
        raw2_events = mne.find_events(raw2)
        raw1_blocks = set(raw1_events[:, -1])
        raw2_blocks = set(raw2_events[:, -1])
        if len(raw1_blocks & raw2_blocks) == 3:  # common block in [1, 4, 8]
            common_block = [1, 4, 8][len(raw1_blocks) // 4]
            if common_block in (4, 8):
                prev_block = common_block - 1
                prev_block_len = master_df[(master_df['subj'] == subj) &
                                           (master_df['block'] == prev_block)
                                           ].shape[0]
                common_block_ix = (np.where(raw1_events[:, -1] == prev_block
                                            )[0] + prev_block_len + 1)
            else:
                raise RuntimeError('You\'re throwing away the entire first '
                                   'EEG run. Why not rename the input files '
                                   'so the unusable one never gets read in.')
        else:  # common block is something other than 1, 4, or 8
            common_block = (raw1_blocks & raw2_blocks) - set([1, 4, 8])
            assert len(common_block) == 1
            common_block = common_block.pop()
            common_block_ix = np.where(raw1_events[:, -1] == common_block)[0]
            assert common_block_ix.size == 1
        first_raw2_ix = raw1_events.shape[0] + 1
        # concatenate raws, then purge events from the partial repeated block
        raw = mne.concatenate_raws([raw, raw2])
        raw_events = mne.find_events(raw)
        raw_events = np.r_[raw_events[:common_block_ix[0]],
                           raw_events[first_raw2_ix:]]
        del (h, raw2, raw1_events, raw2_events, common_block, common_block_ix,
             first_raw2_ix, two_runs)
    # set montage & reference
    raw.set_montage(montage)
    mne.io.set_eeg_reference(raw, ref_channels=['Ch17'], copy=False)

    # decode triggers to get proper event codes
    stim_start_indices = np.where(raw_events[:, -1] == 1)[0]
    # skip first 1-trigger, it's a block number (sigh):
    stim_start_indices = stim_start_indices[1:]
    # trial IDs are 9 binary digits, coming right before the stim_start 1-trig.
    id_lims = np.c_[np.r_[stim_start_indices - 9], stim_start_indices]
    # keep only the 1-triggers, but replace with trial IDs (converted to ints)
    events = raw_events[stim_start_indices]
    for ix, (st, nd) in enumerate(id_lims):
        # convert 4 & 8 to 0 & 1, 9 is # of bits:
        events[ix, -1] = binary_to_decimals(raw_events[st:nd, -1] // 4 - 1, 9)
    # save events to file. Don't use raw.add_events to write them back into the
    # Raw file stim channel, because stimulus IDs 4 and 8 will become
    # confounded with the 4 & 8 bits in the binary trial IDs (actually,
    # stimulus IDs 3 and 7 would be the problem because raw.add_events will
    # **sum** them with the existing 1-triggers).
    mne.write_events(basename + 'eve.txt', events)
    # save Raw object as FIF
    raw.save(basename + 'raw.fif.gz')
