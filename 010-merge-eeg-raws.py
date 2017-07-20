#!/usr/bin/env python3
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

import yaml
import mne
import numpy as np
from os import mkdir
from os import path as op
from expyfun import binary_to_decimals
from pandas import read_csv
from ast import literal_eval


def find_block_start_indices(events):
    stim_starts = np.where(events[:, -1] == 1)[0]
    diffs = np.diff(stim_starts)
    last_stim_of_block = np.where(diffs == 11)  # adjacent trials' diff == 10
    last_stim_of_block_ix = stim_starts[last_stim_of_block]
    return last_stim_of_block_ix + 1


# BASIC FILE I/O
indir = 'eeg-data-raw'
outdir = 'eeg-data-clean'
for _dir in ['', 'events', 'raws']:
    if not op.isdir(op.join(outdir, _dir)):
        mkdir(op.join(outdir, _dir))

# LOAD PARAMS
paramdir = 'params'
global_paramfile = 'global-params.yaml'
analysis_paramfile = 'current-analysis-settings.yaml'
with open(op.join(paramdir, global_paramfile), 'r') as f:
    global_params = yaml.load(f)
isi_range = np.array(global_params['isi_range'])
wav_names = global_params['wav_names']
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
subjects = analysis_params['subjects']
brain_resp_dur = analysis_params['brain_resp_dur']
skip = analysis_params['skip']
del global_params, analysis_params
# load montage
montage = mne.channels.read_montage('easycap-M1-LABSN-with-POOh',
                                    path='montage')
# load long-form dataframe of all trials across all subjects
master_df = read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
# and because pandas.read_csv(... dtype) argument doesn't work:
for col in ('subj', 'block', 'onset', 'offset', 'wav_idx', 'wav_nsamp'):
    master_df[col] = master_df[col].apply(int)
master_df['ttl_id'] = master_df['ttl_id'].apply(literal_eval)

# iterate over subjects
for subj_code, subj in subjects.items():
    if subj_code in skip:
        continue
    # read raws
    header = 'jsalt_binaural_cortical_{0}_{1:03}.vhdr'.format(subj_code, subj)
    basename = '{0:03}-{1}-'.format(subj, subj_code)
    raw = mne.io.read_raw_brainvision(op.join(indir, header),
                                      preload=True, response_trig_shift=None)
    # deal with subjects who had hardware failure and had to restart a block.
    # this is a bit convoluted due to (foolishly) stamping the start of each
    # block with integers (confounded with the 1, 4, and 8 triggers used for
    # stim start and trialID 0/1 bits)
    try:
        h = 'jsalt_binaural_cortical_{0}_{1:03}-2.vhdr'.format(subj_code, subj)
        raw2 = mne.io.read_raw_brainvision(op.join(indir, h), preload=True,
                                           response_trig_shift=None)
        two_runs = True
    except IOError:
        raw_events = mne.find_events(raw)
        two_runs = False
    if two_runs:
        raw1_events = mne.find_events(raw)
        raw2_events = mne.find_events(raw2)
        raw1_blocks = set(raw1_events[:, -1])  # may spuriously include 1,4,8
        raw2_first_block = raw2_events[0, -1]
        # figure out how many blocks we got through in raw1.  Block 1 is not
        # caught, because it's block start ID registers as a stim-start trig
        # and hence np.diff counts it differently.
        raw1_block_start_ix = find_block_start_indices(raw1_events)
        raw1_started_blocks = raw1_events[raw1_block_start_ix, -1]
        n_blocks_started = len(np.unique(raw1_started_blocks))
        if raw1_started_blocks[-1] == raw2_first_block:
            # skip aborted block, it is repeated in second run
            raw1_last_ix = raw1_block_start_ix[-1]
        else:
            raw1_last_ix = raw1_events.shape[0] + 1
        # concatenate raws, then purge events from the partial repeated block
        raw = mne.concatenate_raws([raw, raw2])
        raw_events = mne.find_events(raw)
        raw_events = np.r_[raw_events[:raw1_last_ix],
                           raw_events[(raw1_events.shape[0] + 1):]]
        del (h, raw2, raw1_events, raw2_events, raw2_first_block,
             raw1_block_start_ix, raw1_started_blocks, n_blocks_started,
             raw1_last_ix, two_runs)
    # set montage
    raw.set_montage(montage)
    # DO NOT set reference... which chan is ref gets lost on save
    # mne.io.set_eeg_reference(raw, ref_channels=['Ch17'], copy=False)

    # decode triggers to get proper event codes
    stim_start_indices = np.where(raw_events[:, -1] == 1)[0]
    # skip first 1-trigger, it's a block number (sigh):
    stim_start_indices = stim_start_indices[1:]
    # trial IDs are 9 binary digits, coming right before the stim_start 1-trig.
    id_lims = np.c_[np.r_[stim_start_indices - 9], stim_start_indices]
    # keep only the 1-triggers, but replace with trial IDs (converted to ints)
    events = raw_events[stim_start_indices]
    events[:, -1] = -999  # so we can detect and later delete glitchy trials
    for ix, (st, nd) in enumerate(id_lims):
        # 4 & 8 TTL triggers --> 0 & 1 bits --> integers
        try:
            events[ix, -1] = binary_to_decimals(raw_events[st:nd, -1] // 4 - 1,
                                                n_bits=9)
        except ValueError:
            # skip glitches (e.g., dropped samples around t=1217 during aborted
            # block of subj IA make it look like several trials are merged).
            if not np.all(np.in1d(raw_events[st:nd, -1], (4, 8))):
                # there was a spurious 1-trigger mixed in there, probably
                continue
            else:
                # something else weird and unexpected happened
                raise
    # delete events where we couldn't determine the stimulus (e.g., if EEG
    # recording glitches messed up the TTL stamps)
    events = events[np.where(events[:, -1] > 0)]
    # save events to file. Don't use raw.add_events to write them back into the
    # Raw file stim channel, because stimulus IDs 4 and 8 will become
    # confounded with the 4 & 8 bits in the binary trial IDs (actually,
    # stimulus IDs 3 and 7 would be the problem because raw.add_events will
    # **sum** them with the existing 1-triggers).
    mne.write_events(op.join(outdir, 'events', basename + 'eve.txt'), events)
    # clean Raw object: annotate breaks between blocks
    sfreq = raw.info['sfreq']
    pad_samp = int((isi_range[1] + brain_resp_dur + 1) * sfreq)
    indices = np.where(np.diff(events[:, 0]) > pad_samp)[0]
    onsets = (events[indices, 0] + pad_samp) / sfreq
    durs = ((events[(indices + 1), 0] - pad_samp) / sfreq) - onsets
    indices = indices[np.where(durs > 0)]
    onsets = onsets[np.where(durs > 0)]
    durs = durs[np.where(durs > 0)]
    assert len(indices) in [12, 13]
    # add in beginning and end of run
    first_onset = raw.first_samp / sfreq  # should be zero
    first_dur = (events[0, 0] - pad_samp) / sfreq
    last_onset = (events[-1, 0] + pad_samp) / sfreq
    last_dur = (raw.n_times / sfreq) - last_onset
    onsets = np.r_[first_onset, onsets, last_onset]
    durs = np.r_[first_dur, durs, last_dur]
    descrs = ['BAD_INTERBLOCK'] * len(durs)
    raw.annotations = mne.Annotations(onsets, durs, descrs)
    # save Raw object as FIF
    raw.save(op.join(outdir, 'raws', basename + 'raw.fif.gz'), overwrite=False)
