#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'make-epochs.py'
===============================================================================

This script processes EEG data from mne.io.Raw format into epochs. Baseline
correction is done on the 100ms preceding each stimulus onset; after baseline
correction the epochs are temporally shifted to place time-0 at the
consonant-vowel boundary of each stimulus syllable.
"""
# @author: drmccloy
# Created on Tue Nov 15 12:32:11 2016
# License: BSD (3-clause)

import yaml
import mne
import numpy as np
from os import mkdir
import os.path as op
from pandas import read_csv

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = op.join(indir, 'epochs')
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS
paramdir = 'params'
global_paramfile = 'global-params.yaml'
analysis_paramfile = 'current-analysis-settings.yaml'
# global params
with open(op.join(paramdir, global_paramfile), 'r') as f:
    global_params = yaml.load(f)
isi_range = np.array(global_params['isi_range'])
stim_fs = global_params['stim_fs']

# analysis params
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
subjects = analysis_params['subjects']
reject = analysis_params['eeg']['reject']
bad_channels = analysis_params['bad_channels']
# align_on_cv = analysis_params['align_on_cv']  # always generate both
n_jobs = analysis_params['n_jobs']
skip = analysis_params['skip']
# how long a time of brain response do we care about?
brain_resp_dur = analysis_params['brain_resp_dur']
del global_params, analysis_params

# LOAD DURATION DATA...
wav_params = read_csv(op.join(paramdir, 'wav-properties.tsv'), sep='\t')
df = read_csv(op.join(paramdir, 'cv-boundary-times.tsv'), sep='\t')
df['key'] = df['talker'] + '/' + df['consonant'] + '.wav'
# make sure all keys have a CV transition time
assert set(wav_params['wav_path']) - set(df['key']) == set([])
# merge in wav params (eventID, nsamp)
df = df.merge(wav_params, how='right', left_on='key', right_on='wav_path')
# compute word and vowel durations
df['w_dur'] = df['wav_nsamp'] / stim_fs
df['v_dur'] = df['w_dur'] - df['CV-transition-time']
df.rename(columns={'CV-transition-time': 'c_dur', 'wav_idx': 'event_id',
                   'wav_nsamp': 'nsamp'}, inplace=True)
df = df[['event_id', 'key', 'nsamp', 'c_dur', 'v_dur', 'w_dur']]

# set epoch temporal parameters. Include as much pre-stim time as possible,
# so later we can shift to align on C-V transition.
prev_trial_offset = brain_resp_dur - isi_range.min()
baseline = (prev_trial_offset, 0)
tmin_onset = min([prev_trial_offset, df['c_dur'].min() - df['c_dur'].max()])
tmax_onset = df['w_dur'].max() + brain_resp_dur
tmin_cv = 0 - df['c_dur'].max()
tmax_cv = df['v_dur'].max() + brain_resp_dur

# containers
n_epochs_retained = dict()
# iterate over subjects
for subj_code, subj in subjects.items():
    if subj_code in skip:
        continue

    # read Raws
    basename = '{0:03}-{1}-'.format(subj, subj_code)
    raw_fname = op.join(indir, 'raws-with-projs', basename + 'raw.fif.gz')
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    # raw.set_eeg_reference([])  # EEG ref. chan. set & dropped in prev. script
    raw.info['bads'] = bad_channels[subj_code]

    # make event dicts
    events = mne.read_events(op.join(indir, 'events', basename + 'eve.txt'),
                             mask=None)
    ev = events[:, -1]

    # generate epochs aligned on stimulus onset (baselining needs to be done on
    # onset-aligned epochs, even if we want to end up with CV-aligned epochs).
    # we ignore annotations here, and instead reject epochs based on channel
    # amplitude thresholds (set in the param file)
    picks = mne.pick_types(raw.info, meg=False, eeg=True)
    ev_dict = df.loc[np.in1d(df['event_id'], ev), ['key', 'event_id']]
    ev_dict = ev_dict.set_index('key').to_dict()['event_id']
    epochs_baseline = mne.Epochs(raw, events, ev_dict, tmin_onset, tmax_onset,
                                 baseline, picks, reject=reject, preload=True,
                                 reject_by_annotation=False, proj=True)
    drops = np.where(epochs_baseline.drop_log)[0]

    # generate events file aligned on consonant-vowel transition time.
    # for each stim, shift event later by the duration of the consonant
    orig_secs = np.squeeze([raw.times[samp] for samp in events[:, 0]])
    df2 = df.set_index('event_id', inplace=False)
    events[:, 0] = raw.time_as_index(orig_secs + df2.loc[ev, 'c_dur'])
    ev_fname = op.join(indir, 'events', basename + 'cvalign-eve.txt')
    mne.write_events(ev_fname, events)

    # generate epochs aligned on CV-transition
    epochs = mne.Epochs(raw, events, ev_dict, tmin_cv, tmax_cv,
                        baseline=None, picks=picks, reject=None,
                        preload=True, reject_by_annotation=False)
    # dropped epochs are determined by the baselined epochs object
    epochs.drop(drops)
    # zero out all data in the unbaselined, CV-aligned epochs object
    # (it will get replaced with time-shifted baselined data)
    epochs._data[:, :, :] = 0.
    # for each trial, insert time-shifted baselined data
    df3 = df2.loc[ev, ['c_dur', 'v_dur', 'w_dur']].reset_index(drop=True)
    # get only the retained trials/epochs
    df3 = df3.loc[epochs.selection].reset_index(drop=True)
    for ix, c_dur, v_dur, w_dur in df3.itertuples(index=True, name=None):
        # compute start/end samples for onset- and cv-aligned data
        st_onsetalign = np.searchsorted(epochs_baseline.times, 0.)
        nd_onsetalign = np.searchsorted(epochs_baseline.times,
                                        w_dur + brain_resp_dur)
        st_cvalign = np.searchsorted(epochs.times, 0. - c_dur)
        nd_cvalign = np.searchsorted(epochs.times, v_dur + brain_resp_dur)
        # handle dur. diff. of +/- 1 samp. (anything larger will error out)
        nsamp_onsetalign = nd_onsetalign - st_onsetalign
        nsamp_cvalign = nd_cvalign - st_cvalign
        if np.abs(nsamp_cvalign - nsamp_onsetalign) == 1:
            nd_cvalign = st_cvalign + nsamp_onsetalign
        # insert baselined data, shifted to have C-V alignment
        epochs._data[ix, :, st_cvalign:nd_cvalign] = \
            epochs_baseline._data[ix, :, st_onsetalign:nd_onsetalign]

    # downsample
    epochs = epochs.resample(100, npad=0, n_jobs='cuda')
    epochs_baseline = epochs_baseline.resample(100, npad=0, n_jobs='cuda')

    # save epochs
    n_epochs_retained[subj_code] = len(epochs)
    epochs.save(op.join(outdir, basename + 'cvalign-epo.fif.gz'))
    epochs_baseline.save(op.join(outdir, basename + 'epo.fif.gz'))

# output some info
with open(op.join(outdir, 'epoch-summary.tsv'), 'w') as outfile:
    outfile.write('subj\tn_epochs\n')
    for subj, n_epochs in n_epochs_retained.items():
        outfile.write('{}\t{}\n'.format(subj, n_epochs))
