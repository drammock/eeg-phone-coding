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

from __future__ import division, print_function
import yaml
import mne
import numpy as np
from os import mkdir
from os import path as op
from pandas import read_csv

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = op.join(indir, 'epochs')
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS...
paramdir = 'params'
paramfile = 'global-params.npz'
analysis_param_file = 'current-analysis-settings.yaml'
# ...from NPZ
params = np.load(op.join(paramdir, paramfile))
isi_range = params['isi_range']
wav_names = params['wav_names']
wav_nsamps = params['wav_nsamps']
stim_fs = params['fs'].astype(float)
# ...from YAML
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
subjects = analysis_params['subjects']
do_baseline = analysis_params['eeg']['baseline']
do_reject = None if analysis_params['eeg']['autoreject'] else dict(eeg=40e-6)
align_on_cv = analysis_params['align_on_cv']
n_jobs = analysis_params['n_jobs']
del params, analysis_params

# file naming variables
cv = 'cvalign-' if align_on_cv else ''

# LOAD DURATION DATA...
cvfile = 'cv-boundary-times.tsv'
df = read_csv(op.join(paramdir, cvfile), sep='\t')
df['key'] = df['talker'] + '/' + df['consonant'] + '.wav'
df['key'] = df['key'].apply(unicode)
assert set(wav_names) - set(df['key']) == set([])  # do all keys have a time?
# calculate consonant and vowel durations
wav_durs = wav_nsamps / stim_fs
wav_dur_dict = {fname: dur for fname, dur in zip(wav_names, wav_durs)}
cons_dur_dict = {fname: dur for fname, dur in
                 df[['key', 'CV-transition-time']].values}
vowel_dur_dict = {fname: wav_dur_dict[fname] - cons_dur_dict[fname]
                  for fname in wav_dur_dict.keys()}
cons_durs = np.array([cons_dur_dict[key] for key in wav_names])
vowel_durs = np.array([vowel_dur_dict[key] for key in wav_names])
assert np.allclose(wav_durs, cons_durs + vowel_durs)
# set epoch temporal parameters. Include as much pre-stim time as possible,
# so later we can shift to align on C-V transition.
brain_resp_dur = 0.2  # how long a time of brain response do we care about?
prev_trial_offset = brain_resp_dur - isi_range.min()
baseline = (prev_trial_offset, 0) if do_baseline else None
tmin_onset = min([prev_trial_offset, cons_durs.min() - cons_durs.max()])
tmax_onset = wav_durs.max() + brain_resp_dur
tmin_cv = 0 - cons_durs.max()
tmax_cv = vowel_durs.max() + brain_resp_dur

# integer event IDs are indices into wav_names. This re-creates that mapping:
wavname_ev_id = dict()
for _id, name in enumerate(wav_names):
    wavname_ev_id[name] = _id

# iterate over subjects
for subj_code, subj in subjects.items():
    # read Raws and events
    basename = '{0:03}-{1}-'.format(subj, subj_code)
    events = mne.read_events(op.join(indir, 'events', basename + 'eve.txt'),
                             mask=None)
    raw = mne.io.read_raw_fif(op.join(indir, 'raws', basename + 'raw.fif.gz'),
                              preload=True, add_eeg_ref=False)
    assert raw.first_samp == 0
    # filter
    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    raw.filter(l_freq=1, h_freq=40., picks=picks, n_jobs=n_jobs)
    # make event dicts
    ev_to_id = {k: v for k, v in wavname_ev_id.items() if v in events[:, -1]}
    id_to_ev = {v: k for k, v in wavname_ev_id.items() if v in events[:, -1]}
    # generate epochs aligned on stimulus onset
    if not align_on_cv:
        epochs = mne.Epochs(raw, events, ev_to_id, tmin_onset, tmax_onset,
                            baseline, picks, reject=do_reject,
                            add_eeg_ref=False, preload=True)
    else:
        # generate arrays of stimulus durations that follow presentation order
        wav_dur_by_trial = np.array([wav_dur_dict[id_to_ev[_id]]
                                     for _id in events[:, -1]])
        cons_dur_by_trial = np.array([cons_dur_dict[id_to_ev[_id]]
                                      for _id in events[:, -1]])
        vowel_dur_by_trial = np.array([vowel_dur_dict[id_to_ev[_id]]
                                       for _id in events[:, -1]])
        # for each stim, shift event code later by the duration of the cons.
        orig_secs = np.squeeze([raw.times[samp] for samp in events[:, 0]])
        events[:, 0] = raw.time_as_index(orig_secs + cons_dur_by_trial)
        mne.write_events(op.join(indir, 'events', basename + cv + 'eve.txt'),
                         events)
        # baselining needs to be done on onset-aligned epochs, even though
        # we want to end up with CV-aligned epochs
        epochs = mne.Epochs(raw, events, ev_to_id, tmin_cv, tmax_cv,
                            baseline=None, picks=picks, reject=do_reject,
                            add_eeg_ref=False, preload=True)
        if do_baseline:
            # zero out all data in the unbaselined, CV-aligned epochs object
            epochs._data[:, :, :] = 0.
            # create baselined, onset-aligned data
            ep_onset = mne.Epochs(raw, events, ev_to_id, tmin_onset,
                                  tmax_onset, baseline, picks,
                                  reject=do_reject, add_eeg_ref=False,
                                  preload=True)
            # for each trial...
            for ix, (v_dur, c_dur, w_dur) in enumerate(zip(vowel_dur_by_trial,
                                                           cons_dur_by_trial,
                                                           wav_dur_by_trial)):
                # compute start/end samples for onset- and cv-aligned data
                st_on = np.searchsorted(ep_onset.times, 0.)
                nd_on = np.searchsorted(ep_onset.times, w_dur + brain_resp_dur)
                st_cv = np.searchsorted(epochs.times, 0. - c_dur)
                nd_cv = np.searchsorted(epochs.times, v_dur + brain_resp_dur)
                # handle duration difference of +/- 1 sample (anything larger
                # will error out)
                if np.abs((nd_cv - st_cv) - (nd_on - st_on)) == 1:
                    nd_cv = st_cv + (nd_on - st_on)
                # insert baselined data, shifted to have C-V alignment
                epochs._data[ix, :, st_cv:nd_cv] = ep_onset._data[ix, :,
                                                                  st_on:nd_on]
            del ep_onset, st_cv, nd_cv, st_on, nd_on, ix, v_dur, c_dur, w_dur
    # downsample
    epochs = epochs.resample(100, npad=0, n_jobs=n_jobs)
    # save epochs
    epochs.save(op.join(outdir, basename + cv + 'epo.fif.gz'))

# finish
# np.savez(op.join(paramdir, 'subjects.npz'), **subjects)
