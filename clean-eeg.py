# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'clean-eeg.py'
===============================================================================

This script processes raw EEG data into epochs. Baseline correction is done
on the 100ms preceding each stimulus onset, but after baseline correction the
epochs are temporally re-aligned to place time-0 at the consonant-vowel
boundary of each stimulus syllable.
"""
# @author: drmccloy
# Created on Mon Feb 29 17:18:25 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import mne
import numpy as np
from mne_sandbox.preprocessing import dss
from os import mkdir
from os import path as op
from expyfun import binary_to_decimals
from pandas import read_csv
from ast import literal_eval

# manually-set params
subjects = dict(IJ=1, IL=2, FA=3, IM=4, ID=5, CQ=6, IP=7, FV=8, IA=9)
do_baseline = True
save_dss_data = True
save_dss_mat = True
save_epochs = True

# file i/o
paramdir = 'params'
paramfile = 'global-params.npz'
eegdir = 'eeg-data-raw'
outdir = 'eeg-data-clean'
cvfile = 'cv-boundary-times.tsv'
if not op.isdir(outdir):
    mkdir(outdir)

# load global params
params = np.load(op.join(paramdir, paramfile))
isi_range = params['isi_range']
wav_names = params['wav_names']
wav_nsamps = params['wav_nsamps']
stim_fs = params['fs'].astype(float)
del params

# load C-V transition times
df = read_csv(cvfile, sep='\t')
df['key'] = df['talker'] + '/' + df['consonant'] + '.wav'
df['key'] = df['key'].apply(unicode)
assert set(wav_names) - set(df['key']) == set([])  # have a time for all keys?

# calculate various durations
wav_durs = wav_nsamps / stim_fs
wav_dur_dict = {name: dur for name, dur in zip(wav_names, wav_durs)}
consonant_dur_dict = {key: dur for dur, key in df[['CV-transition-time',
                                                   'key']].values}
consonant_durs = np.array([consonant_dur_dict[key] for key in wav_names])
vowel_durs = np.array([wav_dur_dict[key] for key in wav_names]
                      ) - consonant_durs
vowel_dur_dict = {key: dur for key, dur in zip(wav_names, vowel_durs)}
assert np.allclose(wav_durs, consonant_durs + vowel_durs)

# load master data frame
mdf = read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
# because pandas.read_csv(... dtype) argument doesn't work:
for col in ('subj', 'block', 'onset', 'offset', 'wav_idx', 'wav_nsamp'):
    mdf[col] = mdf[col].apply(int)
mdf['ttl_id'] = mdf['ttl_id'].apply(literal_eval)

# we care about first 200ms of brain response
tmax = wav_durs.max() + 0.2
# include lots of pre-stim time: allows later shifting to align C-V transition
tmin = consonant_durs.min() - consonant_durs.max()
# make sure brain response from previous syllable not in baseline
baseline_times = (0.2 - isi_range.min(), 0)
# if we're aligning based on C-V transition instead of C-onset:
tmin_cv = 0 - consonant_durs.max()
tmax_cv = vowel_durs.max() + 0.2

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
    raw_events = mne.find_events(raw)
    # deal with subjects who had hardware failure and had to restart a block
    try:
        h = 'jsalt_binaural_cortical_{0}_{1:03}-2.vhdr'.format(subj_code, subj)
        raw2 = mne.io.read_raw_brainvision(op.join(eegdir, h), preload=True,
                                           response_trig_shift=None)
        two_runs = True
    except IOError:
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
                prev_block_len = mdf[(mdf['subj'] == subj) &
                                     (mdf['block'] == prev_block)].shape[0]
                common_block_ix = (np.where(raw1_events[:, -1] == prev_block
                                            )[0] + prev_block_len + 1)
            else:
                raise RuntimeError('You\'re throwing away the entire first '
                                   'EEG run. Why not rename the input files '
                                   'so the unusable one never gets read in.')
        else:
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
    # picks
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')
    # decode triggers to get proper event codes
    stim_start_indices = np.where(raw_events[:, -1] == 1)[0]
    stim_start_indices = stim_start_indices[1:]  # 1st trigger is block number
    id_lims = np.c_[np.r_[stim_start_indices - 9], stim_start_indices]
    events = raw_events[stim_start_indices]
    for ix, (st, nd) in enumerate(id_lims):
        events[ix, -1] = binary_to_decimals(raw_events[st:nd, -1] // 4 - 1, 9)
    event_id = {k: v for k, v in master_ev_id.items() if v in events[:, -1]}
    rev_ev_id = {v: k for k, v in master_ev_id.items() if v in events[:, -1]}
    mne.write_events(basename + 'c-aligned-eve.txt', events)
    # generate arrays of stimulus durations respecting presentation order
    consonant_dur_by_trial = np.array([consonant_dur_dict[rev_ev_id[_id]]
                                       for _id in events[:, -1]])
    vowel_dur_by_trial = np.array([vowel_dur_dict[rev_ev_id[_id]]
                                   for _id in events[:, -1]])
    wav_dur_by_trial = np.array([wav_dur_dict[rev_ev_id[_id]]
                                 for _id in events[:, -1]])
    # prepare time-shifted event codes aligned on C-V boundary
    assert raw.first_samp == 0
    orig_secs = np.squeeze([raw.times[samp] for samp in events[:, 0]])
    events_cv = events.copy()
    events_cv[:, 0] = raw.time_as_index(orig_secs + consonant_dur_by_trial)
    mne.write_events(basename + 'v-aligned-eve.txt', events_cv)
    # baseline, reference, and filter
    baseline = baseline_times if do_baseline else None
    mne.io.set_eeg_reference(raw, ref_channels=['Ch17'], copy=False)
    raw.filter(l_freq=1, h_freq=40., n_jobs='cuda')
    # generate epochs aligned on C onset
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        add_eeg_ref=True, baseline=baseline, preload=True)
    epochs = epochs.resample(100, npad=0, n_jobs='cuda')
    # generate epochs aligned on C-V transition
    epochs_cv = mne.Epochs(raw, events_cv, event_id, tmin_cv, tmax_cv,
                           picks=picks, baseline=None, preload=True)
    epochs_cv = epochs_cv.resample(100, npad=0, n_jobs='cuda')
    # zero-out EEG responses irrelevant to the current stimulus (ideally
    # not necessary, but maybe helpful due to short ISIs)
    for e_ix, e in enumerate((epochs_cv, epochs)):
        for ix, (c_dur, v_dur, w_dur) in enumerate(zip(consonant_dur_by_trial,
                                                       vowel_dur_by_trial,
                                                       wav_dur_by_trial)):
            c_start = 0. if e_ix else 0. - c_dur
            v_end = (w_dur if e_ix else v_dur) + 0.2
            c_start_ix = np.searchsorted(e.times, c_start)
            v_end_ix = np.searchsorted(e.times, v_end)
            # replace unbaselined data with baselined data
            if do_baseline and not e_ix:
                mint = np.searchsorted(epochs.times, 0.)
                maxt = np.searchsorted(epochs.times, w_dur + 0.2)
                # deal with +/- 1 sample issues
                if np.abs((v_end_ix - c_start_ix) - (maxt - mint)) == 1:
                    v_end_ix = c_start_ix + (maxt - mint)
                # replace unbaselined data with baselined data
                e._data[ix, :, c_start_ix:v_end_ix] = epochs._data[ix, :,
                                                                   mint:maxt]
            # zero out the early/late parts
            e._data[ix, :, :c_start_ix] = 0.
            e._data[ix, :, v_end_ix:] = 0.
        # compute DSS matrix. Keep all non-zero components for now.
        dss_mat, dss_data = dss(e, data_thresh=None, bias_thresh=None)
        align = 'v' if e_ix else 'c'
        if save_dss_mat:
            np.save(basename + '{}-aligned-dssmat.npy'.format(align),
                    dss_mat, allow_pickle=False)
        if save_dss_data:
            np.save(basename + '{}-aligned-dssdata.npy'.format(align),
                    dss_data, allow_pickle=False)
    # save epochs
    if save_epochs:
        epochs.save(basename + 'c-aligned-epo.fif.gz')
        epochs_cv.save(basename + 'v-aligned-epo.fif.gz')

# finish
np.savez(op.join(paramdir, 'subjects.npz'), **subjects)
