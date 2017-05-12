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
from mne.preprocessing import find_eog_events, create_eog_epochs
import numpy as np
from os import mkdir
import os.path as op
from pandas import read_csv


# no longer using these functions, now that mne blink detection functions
# ignore annotations properly
"""
def merge_overlapping_annotations(raw):
    # find bads (we don't want to merge bads + nonbads)
    descr = raw.annotations.description
    bads = [d.lower().startswith('bad') for d in descr]
    descr = descr[bads]
    onset = raw.annotations.onset[bads]
    offset = onset + raw.annotations.duration[bads]
    # set aside non-bads
    goods = np.logical_not(bads)
    good_onsets = raw.annotations.onset[goods]
    good_descr = raw.annotations.description[goods]
    good_durs = raw.annotations.duration[goods]
    # put bads in order
    order = np.argsort(onset)
    onset = onset[order]
    offset = offset[order]
    descr = descr[order]
    # find overlaps
    leading_overlappers = np.where(onset[1:] < offset[:-1])[0]
    indices = np.arange(len(onset))
    onset_keep_ix = np.setdiff1d(indices, leading_overlappers + 1)
    offset_keep_ix = np.setdiff1d(indices, leading_overlappers)
    onset = onset[onset_keep_ix]
    offset = offset[offset_keep_ix]
    descr = descr[onset_keep_ix]
    # restore non-bads
    raw.annotations.onset = np.r_[onset, good_onsets]
    raw.annotations.duration = np.r_[(offset - onset), good_durs]
    raw.annotations.description = np.r_[descr, good_descr]
    return raw


def find_longest_span(raw):
    onset = raw.annotations.onset
    offset = onset + raw.annotations.duration
    ix_of_longest_span_offset = np.argmax(onset[1:] - offset[:-1])
    beginning = offset[ix_of_longest_span_offset]
    end = onset[(ix_of_longest_span_offset + 1)]
    return (beginning, end)
"""


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
wav_names = global_params['wav_names']
wav_nsamps = np.array(global_params['wav_nsamps'])
stim_fs = global_params['stim_fs']
# analysis params
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
subjects = analysis_params['subjects']
do_baseline = analysis_params['eeg']['baseline']
do_reject = None if analysis_params['eeg']['autoreject'] else dict(eeg=180e-6)
align_on_cv = analysis_params['align_on_cv']
blink_chan = analysis_params['blink_channel']
n_jobs = analysis_params['n_jobs']
skip = analysis_params['skip']
seed = analysis_params['seed']
# how long a time of brain response do we care about?
brain_resp_dur = analysis_params['brain_resp_dur']
del global_params, analysis_params

# reference channel(s)
ref_chans = ['Ch17']

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

# containers
n_blinks_detected = dict()
n_epochs = dict()
# iterate over subjects
for subj_code, subj in subjects.items():
    if subj_code in skip:
        continue
    # read Raws and events
    basename = '{0:03}-{1}-'.format(subj, subj_code)
    events = mne.read_events(op.join(indir, 'events', basename + 'eve.txt'),
                             mask=None)
    raw = mne.io.read_raw_fif(op.join(indir, 'raws', basename + 'raw.fif.gz'),
                              preload=True)
    sfreq = raw.info['sfreq']
    # set EEG reference
    """
    raw.del_proj()  # necessary if re-running
    """
    mne.io.set_eeg_reference(raw, ref_chans, copy=False)
    raw.drop_channels(ref_chans)
    # make sure we don't cut off any events by mistake
    assert raw.first_samp == 0
    picks = mne.pick_types(raw.info, meg=False, eeg=True)
    """
    # filter -- already done in 015-reannotate.py
    raw.filter(l_freq=0.1, h_freq=40., picks=picks, n_jobs=n_jobs)
    """
    # add blink projector to original raw
    blink_events = find_eog_events(raw, ch_name=blink_chan[subj_code],
                                   reject_by_annotation=True)
    blink_epochs = create_eog_epochs(raw, picks=picks, tmin=-0.5, tmax=0.5,
                                     ch_name=blink_chan[subj_code])
    ssp_blink_proj = mne.compute_proj_epochs(blink_epochs, n_grad=0, n_mag=0,
                                             n_eeg=2, n_jobs=n_jobs,
                                             desc_prefix=None, verbose=None)
    raw = raw.add_proj(ssp_blink_proj)
    # re-save raw with projector added; save blinks for later plotting / QA
    raw.save(op.join(indir, 'raws', basename + 'raw.fif.gz'), overwrite=True)
    mne.write_events(op.join(indir, 'blinks', basename + 'blink-eve.txt'),
                     blink_events)
    blink_epochs.save(op.join(indir, 'blinks', basename + 'blink-epo.fif.gz'))
    n_blinks_detected[subj_code] = len(blink_epochs)
    del (blink_epochs, ssp_blink_proj)
    # make event dicts
    ev_to_id = {k: v for k, v in wavname_ev_id.items() if v in events[:, -1]}
    id_to_ev = {v: k for k, v in wavname_ev_id.items() if v in events[:, -1]}
    # generate epochs aligned on stimulus onset
    if not align_on_cv:
        epochs = mne.Epochs(raw, events, ev_to_id, tmin_onset, tmax_onset,
                            baseline, picks, reject=do_reject, preload=True)
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
                            preload=True)
        if do_baseline:
            # zero out all data in the unbaselined, CV-aligned epochs object
            epochs._data[:, :, :] = 0.
            # create baselined, onset-aligned data
            ep_onset = mne.Epochs(raw, events, ev_to_id, tmin_onset,
                                  tmax_onset, baseline, picks,
                                  reject=None, preload=True)
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
    n_epochs[subj_code] = len(epochs)
    epochs.save(op.join(outdir, basename + cv + 'epo.fif.gz'))
# output some info
print('\n\nBLINKS')
for subj_code, n_blinks in n_blinks_detected.items():
    print('{}{:>5}'.format(subj_code, n_blinks), end='   ')
print('\n\nEPOCHS')
for subj_code, n_epochs in n_epochs.items():
    print('{}{:>5}'.format(subj_code, n_epochs), end='   ')
print()
