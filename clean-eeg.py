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
from os import mkdir
from os import path as op
from expyfun import binary_to_decimals
from pandas import read_csv

# manually-set params
subjects = dict(IJ=1)
plot_evokeds = False
if plot_evokeds:
    from matplotlib import pyplot as plt
    from matplotlib import rcParams
    rcParams['lines.linewidth'] = 0.5

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
# wav_array = params['wav_array']
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
"""
print('consonant min: {}  max: {}'.format(round(consonant_durs.min(), 3),
                                          round(consonant_durs.max(), 3)))
print('vowel min: {}  max: {}'.format(round(vowel_durs.min(), 3),
                                      round(vowel_durs.max(), 3)))
"""
# we care about first 200ms of brain response
tmax = wav_durs.max() + 0.2
# include extra pre-zero time: allows later shifting to align on C-V transition
tmin = consonant_durs.min() - consonant_durs.max()
# make sure brain response from previous syllable not in baseline
baseline = (0.2 - isi_range.min(), 0)

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
    # decode triggers to get proper event codes
    stim_start_indices = np.where(raw_events[:, -1] == 1)[0]
    stim_start_indices = stim_start_indices[1:]  # 1st trigger is block number
    id_lims = np.c_[np.r_[stim_start_indices - 9], stim_start_indices]
    events = raw_events[stim_start_indices]
    for ix, (st, nd) in enumerate(id_lims):
        events[ix, -1] = binary_to_decimals(raw_events[st:nd, -1] // 4 - 1, 9)
    event_id = {k: v for k, v in master_ev_id.items() if v in events[:, -1]}
    rev_ev_id = {v: k for k, v in master_ev_id.items() if v in events[:, -1]}
    mne.write_events(basename + 'raw-eve.txt', events)
    # artifact removal
    raw.filter(l_freq=0.5, h_freq=40., l_trans_bandwidth=0.4,
               n_jobs='cuda', copy=False)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')
    # TODO: implement as DSS instead of ICA
    ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
    ica.fit(raw, picks=picks, decim=10, reject=dict(eeg=250e-6))
    ica.apply(raw, copy=False)
    # generate epochs
    mne.io.set_eeg_reference(raw, ref_channels=['Ch17'], copy=False)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        add_eeg_ref=True, baseline=baseline, preload=True)
    # prepare to re-align epochs
    orig_secs = np.squeeze([raw.index_as_time(samp) for samp in events[:, 0]])
    consonant_dur_by_trial = np.array([consonant_dur_dict[rev_ev_id[_id]]
                                       for _id in events[:, -1]])
    wav_dur_by_trial = np.array([wav_dur_dict[rev_ev_id[_id]]
                                 for _id in events[:, -1]])
    tmin_cv = 0 - consonant_durs.max()
    tmax_cv = vowel_durs.max()
    # shift timepoint of stim-start trigger to C-V boundary. Must do this
    # _after_ baselining. Can't use epochs.crop due to +/-1 sample issues.
    events_cv = events.copy()
    events_cv[:, 0] = raw.time_as_index(orig_secs + consonant_dur_by_trial)
    epochs_cv = mne.Epochs(raw, events_cv, event_id, tmin_cv, tmax_cv,
                           picks=picks, baseline=None, preload=True)
    epochs_cv._data[:] = np.nan
    # zero-pad and crop epochs
    for ix, (c_dur, w_dur) in enumerate(zip(consonant_dur_by_trial,
                                            wav_dur_by_trial)):
        _tmin = c_dur - consonant_durs.max()
        consonant_start_ix = np.searchsorted(epochs.times, 0.)
        vowel_end_ix = np.searchsorted(epochs.times, w_dur + 0.2)
        tmin_ix = np.searchsorted(epochs.times, _tmin)
        tmax_ix = tmin_ix + epochs_cv.times.size
        # zero-out EEG responses irrelevant to the current stimulus
        epochs_cv._data[ix, :, :consonant_start_ix] = 0.
        epochs_cv._data[ix, :, vowel_end_ix:] = 0.
        # crop
        epochs_cv._data[ix] = epochs._data[ix, :, tmin_ix:tmax_ix]
    assert np.all(~np.isnan(epochs_cv._data))
    # TODO: apply DSS here, instead of applying it to Raw
    # (will it still work now that the times have shifted???)
    # may need to abandon baselining and shift trigger times from the get-go...
    mne.write_events(basename + 'eve.txt', events_cv)
    epochs_cv.save(basename + 'epo.fif.gz')
    # generate evoked
    # TODO: will I even need evokeds after DSS?  Should calculation of evokeds
    # come after DSS or before?
    print('generating evokeds...')
    evoked_dict = {key: epochs_cv[rev_ev_id[_id]].average()
                   for key, _id in event_id.items()}
    mne.write_evokeds(basename + 'ave.fif', evoked_dict.values())
    # sample plot
    if plot_evokeds:
        nrow, ncol = (3, 2)
        fig, axs = plt.subplots(nrow, ncol)
        axs = np.expand_dims(axs, axis=1) if len(axs.shape) == 1 else axs
        for ix, (key, evoked) in enumerate(evoked_dict.items()):
            cv_boundary = consonant_dur_dict[key] * 1000.  # convert to ms
            if ix < nrow * ncol:
                ax = axs[ix // ncol, ix % ncol]
                _ = evoked.plot(axes=ax, show=False, titles=key)
                ylim = ax.get_ylim()
                _ = ax.vlines(0 - cv_boundary, *ylim, colors='red')
                _ = ax.set_ylim(*ylim)
            else:
                break
        plt.tight_layout()
        fig.show()

# finish
np.savez(op.join(paramdir, 'subjects.npz'), **subjects)
