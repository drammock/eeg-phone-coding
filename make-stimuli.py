# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'make-stimuli.py'
===============================================================================

This script makes audio stimuli from folders of component WAV files.
"""
# @author: drmccloy
# Created on Mon Nov 30 13:41:39 2015
# License: BSD (3-clause)

import os
import numpy as np
from glob import glob
from os import path as op
from itertools import combinations
#from numpy.testing import assert_equal
from expyfun.io import read_wav, write_wav
from pyglet.media import load as pygload


# file paths
workdir = os.getcwd()
videodir = op.join(workdir, 'videos')
stimroot = 'stimuli-rms'
testdirs = ('hin-m', 'hun-f', 'swh-m', 'nld-f')
engtestdirs = ('eng-m1', 'eng-f1')
traindirs = ('eng-m2', 'eng-f2')

# read in videos to get block durations
videopaths = sorted(glob(op.join(videodir, '*.m4v')))
videonames = [op.split(x)[1] for x in videopaths]
videodurs = np.array([pygload(vp).duration for vp in videopaths])
max_block_dur = min(videodurs)

# randomization
rand = np.random.RandomState(seed=0)
isi_lims = (0.25, 0.75)
n_subj = 12
combos = list(combinations(testdirs, 2))
trainfiles = list()
testfiles = list()
nreps = 20
zeroint = np.zeros(1, dtype=int)

# sort audio into training and test
for stimdir in traindirs + testdirs + engtestdirs:
    stimpath = op.join(stimroot, stimdir)
    wavs = sorted(glob(op.join(stimpath, '*.wav')))
    if stimdir in traindirs:  # exclude excess tokens
        wavs = [x for x in wavs if x[-5] in ('0', '1', '2')]
        trainfiles.extend(wavs)
    else:
        if stimdir in engtestdirs:  # include only 1 token
            wavs = [x for x in wavs if x[-5] == '0']
        testfiles.extend(wavs)
allfiles = trainfiles + testfiles
talkers = [x.split(os.sep)[-2] for x in allfiles]
train_mask = np.array([x in traindirs for x in talkers])

# read in wav data
wav_and_fs = [read_wav(x) for x in allfiles]
fs = [x[1] for x in wav_and_fs]
wavs = [x[0] for x in wav_and_fs]
nchan = np.array([x.shape[0] for x in wavs])
assert len(wavs) == len(allfiles)  # make sure they all loaded
assert len(set(fs)) == 1  # make sure sampling rate consistent
assert len(set(nchan)) == 1  # make sure all mono or all stereo
fs = float(fs[0])
nchan = nchan[0]
# store wav data in one big array
wav_nsamps = np.array([x.shape[-1] for x in wavs])
wavarray = np.zeros((len(wavs), nchan, wav_nsamps.max()))
for ix, (wav, dur) in enumerate(zip(wavs, wav_nsamps)):
    wavarray[ix, :, :dur] = wav
del wavs, wav_and_fs
# training wavs are the same for all subjs
train_wavs = wavarray[train_mask]
train_nsamps = wav_nsamps[train_mask]

for subj in range(n_subj):
    # select which videos to show
    video_order = rand.permutation(len(videodurs))
    this_video_names = np.array(videonames)[video_order].tolist()
    this_video_durs = videodurs[video_order]
    # handle subject-specific language selection
    this_testdirs = combos[subj % len(combos)]
    this_test_talkers = this_testdirs + engtestdirs
    this_test_mask = np.array([x in this_test_talkers for x in talkers])
    this_test_wavs = wavarray[this_test_mask]
    this_test_nsamps = wav_nsamps[this_test_mask]
    # make sure there is zero overlap between training and test
    assert np.all(np.logical_not(np.logical_and(train_mask, this_test_mask)))
    # combine training and test sounds
    this_wavs = np.concatenate((train_wavs, this_test_wavs), axis=0)
    this_nsamps = np.concatenate((train_nsamps, this_test_nsamps), axis=0)
    # set stimulus order for whole experiment
    nsyll = this_wavs.shape[0]
    order = np.array([rand.permutation(nsyll) for _ in range(nreps)]).ravel()
    all_nsamps = this_nsamps[order]
    # create ISIs (one more than we need, but makes zipping easier)
    isi_secs = np.linspace(*isi_lims, num=len(order))
    isi_order = rand.permutation(len(isi_secs))
    isi_nsamps = np.round(fs * isi_secs).astype(int)[isi_order]
    # global onset/offset sample indices
    syll_onset_samp = np.r_[0, np.cumsum(all_nsamps + isi_nsamps)][:-1]
    syll_offset_samp = np.cumsum(all_nsamps + np.r_[0, isi_nsamps[:-1]])
    assert np.array_equal(syll_offset_samp - syll_onset_samp, all_nsamps)
    # split into blocks
    all_blocks_max_nsamps = np.floor(this_video_durs * fs).astype(int)
    all_blocks_wavs = list()
    all_blocks_idxs = list()
    first_idx = 0
    for this_block_max_nsamp in all_blocks_max_nsamps:
        rel_offsets = syll_offset_samp - syll_onset_samp[first_idx]
        if rel_offsets.max() < this_block_max_nsamp:
            last_idx = nsyll * nreps
        else:
            last_idx = np.where(rel_offsets > this_block_max_nsamp)[0].min()
        first_samp = syll_onset_samp[first_idx]
        last_samp = syll_offset_samp[last_idx - 1]
        nsamp = last_samp - first_samp
        assert this_block_max_nsamp >= nsamp
        this_block_wav = np.zeros((nchan, nsamp))
        this_block_syll_order = order[first_idx:last_idx]
        this_block_syll_onsets = (syll_onset_samp[first_idx:last_idx] -
                                  syll_onset_samp[first_idx])
        this_block_syll_offsets = (syll_offset_samp[first_idx:last_idx] -
                                   syll_onset_samp[first_idx])  # not a typo
        for onset, offset, idx in zip(this_block_syll_onsets,
                                      this_block_syll_offsets,
                                      this_block_syll_order):
            this_block_wav[:, onset:offset] = this_wavs[idx, :,
                                                        :this_nsamps[idx]]
        all_blocks_wavs.append(this_block_wav)
        all_blocks_idxs.append(this_block_syll_order)
        first_idx = last_idx
        if first_idx == nsyll * nreps:
            break
