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
from numpy.testing import assert_equal
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
videodurs = [pygload(vp).duration for vp in videopaths]
mindur = min(videodurs)

# randomization
rng = np.random.RandomState(seed=0)
isi_lims = (0.25, 0.75)
n_subj = 12
combos = list(combinations(testdirs, 2))
trainfiles = list()
testfiles = list()
reps = 20

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
assert_equal(len(set(fs)), 1)  # make sure sampling rate consistent
fs = float(fs[0])
wavs = [x[0] for x in wav_and_fs]
assert_equal(len(wavs), len(allfiles))  # make sure they all loaded
# store wav data in one big array
wav_nsamps = np.array([x.shape[-1] for x in wavs])
wav_durs = wav_nsamps / fs
wavarray = np.zeros((len(wavs), 1, wav_nsamps.max()))
for ix, (wav, dur) in enumerate(zip(wavs, wav_nsamps)):
    wavarray[ix, :, :dur] = wav
del wavs, wav_and_fs
# training wavs are the same for all subjs
train_wavs = wavarray[train_mask]
train_samps = wav_nsamps[train_mask]

# handle subject-specific language selection
for subj in range(n_subj):
    this_testdirs = combos[subj % len(combos)]
    this_test_talkers = this_testdirs + engtestdirs
    this_test_mask = np.array([x in this_test_talkers for x in talkers])
    this_test_wavs = wavarray[this_test_mask]
    this_test_samps = wav_nsamps[this_test_mask]
    # make sure there is zero overlap between training and test
    assert np.all(np.logical_not(np.logical_and(train_mask, this_test_mask)))
    # shuffle training and test sounds
    this_all_wavs = np.concatenate((train_wavs, this_test_wavs), axis=0)
    this_all_samps = np.concatenate((train_samps, this_test_samps), axis=0)
    n_syll = this_all_wavs.shape[0]
    # TODO: pick up here, and don't loop by block using rep in reps
    # instead, do it based on mindur: create the master order across all
    # blocks as shuff, then split into wavs of appropriate length
    for rep in range(reps):
        shuff = rng.permutation(n_syll)
        ordered_wavs = this_all_wavs[shuff]
        ordered_samps = this_all_samps[shuff]
        # create ISIs (one more than we need, but makes zipping easier)
        isi_shuff = rng.permutation(n_syll)
        isi_secs = np.linspace(*isi_lims, num=n_syll)[isi_shuff]
        isi_samps = np.round(fs * isi_secs).astype(int)
        # assemble stimulus wav file for this block
        total_samps = ordered_samps.sum() + isi_samps[:-1].sum()
        out_wav = np.zeros((ordered_wavs.shape[1], total_samps))
        playhead = 0
        for syll, samp, isi in zip(ordered_wavs, ordered_samps, isi_samps):
            out_wav[:, playhead:playhead+samp] = syll
            playhead = playhead + samp + isi



raise RuntimeError()


'''
test_wavs = wavarray[np.logical_not(train_mask)]
train_wavs = wavarray[np.array(train_mask)]
#train_indices = np.tile(range(len(train_wavs)), train_reps)
'''

# iterate over subjects
for subj in range(n_subj):
    # use two foreign languages per subject
    this_testdirs = combos[subj % len(combos)]
    this_test_talkers = this_testdirs + engtestdirs
    this_test_indices = [x in this_test_talkers for x in talkers]
    this_test_wavs = np.array(test_wavs)[test_indices]
    # randomization
    rep_test_indices = np.tile(range(len(this_test_wavs)), test_reps)
    rep_wavs = np.concatenate((train_wavs[train_mask],
                               this_test_wavs[rep_test_indices]))
    this_order = rng.permutation(len(rep_wavs))
    this_wavs = rep_wavs[this_order]
    this_isis = np.linspace(*isi_lims, num=len(this_wavs) - 1)
    # build trials and blocks
    this_stimdurs = np.array([x.shape[-1] for x in this_wavs])
    np.vstack((this_stimdurs, this_isis))
