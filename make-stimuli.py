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

# file paths
workdir = os.getcwd()
stimroot = 'stimuli-rms'
testdirs = ('hin-m', 'hun-f', 'swh-m', 'nld-f')
engtestdirs = ('eng-m1', 'eng-f1')
traindirs = ('eng-m2', 'eng-f2')

rng = np.random.RandomState(seed=0)
isi_lims = (0.25, 0.75)
n_subj = 12
combos = list(combinations(testdirs, 2))
trainfiles = list()
testfiles = list()
train_reps, test_reps = (20, 20)

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
train_indices = [x in traindirs for x in talkers]

# read in wav data
wav_and_fs = [read_wav(x) for x in allfiles]
fs = [x[1] for x in wav_and_fs]
assert_equal(len(set(fs)), 1)  # make sure sampling rate consistent
fs = float(fs[0])
wavs = np.array([x[0] for x in wav_and_fs])
assert_equal(len(wavs), len(allfiles))  # make sure they all loaded
test_wavs = wavs[~train_indices]
train_wavs = wavs[train_indices]
train_indices = np.tile(range(len(train_wavs)), train_reps)
del wavs, wav_and_fs

# iterate over subjects
for subj in range(n_subj):
    # use two foreign languages per subject
    this_testdirs = combos[subj % len(combos)]
    this_test_talkers = this_testdirs + engtestdirs
    this_test_indices = [x in this_test_talkers for x in talkers]
    this_test_wavs = np.array(test_wavs)[test_indices]
    # randomization
    rep_test_indices = np.tile(range(len(this_test_wavs)), test_reps)
    rep_wavs = np.concatenate((train_wavs[train_indices],
                               this_test_wavs[rep_test_indices]))
    this_order = rng.permutation(len(rep_wavs))
    this_wavs = rep_wavs[this_order]
    this_isis = np.linspace(*isi_lims, len(this_wavs) - 1)
    this_stimdurs = np.array([x.shape[-1] for x in this_wavs])

