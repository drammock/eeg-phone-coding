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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import numpy as np
import pandas as pd
from glob import glob
from os import path as op
from itertools import combinations
from expyfun.io import read_wav, write_wav
from expyfun.stimuli import resample
from pyglet.media import load as pygload

# file paths
workdir = os.getcwd()
videodir = op.join(workdir, 'videos')
stimroot = 'stimuli-rms'
testdirs = ('hin-m', 'hun-f', 'swh-m', 'nld-f')
engtestdirs = ('eng-m1', 'eng-f1')
traindirs = ('eng-m2', 'eng-f2')
outdir = op.join(workdir, 'stimuli-final')
paramdir = op.join(workdir, 'params')
if not op.isdir(paramdir):
    os.mkdir(paramdir)

# config
rand = np.random.RandomState(seed=0)
isi_lims = (0.25, 0.75)
n_subj = 12
combos = list(combinations(testdirs, 2))
trainfiles = list()
testfiles = list()
nrep = 20
fs_out = 24414
do_resample = True
write_wavs = True

# sort audio into training and test
for stimdir in traindirs + testdirs + engtestdirs:
    stimpath = op.join(stimroot, stimdir)
    wavs = sorted(glob(op.join(stimpath, '*.wav')))
    if stimdir in traindirs:  # include only 3 tokens (a few CVs have 4)
        wavs = [x for x in wavs if x[-5] in ('0', '1', '2')]
        trainfiles.extend(wavs)
    else:
        if stimdir in engtestdirs:  # include only 1 token
            wavs = [x for x in wavs if x[-5] == '0']
        testfiles.extend(wavs)
allfiles = trainfiles + testfiles
talkers = [x.split(os.sep)[-2] for x in allfiles]
syllables = [op.splitext(op.split(x)[1])[0] for x in allfiles]
train_mask = np.array([x in traindirs for x in talkers])

# read in wav data
wav_and_fs = [read_wav(x) for x in allfiles]
fs_in = [x[1] for x in wav_and_fs]
wavs = [x[0] for x in wav_and_fs]
nchan = np.array([x.shape[0] for x in wavs])
assert len(wavs) == len(allfiles)  # make sure they all loaded
assert len(set(fs_in)) == 1  # make sure sampling rate consistent
assert len(set(nchan)) == 1  # make sure all mono or all stereo
fs = float(fs_in[0])
nchan = nchan[0]
# resample to fs_out
if do_resample:
    wavs = [resample(x, fs_out, fs, n_jobs='cuda') for x in wavs]
    fs = float(fs_out)
# store wav data in one big array (shorter wavs zero-padded at end)
wav_nsamps = np.array([x.shape[-1] for x in wavs])
wav_array = np.zeros((len(wavs), nchan, wav_nsamps.max()))
for ix, (wav, dur) in enumerate(zip(wavs, wav_nsamps)):
    wav_array[ix, :, :dur] = wav
del wavs, wav_and_fs

# read in videos to get block durations
videopaths = sorted(glob(op.join(videodir, '*.m4v')))
videonames = [op.split(x)[1] for x in videopaths]
videodurs = np.array([pygload(vp).duration for vp in videopaths])

# initialize dataframe to store various params
df = pd.DataFrame()

for subj in range(n_subj):
    print('subj {:02}, block'.format(subj), end=' ')
    # init some vars
    all_blocks_syll_order = list()
    all_blocks_onset_samp = list()
    all_blocks_offset_samp = list()
    # set up output directory
    subjdir = 'subj-{:02}'.format(subj)
    if not op.isdir(op.join(workdir, outdir, subjdir)):
        os.makedirs(op.join(workdir, outdir, subjdir))
    # select which videos to show
    video_order = rand.permutation(len(videodurs))
    this_video_names = np.array(videonames)[video_order].tolist()
    this_video_durs = videodurs[video_order]
    # handle subject-specific language selection
    this_testdirs = combos[subj % len(combos)]
    this_test_talkers = this_testdirs + engtestdirs
    this_test_mask = np.array([x in this_test_talkers for x in talkers])
    # make sure there is zero overlap between training and test
    assert np.all(np.logical_not(np.logical_and(train_mask, this_test_mask)))
    # combine training and test sounds
    this_idx = np.r_[np.where(train_mask)[0], np.where(this_test_mask)[0]]
    this_nsamps = wav_nsamps[this_idx]
    # set stimulus order for whole experiment
    nsyll = this_idx.size
    order = np.array([rand.permutation(this_idx) for _ in range(nrep)]).ravel()
    # create ISIs (one more than we need, but makes zipping easier)
    isi_secs = np.linspace(*isi_lims, num=len(order))
    isi_order = rand.permutation(len(isi_secs))
    isi_nsamps = np.round(fs * isi_secs).astype(int)[isi_order]
    # global onset/offset sample indices
    all_nsamps = wav_nsamps[order]
    syll_onset_samp = np.r_[0, np.cumsum(all_nsamps + isi_nsamps)][:-1]
    syll_offset_samp = np.cumsum(all_nsamps + np.r_[0, isi_nsamps[:-1]])
    assert np.array_equal(syll_offset_samp - syll_onset_samp, all_nsamps)
    # split into blocks
    all_blocks_max_nsamps = np.floor(this_video_durs * fs).astype(int)
    first_idx = 0
    for block_idx, this_block_max_nsamp in enumerate(all_blocks_max_nsamps):
        print(block_idx, end=' ')
        # calculate which syllables will fit in this block
        rel_offsets = syll_offset_samp - syll_onset_samp[first_idx]
        if rel_offsets.max() < this_block_max_nsamp:
            last_idx = nsyll * nrep
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
            # patch syllable into block wav
            if write_wavs:
                this_block_wav[:, onset:offset] = wav_array[idx, :,
                                                            :wav_nsamps[idx]]
            # append record to data frame
            is_training = talkers[idx] in traindirs
            record = dict(subj=subj, block=block_idx,
                          vdur=this_video_durs[block_idx],
                          vname=this_video_names[block_idx],
                          talker=talkers[idx], syll=syllables[idx],
                          train=is_training, onset=onset, offset=offset)
            df = df.append(record, ignore_index=True)
        # write out block wav
        if write_wavs:
            fname = 'block-{:02}.wav'.format(block_idx)
            write_wav(op.join(outdir, subjdir, fname), this_block_wav, int(fs),
                      overwrite=True)
        # iterate
        first_idx = last_idx
        if first_idx == nsyll * nrep:
            break
    print()  # newline between subjs
# save dataframe (note the bytes on the separator... pandas bug)
column_order = ['subj', 'block', 'vname', 'vdur', 'talker', 'syll', 'onset',
                'offset', 'train']
df.to_csv(op.join(paramdir, 'master-dataframe.tsv'), sep=b'\t', index=False,
          columns=column_order)
# save global params
wavnames = [x[1+x.index(op.sep):] for x in allfiles]
globalvars = dict(wav_array=wav_array, wav_nsamps=wav_nsamps, fs=fs,
                  wavnames=wavnames)
np.savez(op.join(paramdir, 'global-params.npz'), **globalvars)
