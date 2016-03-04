# -*- coding: utf-8 -*-
"""
===============================================================================
Script ''
===============================================================================

This script plays audio and video.
"""
# @author: drmccloy
# Created on Mon Feb 29 17:18:25 2016
# License: BSD (3-clause)


import numpy as np
import pandas as pd
import expyfun as ef
from glob import glob
import os.path as op

# load experiment parameters
globalvars = np.load(op.join('params', 'global-params.npz'))
wav_array = globalvars['wav_array']
wav_nsamp = globalvars['wav_nsamps']
wav_names = globalvars['wavnames'].tolist()
fs = ef.stimuli.get_tdt_rates()['25k'] if round(globalvars['fs']) == 24414 \
    else globalvars['fs']
del globalvars
# load & calc trial-level params
df = pd.read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t')
df['wav_path'] = df['talker'] + op.sep + df['syll'] + '.wav'
df['wav_idx'] = [wav_names.index(x) for x in df['wav_path']]
df['wav_nsamp'] = [wav_nsamp[x] for x in df['wav_idx']]
df['onset_sec'] = df['onset'] / fs
df['offset_sec'] = df['offset'] / fs
df['trial_id'] = (df['block'].astype(int).apply(format, args=('02',)) + '_' +
                  df['talker'] + '_' + df['syll'])
# video paths
video = sorted(glob(op.join('videos', '*.m4v')))
assert len(video) == 20

# startup ExperimentController
ec_args = dict(exp_name='jsalt-follow-up', full_screen=True, enable_video=True,
               participant='foo', session='1', version='dev', stim_rms=0.02,
               stim_db=70.)
with ef.ExperimentController(**ec_args) as ec:
    screen_period = 1. / ec.estimate_screen_fs(60)
    subj = int(ec.session)
    audio = sorted(glob(op.join('stimuli-final', 'subj-{:02}'.format(subj),
                                '*.wav')))
    blocks = len(audio)
    del audio
    assert blocks in (12, 13)
    # reduce data frame to subject-specific data
    df = df[df['subj'].isin([subj])].reset_index(drop=True)
    for block in range(blocks):
        # subset data frame for this block
        this_df = df[df['block'].isin([block])].copy()
        block_lens = df[['subj', 'block']].groupby('block').aggregate(np.size)
        digits = np.ceil(np.log2(block_lens.max())).astype(int).values
        this_df['ttl_id'] = this_df.index.to_series().apply(np.binary_repr,
                                                            width=digits)
        this_df['next_onset_sec'] = np.roll(this_df['onset_sec'].values, -1)
        # get the video file
        assert len(set(this_df['vname'])) == 1
        vname = this_df['vname'].values[0]
        vpath = op.join('videos', vname)
        assert vpath in video
        ec.load_video(vpath)
        ec.video.set_scale(fill=True)
        ec.call_on_next_flip(ec.video.play)
        # construct the full ttl id
        ttl_base = np.binary_repr(subj, 4) + np.binary_repr(block, 4)
        this_df['ttl_id'] = this_df['ttl_id'].apply(lambda x: ttl_base + x)
        this_df['ttl_id'] = this_df['ttl_id'].apply(lambda x:
                                                    [int(y) for y in list(x)])
        # iterate through syllables
        strings = this_df['trial_id'].values.astype(str)
        floats = this_df[['onset_sec', 'offset_sec']].values
        ints = this_df[['wav_idx', 'wav_nsamp']].values
        objs = this_df['ttl_id'].values
        ec.load_buffer(np.zeros((2, 24414)))
        for ix, (trial_id, (onset, offset), (wav_idx, wav_nsamp),
                 ttl_id) in enumerate(zip(strings, floats, ints, objs)):
            #wav = wav_array[wav_idx, :, :wav_nsamp]
            #ec.load_buffer(wav)
            ec.identify_trial(ec_id=trial_id, ttl_id=ttl_id)
            # start initial stimulus
            if not ix:
                t = ec.flip()
                t_zero = ec.start_stimulus()
            # timing
            this_audio_start = t_zero + onset - screen_period / 2
            this_audio_stop = t_zero + offset
            t = ec.get_time()
            if ix:
                while t < this_audio_start:
                    t = ec.flip()
                    #flips.append(t)
                t = ec.start_stimulus()
            while t < this_audio_stop:
                t = ec.flip()
                #flips.append(t)
            ec.stop()
            ec.trial_ok()
            #ec.flush()

'''
from matplotlib import pyplot as plt
_ = plt.hist(np.diff(flips), bins=100)
'''
