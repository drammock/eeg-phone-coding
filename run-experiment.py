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
from pandas import read_csv
from expyfun import ExperimentController
from expyfun.stimuli import get_tdt_rates
from glob import glob
import os.path as op

# load experiment parameters
globalvars = np.load(op.join('params', 'global-params.npz'))
wav_array = globalvars['wav_array']
wav_nsamp = globalvars['wav_nsamps']
wav_names = globalvars['wavnames'].tolist()
fs = (get_tdt_rates()['25k'] if round(globalvars['fs']) == 24414 else
      globalvars['fs'])
del globalvars
# load & calc trial-level params
df = read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t')
df['wav_path'] = df['talker'] + '/' + df['syll'] + '.wav'  # do NOT use op.sep
df['wav_idx'] = [wav_names.index(x) for x in df['wav_path']]
df['wav_nsamp'] = [wav_nsamp[x] for x in df['wav_idx']]
df['onset_sec'] = df['onset'] / fs
df['offset_sec'] = df['offset'] / fs
df['trial_id'] = (df['block'].astype(int).apply(format, args=('02',)) + '_' +
                  df['talker'] + '_' + df['syll'])
digits = np.ceil(np.log2(wav_array.shape[0])).astype(int)
df['ttl_id'] = df['wav_idx'].apply(np.binary_repr, width=digits)
df['ttl_id'] = df['ttl_id'].apply(lambda x: [int(y) for y in list(x)])

# video paths
video = sorted(glob(op.join('videos', '*.m4v')))
assert len(video) == 20

# startup ExperimentController
continue_key = 1
ec_args = dict(exp_name='jsalt-follow-up', full_screen=True, enable_video=True,
               participant='foo', session='0', version='dev',  # 0ee0951
               stim_rms=0.01, stim_db=65., output_dir='expyfun-data-raw')

with ExperimentController(**ec_args) as ec:
    screen_period = 1. / ec.estimate_screen_fs(60)
    subj = int(ec.session)
    audio = sorted(glob(op.join('stimuli-final', 'subj-{:02}'.format(subj),
                                '*.wav')))
    blocks = len(audio)
    del audio
    assert blocks in (12, 13)
    # reduce data frame to subject-specific data
    subj_df = df[df['subj'].isin([subj])].reset_index(drop=True)
    for block in range(blocks):
        # subset data frame for this block
        blk_df = subj_df[subj_df['block'].isin([block])].copy()
        blk_len = df[['subj', 'block']].groupby('block').aggregate(np.size)
        # get the video file
        assert len(set(blk_df['vname'])) == 1
        vname = blk_df['vname'].values[0]
        vpath = op.join('videos', vname)
        assert vpath in video
        ec.load_video(vpath)
        ec.video.set_scale('fit')
        ec.call_on_next_flip(ec.video.play)
        # prepare syllable-level variables
        strings = blk_df['trial_id'].values.astype(str)
        floats = blk_df[['onset_sec', 'offset_sec']].values
        ints = blk_df['wav_idx'].values
        lists = blk_df['ttl_id'].values
        # iterate through syllables
        for ix, (trial_id, (onset, offset), wav_idx,
                 ttl_id) in enumerate(zip(strings, floats, ints, lists)):
            ec.load_buffer(wav_array[wav_idx])
            ec.identify_trial(ec_id=trial_id, ttl_id=dict(id_=ttl_id,
                                                          wait_for_last=False,
                                                          delay=0.01))
            # start initial stimulus
            if not ix:
                t_zero = ec.start_stimulus()
            # timing
            this_audio_start = t_zero + onset
            this_audio_stop = t_zero + offset
            t = ec.get_time()
            if ix:
                while t < this_audio_start - screen_period:
                    t = ec.flip()
                ec.start_stimulus(flip=False)
            while t < this_audio_stop:
                t = ec.flip()
            ec.stop()
            ec.trial_ok()
        ec.video.set_visible(False, flip=True)
        ec.delete_video()
        ec.flush()
        if block == blocks - 1:
            msg = ('All done! Sit tight and we will come disconnect the EEG.')
            max_wait = 5.
        else:
            msg = ('End of block {} of {}. Take a break now if you like, '
                   'then press {} to start the next block.'
                   ''.format(block + 1, blocks, continue_key))
            max_wait = np.inf
        ec.screen_prompt(msg, live_keys=[continue_key], max_wait=max_wait)
