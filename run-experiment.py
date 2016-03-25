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
from expyfun import ExperimentController, get_keyboard_input
from expyfun.stimuli import get_tdt_rates
from platform import system
from subprocess import Popen
from glob import glob
import os.path as op

# load experiment parameters
globalvars = np.load(op.join('params', 'global-params.npz'))
pad = globalvars['pad']
wav_array = globalvars['wav_array']
wav_nsamp = globalvars['wav_nsamps']
wav_names = globalvars['wavnames'].tolist()
fs = (get_tdt_rates()['25k'] if round(globalvars['fs']) == 24414 else
      globalvars['fs'])
del globalvars
stim_rms = np.sqrt(2 * 0.01 ** 2)  # rms'd the mono files to 0.01
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
if system() == 'Windows':
    exe = op.join('C:\\', 'Program Files', 'VideoLAN', 'VLC', 'vlc.exe')
else:
    exe = 'vlc'

# instructions
instructions = ('In this experiment you get to watch cartoons! You\'ll watch '
                '13 or 14 episodes of Shaun the Sheep (about 6 minutes each). '
                'The cartoon\'s audio is muted, and you will hear some speech '
                'sounds instead; all you have to do is watch the cartoon and '
                'passively listen. '
                'Press the 1 button when you\'re ready to start.')

# startup ExperimentController
continue_key = 1
ec_args = dict(exp_name='jsalt-follow-up', full_screen=True,
               #participant='pilot', session='5',
               stim_rms=stim_rms, stim_db=65., check_rms='wholefile',
               version='0ee0951', output_dir='expyfun-data-raw')

with ExperimentController(**ec_args) as ec:
    ec.set_visible(False)
    subj = int(ec.session) - 1  # convert to 0-indexed
    audio = sorted(glob(op.join('stimuli-final', 'subj-{:02}'.format(subj),
                                '*.wav')))
    blocks = len(audio)
    del audio
    assert blocks in (13, 14)
    starting_block = get_keyboard_input('starting block (leave blank & push '
                                        'ENTER to start at beginning): ',
                                        default=0, out_type=int,
                                        valid=range(blocks))
    ec.set_visible(True)
    ec.screen_prompt(instructions)
    # reduce data frame to subject-specific data
    subj_df = df[df['subj'].isin([subj])].reset_index(drop=True)
    for block in range(starting_block, blocks):
        ec.screen_prompt('Here we go!', max_wait=0.7, live_keys=[], attr=False)
        ec.set_visible(False)
        # subset data frame for this block
        blk_df = subj_df[subj_df['block'].isin([block])].copy()
        blk_len = df[['subj', 'block']].groupby('block').aggregate(np.size)
        # get the video file
        assert len(set(blk_df['vname'])) == 1
        vname = blk_df['vname'].values[0]
        vpath = op.join('videos', vname)
        assert vpath in video
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
                                                          wait_for_last=True,
                                                          delay=0.02))
            # start initial stimulus
            if not ix:
                Popen([exe, '-f', '--no-audio', '--no-video-title',
                       '--play-and-exit', vpath])
                ec.wait_secs(pad)
                t_zero = ec.start_stimulus(flip=False)
            this_audio_start = t_zero + onset
            this_audio_stop = t_zero + offset
            if ix:
                ec.start_stimulus(flip=False, when=this_audio_start)
            ec.listen_presses()
            ec.wait_until(this_audio_stop)
            ec.stop()
            ec.get_presses()
            ec.trial_ok()
        ec.flush()
        ec.system_beep()
        ec.set_visible(True)
        if block == blocks - 1:
            ec.system_beep()
            msg = 'All done! We will come disconnect the EEG now.'
            max_wait = 1.5
        else:
            extra = (' This will be the last block, so the audio might end '
                     'before the cartoon does.') if block == blocks - 2 else ''
            msg = ('End of block {} of {}. Take a break now if you like, '
                   'then press {} to start the next block.'
                   ''.format(block + 1, blocks, continue_key)) + extra
            max_wait = np.inf
        ec.screen_prompt(msg, live_keys=[continue_key], max_wait=max_wait)
