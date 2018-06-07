#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-dss.py'
===============================================================================

This script plots example train/test data.
"""
# @author: drmccloy
# Created on Thu Feb 22 18:00:20 PST 2018
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from expyfun.io import read_wav
from aux_functions import merge_features_into_df

# FLAGS
savefig = True
n_keep = 1
n_start = 53
dss_fs = 100.  # downsampling happened during epoching
plt.ioff()
target = 'manuscript'  # presentation or manuscript
ftype = 'svg'

if target == 'presentation':
    figure_paramfile = 'jobtalk-figure-params.yaml'
    outdir = op.join('figures', 'jobtalk')
    plt.style.use('dark_background')
else:
    figure_paramfile = 'manuscript-figure-params.yaml'
    outdir = op.join('figures', 'manuscript')
    plt.style.use({'font.serif': 'Charis SIL', 'font.family': 'serif'})

# BASIC FILE I/O
paramdir = 'params'
indir = 'eeg-data-clean'
stimdir = op.join('stimulus-generation', 'stimuli-final', 'subj-00')

# figure params
with open(op.join(paramdir, figure_paramfile), 'r') as f:
    figure_params = yaml.load(f)
    colordict = figure_params['colordict']
    axislabelsize = figure_params['axislabelsize']
    ticklabelsize = figure_params['ticklabelsize']
    axislabelcolor = figure_params['axislabelcolor']
    ticklabelcolor = figure_params['ticklabelcolor']
axislabelkwargs = dict(color=axislabelcolor, size=axislabelsize)
ticklabelkwargs = dict(color=ticklabelcolor, size=ticklabelsize)

# global params
global_paramfile = 'global-params.yaml'
with open(op.join(paramdir, global_paramfile), 'r') as f:
    global_params = yaml.load(f)
    stim_fs = global_params['stim_fs']

# analysis params
analysis_paramfile = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
    bad_channels = analysis_params['bad_channels']
    brain_resp_dur = analysis_params['brain_resp_dur']
    n_components = analysis_params['dss']['n_components']
del global_params, analysis_params

# wav params (n samples per syllable, etc)
wav_params = pd.read_csv(op.join(paramdir, 'wav-properties.tsv'), sep='\t')

# load audio file
wav_data, wav_fs = read_wav(op.join(stimdir, 'block-00.wav'))

# load trial info; subset to one subject / block
wav_df = pd.read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
wav_df = wav_df.loc[(wav_df['subj'] == 0) & (wav_df['block'] == 0)]
# convert to IPA
wav_df = merge_features_into_df(wav_df, paramdir, 'all-features.tsv')
# purge unneeded columns; keep first n tokens
wav_df = wav_df[['talker', 'syll', 'train', 'onset', 'offset', 'lang', 'ipa']]
wav_df = wav_df.iloc[n_start:(n_start + n_keep), :]
wav_df.rename(columns=dict(onset='wav_first_samp', offset='wav_last_samp'),
              inplace=True)
# fix dtypes
wav_df['wav_first_samp'] = wav_df['wav_first_samp'].map(int)
wav_df['wav_last_samp'] = wav_df['wav_last_samp'].map(int)
# color
wav_df['color'] = wav_df['talker'].map(lambda x: x[-2:]).map(colordict)

# CV boundary offsets
cvfile = 'cv-boundary-times.tsv'
cv_df = pd.read_csv(op.join(paramdir, cvfile), sep='\t')
# offset for the first syllable; one is sufficient to align wav w/ EEG
cv_offset = cv_df.loc[(cv_df['talker'] == wav_df.at[n_start, 'talker']) &
                      (cv_df['consonant'] == wav_df.at[n_start, 'syll']),
                      'CV-transition-time'].values
# merge the WAV data to get stim offset info
cv_df['key'] = cv_df['talker'] + '/' + cv_df['consonant'] + '.wav'
cv_df = cv_df.merge(wav_params, how='right', left_on='key',
                    right_on='wav_path')
# compute word and vowel durations
cv_df['w_dur'] = cv_df['wav_nsamp'] / stim_fs
cv_df['v_dur'] = cv_df['w_dur'] - cv_df['CV-transition-time']
cv_df.rename(columns={'CV-transition-time': 'wav_cv', 'wav_idx': 'event_id',
                      'wav_nsamp': 'nsamp', 'consonant': 'syll'}, inplace=True)
# get epoch tmin and tmax (relative to CV transition point)
tmin_cv = cv_df['wav_cv'].max()
tmax_cv = cv_df['v_dur'].max() + brain_resp_dur

# load DSS
subj, subj_code = 1, 'IJ'
basename = '{0:03}-{1}-'.format(subj, subj_code)
dss_fname = op.join(indir, 'dss', basename + 'cvalign-dss-data.npy')
dss = np.load(dss_fname)
data = dss[n_start:(n_start + n_keep), :n_components, :]
# load info
raw_fname = op.join(indir, 'raws-with-projs', basename + 'raw.fif.gz')
info = mne.io.read_info(raw_fname)
# load event dict
ev_fname = op.join(indir, 'events', basename + 'cvalign-eve.txt')
events = mne.read_events(ev_fname)
# scale data
y_offsets = np.arange(data.shape[1])[::-1, np.newaxis]
scaler = np.full(y_offsets.shape, 1e-2)
baselines = data[..., 0, np.newaxis]  # data.mean(axis=-1)[..., np.newaxis]
data = ((data - baselines) / scaler) + y_offsets

# embed syllable audio at right place in epoch
wav_df = wav_df.merge(cv_df[['talker', 'syll', 'wav_cv']], how='inner')
wav_df['epoch_start'] = wav_df['wav_cv'] - tmin_cv
wav_df['epoch_end'] = wav_df['wav_cv'] + tmax_cv
wav_df['on_pad'] = (wav_df['epoch_start'] * wav_fs).astype(int)
wav_df['off_pad'] = ((wav_df['epoch_end'] * wav_fs).astype(int) +
                     wav_df['wav_first_samp'] - wav_df['wav_last_samp'])

# init figure
height = 1.5 + n_keep * 1.5
n_subplots = 2 * n_keep
height_ratios = (1, 3) * n_keep
fig, axs = plt.subplots(n_subplots, 1, figsize=(3, height), sharex=True,
                        gridspec_kw=dict(height_ratios=height_ratios))
#fig.subplots_adjust(left=0.16, bottom=0.08, right=1., top=1.01, hspace=0.)
fig.subplots_adjust(left=0.08, bottom=0.16, right=1., top=1.01, hspace=0.)

# plot audio
columns = ['talker', 'ipa', 'on_pad', 'off_pad', 'wav_first_samp',
           'wav_last_samp', 'color']
for (ix, talker, ipa, on_pad, off_pad, on, off,
     color) in wav_df[columns].itertuples():
    ax = axs[ix * 2]
    on_zeros = np.zeros(abs(on_pad))
    off_zeros = np.zeros(off_pad)
    this_wav = np.concatenate((on_zeros, wav_data[0, on:off], off_zeros))
    wav_t = np.linspace(0, (len(this_wav) / wav_fs), len(this_wav))
    ax.plot(wav_t, this_wav, color=color, linewidth=0.5)
    ax.set_ylim(-0.05, 0.12)
    ax.set_yticks([0])
    ax.set_yticklabels([ipa + 'ɑ'], size=12, fontweight='bold', ha='right',
                       va='center', color=color)
    ax.tick_params(length=0, axis='y', pad=-3)
    for spine in ax.spines.values():
        spine.set_visible(False)

# plot DSS
dss_t = np.linspace(0, (data.shape[-1] / dss_fs), data.shape[-1])
for ix, (epoch, color) in enumerate(zip(data, wav_df['color'])):
    ax = axs[ix * 2 + 1]
    for channel in epoch:
        ax.plot(dss_t, channel, color=color, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if ix < (n_keep - 1):
        ax.xaxis.set_visible(False)
    # channel labels
    ax.set_yticks(y_offsets)
    yticklabels = ['dss{}'.format(x) for x in np.arange(data.shape[1]) + 1]
    ax.set_yticklabels(yticklabels, **ticklabelkwargs)
    ax.tick_params(axis='y', length=0, pad=-3)
    if n_keep > 1:
        ax.set_ylabel('trial {}'.format(ix + 1), **axislabelkwargs)

# garnish
ax.set_xlabel('time (s)', **axislabelkwargs)
xticks = np.round(np.linspace(0, 1, 6), 1)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, **ticklabelkwargs)
ax.tick_params(axis='x', length=5, pad=2.5)

# CV transition line (spans all axes)
t1 = axs[0].transData
t2 = axs[-1].transData.inverted()
ymin = axs[-1].get_ylim()[0]
ymax = t2.transform(t1.transform(axs[0].get_ylim()))[1]
ylims = axs[-1].get_ylim()
axs[-1].vlines((wav_df['wav_cv'] - wav_df['epoch_start']).iloc[0],
               ymin, ymax, colors='0.7', linewidths=0.5, clip_on=False)
axs[-1].set_ylim(*ylims)

if savefig:
    fig.savefig(op.join(outdir, 'fig-dss.{}'.format(ftype)))
else:
    plt.ion()
    plt.show()
