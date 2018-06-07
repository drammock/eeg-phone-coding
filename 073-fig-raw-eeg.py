#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-raw-eeg.py'
===============================================================================

This script plots an example EEG signal.
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
from aux_functions import merge_features_into_df, plot_segmented_wav

# FLAGS
savefig = True
n_keep = 3
n_start = 52
include_stim_channel = False
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
    blu = figure_params['blu']
    bad_color = figure_params['baddatacolor']
    good_color = figure_params['gooddatacolor']
    colordict = figure_params['colordict']
    axislabelsize = figure_params['axislabelsize']
    ticklabelsize = figure_params['ticklabelsize']
    ticklabelcolor = figure_params['ticklabelcolor']

# analysis params
analysis_paramfile = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
    bad_channels = analysis_params['bad_channels']
    brain_resp_dur = analysis_params['brain_resp_dur']

# wav params (n samples per syllable, etc)
wav_params = pd.read_csv(op.join(paramdir, 'wav-properties.tsv'), sep='\t')

# load audio file
wav_data, wav_fs = read_wav(op.join(stimdir, 'block-00.wav'))

# load trial info; subset to one subject / block
wav_df = pd.read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
wav_df = wav_df.loc[(wav_df['subj'] == 0) & (wav_df['block'] == 0)]
# convert to IPA
wav_df = merge_features_into_df(wav_df, paramdir, 'all-features.tsv')
# purge unneeded columns; keep first n tokens (plus one; will be truncated)
wav_df = wav_df[['talker', 'syll', 'train', 'onset', 'offset', 'lang', 'ipa',
                 'onset_sec', 'offset_sec']]
wav_df = wav_df.iloc[n_start:(n_start + n_keep), :]
# fix dtypes
wav_df['onset'] = wav_df['onset'].map(int)
wav_df['offset'] = wav_df['offset'].map(int)
# create wav split points
wav_df['on'] = np.stack([wav_df['onset'], np.roll(wav_df['offset'], 1)],
                        axis=0).mean(axis=0).astype(int)
wav_df['off'] = np.roll(wav_df['on'], -1)
# fix first stim onset, remove extra last stim
wav_df.at[0, 'on'] = 0
wav_df = wav_df.iloc[:-1, :]

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
cv_df['w_dur'] = cv_df['wav_nsamp'] / wav_fs
cv_df['v_dur'] = cv_df['w_dur'] - cv_df['CV-transition-time']
cv_df.rename(columns={'CV-transition-time': 'wav_cv', 'wav_idx': 'event_id',
                      'wav_nsamp': 'nsamp', 'consonant': 'syll'}, inplace=True)
# get epoch tmin and tmax (relative to CV transition point)
tmin_cv = cv_df['wav_cv'].max()
tmax_cv = cv_df['v_dur'].max() + brain_resp_dur

# read Raws
subj, subj_code = 1, 'IJ'
basename = '{0:03}-{1}-'.format(subj, subj_code)
raw_fname = op.join(indir, 'raws-with-projs', basename + 'raw.fif.gz')
raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
raw.info['bads'] = bad_channels[subj_code]
raw.apply_proj()

# load event dict
ev_fname = op.join(indir, 'events', basename + 'cvalign-eve.txt')
events = mne.read_events(ev_fname)
# determine EEG time range
pad = 0.1
start = (events[n_start, 0] / raw.info['sfreq']) - cv_offset
wav_df['eeg_cv'] = events[n_start:(n_start + n_keep), 0] / raw.info['sfreq']
wav_df['eeg_tmin'] = wav_df['eeg_cv'] - tmin_cv
wav_df['eeg_tmax'] = wav_df['eeg_cv'] + tmax_cv
wav_df = wav_df.merge(cv_df[['talker', 'syll', 'wav_cv']], how='inner')
# color
wav_df['color'] = wav_df['talker'].map(lambda x: x[-2:]).map(colordict)

# get EEG data
strt = raw.time_as_index(wav_df['eeg_tmin'].iat[0] - pad)[0]
stop = raw.time_as_index(wav_df['eeg_tmax'].iat[-1] + pad)[0]
data, times = raw.get_data(start=strt, stop=stop, return_times=True)
if not include_stim_channel:
    data = data[:-1]

# scale data
y_offsets = np.arange(data.shape[0])[::-1, np.newaxis]
scaler = np.full(y_offsets.shape, 80e-6)
if include_stim_channel:
    scaler[-1] = 16
baselines = data.mean(axis=-1)[:, np.newaxis]
data = ((data - baselines) / scaler) + y_offsets

# channel colors
ch_colors = [bad_color if ch in raw.info['bads'] else good_color
             for ch in raw.ch_names]
if include_stim_channel:
    ch_colors[-1] = blu

# init figure
width = 1.5 + n_keep * 2
fig, axs = plt.subplots(2, 1, figsize=(width, 6), sharex=True,
                        gridspec_kw=dict(height_ratios=(1, 9)))
#fig.subplots_adjust(left=0.02, bottom=0.02, right=1., top=0.95, hspace=0.)
fig.subplots_adjust(left=0.1, bottom=0.01, right=1., top=0.95, hspace=0.)

# plot audio
#t_lims = (wav_df['onset_sec'].iat[0], wav_df['offset_sec'].iat[-1])
t_lims = (times[0], times[-1])
offset = (wav_df['eeg_cv'].iat[0] -
          (wav_df['onset_sec'].iat[0] + wav_df['wav_cv'].iat[0]))
plot_segmented_wav(df=wav_df, wav=wav_data, fs=wav_fs, pad=pad, offset=offset,
                   t_lims=t_lims, ax=axs[0], ann_talker=(n_keep > 1))

# plot EEG
for channel, color in zip(data, ch_colors):
    axs[1].plot(times, channel, color=color, linewidth=0.5)

# channel labels
axs[1].set_yticks(y_offsets)
axs[1].set_yticklabels(raw.ch_names, size=ticklabelsize)
for ticklabel, color in zip(axs[1].get_yticklabels(), ch_colors):
    ticklabel.set_color(color)
axs[1].tick_params(axis='y', length=0, pad=0)  # -15

# spines
axs[1].xaxis.set_visible(False)
for spine in axs[1].spines.values():
    spine.set_visible(False)

# event lines and epoch fills that span both axes
ylims = axs[1].get_ylim()
lower_ymin = 0.025
upper_ymax = 0.9  # upper axes relative coords
# transforms
t1 = axs[0].transAxes
t2 = axs[1].transAxes
t3 = axs[1].transData.inverted()
# CV-transition lines
ymin = t3.transform(t2.transform((0, lower_ymin)))[1]
ymax = t3.transform(t1.transform((0, upper_ymax)))[1]
axs[1].vlines(wav_df['eeg_cv'], ymin, ymax, colors=wav_df['color'],
              linewidths=0.5, clip_on=False)
# epoch fills
ymax = t2.inverted().transform(t1.transform((0, upper_ymax)))[1]
columns = ['eeg_tmin', 'eeg_tmax', 'color']
for ix, tmin, tmax, color in wav_df[columns].itertuples():
    axs[1].axvspan(tmin, tmax, lower_ymin, ymax, facecolor=color, alpha=0.3,
                   clip_on=False)
    if n_keep > 1:
        axs[1].annotate('trial {}'.format(ix + 1),
                        xy=((tmin + tmax)/2, lower_ymin),
                        xytext=(0, -16), textcoords='offset points',
                        ha='center', va='top', size=axislabelsize, color=color)

# now reset the ylims
axs[1].set_ylim(*ylims)

# remove facecolor (masks annotations from upper plot)
axs[1].set_facecolor('none')

if savefig:
    fig.savefig(op.join(outdir, 'fig-raw-eeg.{}'.format(ftype)))
else:
    plt.ion()
    plt.show()
