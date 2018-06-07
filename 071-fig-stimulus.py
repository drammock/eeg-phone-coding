#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-stimulus.py'
===============================================================================

This script plots an example auditory stimulus.
"""
# @author: drmccloy
# Created on Wed Feb 21 17:15:28 PST 2018
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from expyfun.io import read_wav
from aux_functions import merge_features_into_df

# FLAGS
savefig = True
n_keep = 3
n_start = 52
plt.ioff()
target = 'manuscript'  # presentation or manuscript
ftype = 'svg'

if target == 'presentation':
    outdir = op.join('figures', 'jobtalk')
    figure_paramfile = 'jobtalk-figure-params.yaml'
else:
    outdir = op.join('figures', 'manuscript')
    figure_paramfile = 'manuscript-figure-params.yaml'


# BASIC FILE I/O
stimdir = op.join('stimulus-generation', 'stimuli-final', 'subj-00')
paramdir = 'params'

# figure params
with open(op.join(paramdir, figure_paramfile), 'r') as f:
    figure_params = yaml.load(f)
    colordict = figure_params['colordict']

colordict.update(m1='0.75')

# load trial info
df = pd.read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
# subset to one subject / block
df = df.loc[(df['subj'] == 0) & (df['block'] == 0)]
# convert to IPA
df = merge_features_into_df(df, paramdir, 'all-features.tsv')
# purge unneeded columns
df = df[['talker', 'syll', 'train', 'onset', 'offset', 'lang', 'ipa']]
# reduce to first N tokens
df = df.iloc[n_start:(n_start + n_keep + 1), :]
# fix dtype
df['onset'] = df['onset'].map(int)
df['offset'] = df['offset'].map(int)

# create wav split points
df['on'] = np.stack([df['onset'], np.roll(df['offset'], 1)], axis=0
                    ).mean(axis=0).astype(int)
df['off'] = np.roll(df['on'], -1)

# color
df['color'] = df['talker'].map(lambda x: x[-2:] if x[:3] == 'eng'
                               else x[:-2]).map(colordict)

# load wav file
wav_data, fs = read_wav(op.join(stimdir, 'block-00.wav'))

# fix first stim onset, remove extra last stim
df.at[n_start, 'on'] = df.at[n_start, 'onset'] - fs / 5
df = df.iloc[:-1, :]

# annotation setup
zeropad = int(df.at[n_start, 'on'])  # - fs / 5)
nsamp = int(df['off'].iloc[-1])
minimum = wav_data[0, :nsamp].min()
maximum = wav_data[0, :nsamp].max()
df['tcode'] = df['talker'].str.split('-').map(lambda x: x[1].upper()
                                              if x[0] == 'eng' else x[0])

# init figure
if target == 'presentation':
    plt.style.use(['dark_background',
                   op.join(paramdir, 'matplotlib-style-jobtalk.yaml')])
fig, ax = plt.subplots(1, 1, figsize=(4, 1.5))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.15, top=0.78)

# plot
columns = ['tcode', 'ipa', 'on', 'off', 'color']
for ix, talker, ipa, tmin, tmax, color in df[columns].itertuples():
    this_wav = wav_data[0, int(tmin):int(tmax)]
    if not ix:
        tmin = zeropad
        this_wav = np.concatenate([np.zeros(abs(tmin),), this_wav])
    this_times = np.arange(tmin, tmax)
    ax.plot(this_times, this_wav, color=color, linewidth=1)

    # annotate
    ann_kwargs = dict(textcoords='offset points', va='baseline',
                      color=color, size=14, fontweight='bold',
                      clip_on=False, family='serif', xytext=(0, 6))
    xy = (this_times.mean(), maximum)
    # consonant
    ax.annotate(ipa, xy=xy, ha='right', **ann_kwargs)
    # vowel
    ax.annotate('ɑ', xy=xy, ha='left', **ann_kwargs)
    # talker
    ann_kwargs.update(va='top', xytext=(0, -4), size=10, family='sans-serif')
    ax.annotate(talker, xy=(xy[0], minimum), ha='center', **ann_kwargs)

ax.set_xlim(zeropad, nsamp)
ax.axis('off')

if savefig:
    fig.savefig(op.join(outdir, 'fig-stimulus.{}'.format(ftype)))
else:
    plt.ion()
    plt.show()
