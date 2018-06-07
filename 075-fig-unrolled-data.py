#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-unrolled-data.py'
===============================================================================

This script plots example DSS signals, unrolled, with the first 3 highlighted.
"""
# @author: drmccloy
# Created on Wed Feb 28 12:50:01 PST 2018
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from aux_functions import merge_features_into_df

# FLAGS
savefig = True
n_keep = 3
n_trials = 21
dss_fs = 100.  # downsampling happened during epoching
plt.ioff()
target = 'manuscript'  # presentation or manuscript

if target == 'presentation':
    figure_paramfile = 'jobtalk-figure-params.yaml'
    outdir = op.join('figures', 'jobtalk')
    plt.style.use('dark_background')
else:
    figure_paramfile = 'manuscript-figure-params.yaml'
    outdir = op.join('figures', 'manuscript')

# BASIC FILE I/O
paramdir = 'params'
indir = 'eeg-data-clean'

# figure params
with open(op.join(paramdir, figure_paramfile), 'r') as f:
    figure_params = yaml.load(f)
    bad_color = figure_params['baddatacolor']
    good_color = figure_params['gooddatacolor']
    colordict = figure_params['colordict']
    axislabelsize = figure_params['axislabelsize']
    ticklabelsize = figure_params['ticklabelsize']
    axislabelcolor = figure_params['axislabelcolor']

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

# load trial info; subset to one subject / block
wav_df = pd.read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
wav_df = wav_df.loc[(wav_df['subj'] == 0) & (wav_df['block'] == 0)]
# convert to IPA
wav_df = merge_features_into_df(wav_df, paramdir, 'all-features.tsv')
# purge unneeded columns; keep first n tokens
wav_df = wav_df[['talker', 'syll', 'train', 'onset', 'offset', 'lang', 'ipa']]
wav_df = wav_df.iloc[:n_trials, :]
# color
wav_df['color'] = wav_df['talker'].map(lambda x: x[-2:] if x[:3] == 'eng' else
                                       x[:3]).map(colordict)
wav_df['color'].fillna(bad_color, inplace=True)  # just in case

# load DSS
subj, subj_code = 1, 'IJ'
basename = '{0:03}-{1}-'.format(subj, subj_code)
dss_fname = op.join(indir, 'dss', basename + 'cvalign-dss-data.npy')
dss = np.load(dss_fname)
data = dss[:n_trials, :n_components, :]
# load info
raw_fname = op.join(indir, 'raws-with-projs', basename + 'raw.fif.gz')
info = mne.io.read_info(raw_fname)
# load event dict
ev_fname = op.join(indir, 'events', basename + 'cvalign-eve.txt')
events = mne.read_events(ev_fname)
# unroll channels
orig_nsamp = data.shape[-1]  # use later for x-axis ticks
data = data.reshape(data.shape[0], -1)
# scale data
y_offsets = np.arange(data.shape[0])[::-1, np.newaxis]
scaler = np.full(y_offsets.shape, 2e-2)
baselines = data[..., 0, np.newaxis]  # use first value to v-align (not mean)
data = ((data - baselines) / scaler) + y_offsets

# colors: grey out the train/test data
wav_df['train_color'] = wav_df['color']
wav_df['test_color'] = wav_df['color']
wav_df['dss_color'] = wav_df['color']
wav_df.at[np.logical_not(wav_df['train']), 'train_color'] = bad_color
wav_df.at[(wav_df['lang'] != 'eng') ^ (wav_df['train']),
          'test_color'] = bad_color
wav_df.at[n_keep:, 'dss_color'] = bad_color

# prepare to loop
coltypes = ['dss_color', 'color', 'train_color', 'test_color']
figtypes = ['dss', 'unrolled', 'train', 'test']
fignames = ['fig-data-{}.pdf'.format(x) for x in figtypes]
trialcounts = ['≈ {} {} trials'.format(n, typ) for n, typ in
               zip([4700, 4700, 2700, 900],
                   ['total', 'total', 'training', 'test'])]

for column, figname, title in zip(coltypes, fignames, trialcounts):
    # init figure
    fig, ax = plt.subplots(figsize=(3, 6))
    fig.subplots_adjust(left=0.16, bottom=0.05, right=0.96, top=0.95,
                        hspace=0.)

    # plot DSS
    dss_t = np.linspace(0, (data.shape[-1] / dss_fs), data.shape[-1])
    for ix, (trial, color) in enumerate(zip(data, wav_df[column])):
        lw = 1 if ((column != 'color') and (color != bad_color)) else 0.5
        ax.plot(dss_t, trial, color=color, linewidth=lw)

    # trial labels
    wav_df['label'] = wav_df['talker'].str.split('-').str[-1].str.upper()
    wav_df.at[wav_df['lang'] != 'eng', 'label'] = wav_df['lang']
    wav_df['label'] = wav_df['label'].str.cat(wav_df['ipa'], sep=' ')
    wav_df['label'] = wav_df['label'].str.cat(['ɑ'] * wav_df.shape[0], sep='')
    ax.set_yticks(y_offsets)
    ax.set_yticklabels(wav_df['label'], size=ticklabelsize, family='serif')
    for ticklabel, color in zip(ax.get_yticklabels(), wav_df[column]):
        ticklabel.set_color(color)
        if (column != 'color') and (color != bad_color):
            ticklabel.set_fontweight('bold')
    ax.tick_params(axis='y', length=0, pad=2)

    # epoch separation lines
    ymin = y_offsets[-1, 0] - 0.5
    ymax = y_offsets[0, 0] + 0.5
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(dss_t[0], dss_t[-1])
    dss_boundaries = (np.arange(n_components)[1:]) * (orig_nsamp / dss_fs)
    ax.vlines(dss_boundaries, ymin=ymin, ymax=ymax, colors=bad_color,
              linewidths=0.5, linestyle=':')

    # ellipses
    ax.set_xticks([])
    ax.set_xlabel('...', color=axislabelcolor, size=axislabelsize, rotation=90)
    ax.set_title(title)
    # spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    if savefig:
        fig.savefig(op.join(outdir, figname))

if not savefig:
    plt.ion()
    plt.show()
