#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'fig-training-labels.py'
===============================================================================

This script plots training labels.
"""
# @author: drmccloy
# Created on Thu Mar  1 16:49:00 PST 2018
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aux_functions import merge_features_into_df

# FLAGS
savefig = True
n_trials = 21
feat_sys = 'phoible_redux'
n_feats = 4
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
feature_sys_fname = 'all-features.tsv'

# figure params
with open(op.join(paramdir, figure_paramfile), 'r') as f:
    figure_params = yaml.load(f)
    bad_color = figure_params['baddatacolor']
    good_color = figure_params['gooddatacolor']
    colordict = figure_params['colordict']
    axislabelsize = figure_params['axislabelsize']
    ticklabelsize = figure_params['ticklabelsize']
    axislabelcolor = figure_params['axislabelcolor']

# analysis params
analysis_paramfile = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
    feature_systems = analysis_params['feature_systems']
    feature_mappings = analysis_params['feature_mappings']
    canonical_phone_order = analysis_params['canonical_phone_order']

# reduce to the feature system we are using
feat_sys_names = dict(jfh_sparse='PSA', spe_sparse='SPE',
                      phoible_redux='PHOIBLE')
this_feat_sys = feature_systems[feat_sys]
this_feat_map = feature_mappings[feat_sys]
this_feat_sys_name = feat_sys_names[feat_sys]
this_feat_abbrevs = dict(consonantal='cons.', labial='lab.',
                         continuant='cont.', voiced='voi.',
                         coronal='cor.', dorsal='dors.',
                         anterior='anter.', sonorant='son.',
                         distributed='distr.', strident='strid.')

# load trial info; subset to one subject / block
wav_df = pd.read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
wav_df = wav_df.loc[(wav_df['subj'] == 0) & (wav_df['block'] == 0)]
# convert to IPA & add feature columns
wav_df = merge_features_into_df(wav_df, paramdir, 'all-features.tsv')
# purge unneeded columns; keep first n tokens
drop_cols = ['subj', 'block', 'trial_id', 'ttl_id', 'onset', 'offset',
             'onset_sec', 'offset_sec', 'wav_path', 'wav_idx', 'wav_nsamp',
             'vid_name', 'vid_dur']
wav_df.drop(labels=drop_cols, axis=1, inplace=True)
wav_df = wav_df.iloc[:n_trials, :]

# color of each trial
wav_df['color'] = wav_df['talker'].map(lambda x: x[-2:] if x[:3] == 'eng' else
                                       x[:3]).map(colordict)
wav_df['color'].fillna(bad_color, inplace=True)  # just in case
# colors: grey out the train/test data
wav_df['train_color'] = wav_df['color']
wav_df['test_color'] = wav_df['color']
wav_df.at[np.logical_not(wav_df['train']), 'train_color'] = bad_color
wav_df.at[(wav_df['lang'] != 'eng') ^ (wav_df['train']),
          'test_color'] = bad_color

# select just the features we need
featmat = wav_df[this_feat_sys].iloc[:, :n_feats]
featmat.rename(columns=this_feat_map, inplace=True)
featmat.index = wav_df['ipa']

# init figure
fig, ax = plt.subplots(figsize=(1.2, 6))
fig.subplots_adjust(left=0.2, bottom=0.05, right=0.85, top=0.95, hspace=0.)
ax.set_xlim(-0.5, n_feats - 0.5)
ax.set_ylim(-0.5, n_trials - 0.5)

# plot featmat
y_offsets = np.arange(n_trials)[::-1]
for y, row, color in zip(y_offsets, featmat.itertuples(),
                         wav_df['train_color']):
    for x, value in enumerate(row[1:]):  # only plot training / test rows
        if color != bad_color:           # don't draw sparse cells
            kwargs = dict(textcoords='offset points', va='center', color=color)
            # training labels / test predictions
            if not np.isnan(value):
                ax.annotate(str(int(value)), xy=(x, y), xytext=(0, -1),
                            ha='center', fontweight='bold',
                            size=ticklabelsize + 2, **kwargs)
            # ellipses at end of each row
            new_y = ax.transData.transform((0, y))
            new_y = ax.transAxes.inverted().transform(new_y)[1]
            ax.annotate('...', xy=(1, new_y), xycoords='axes fraction',
                        xytext=(0, 0.25), ha='left', size=axislabelsize,
                        **kwargs)
# feat names
for x, feat in enumerate(featmat.columns):
    ax.annotate(this_feat_abbrevs[feat], xy=(x, y_offsets[0]), xytext=(0, 8),
                textcoords='offset points', ha='center', va='bottom',
                rotation=90, color=axislabelcolor, size=ticklabelsize)
# garnish y
ax.set_yticks(y_offsets)
ax.set_yticklabels(wav_df['ipa'], size=ticklabelsize + 2, family='serif')
for ticklabel, color in zip(ax.get_yticklabels(), wav_df['train_color']):
    if color == bad_color:
        ticklabel.set_color('k')
    else:
        ticklabel.set_color(color)
        ticklabel.set_fontweight('bold')
ax.tick_params(axis='y', length=0, pad=3)
# lower ellipses
ax.set_xticks([])
ax.set_xlabel('...', color=axislabelcolor, size=axislabelsize, rotation=90)
ax.annotate('...', xy=(1, 1), xycoords='axes fraction', xytext=(0, 8),
            textcoords='offset points', ha='left', va='center',
            color=axislabelcolor, size=axislabelsize)
# spines
for spine in ax.spines.values():
    spine.set_visible(False)

# save
if savefig:
    fig.savefig(op.join(outdir, 'fig-train-labels.pdf'))
else:
    plt.ion()
    plt.show()
