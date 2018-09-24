#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-stimulus-times.py'
===============================================================================

This script visualizes the stimulus timecourses in various ways.
"""
# @author: drmccloy
# Created on Fri Sep 21 11:15:29 PDT 2018
# License: BSD (3-clause)

import os
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# FLAGS
savefig = True
plt.ioff()
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 30)

# paths
paramdir = 'params'
outdir = os.path.join('figures', 'supplement')

# load global params
param_file = 'global-params.yaml'
with open(os.path.join(paramdir, param_file), 'r') as f:
    params = yaml.load(f)
    stim_fs = params['stim_fs']
# load colors
with open(os.path.join(paramdir, 'colors.yaml'), 'r') as cf:
    colors = yaml.load(cf)
plt.style.use(os.path.join(paramdir, 'font-charis.yaml'))
# load ASCII-IPA mapping
with open(os.path.join(paramdir, 'ascii-to-ipa.json'), 'r') as _file:
    ipadict = json.load(_file)
# load stimulus durations
stimdurs = pd.read_csv(os.path.join(paramdir, 'wav-properties.tsv'), sep='\t')
stimdurs['dur'] = stimdurs['wav_nsamp'] / stim_fs
# load C-V transition times
cvtimes = pd.read_csv(os.path.join(paramdir, 'cv-boundary-times.tsv'),
                      sep='\t')
# only English talkers
cvtimes = cvtimes.loc[cvtimes['talker'].str.startswith('eng')]
# abstract away from individual tokens
cvtimes['ascii'] = cvtimes['consonant'].transform(lambda x:
                                                  x[:-2].replace('-', '_'))
# convert to IPA
cvtimes['ipa'] = cvtimes['ascii'].map(ipadict)
# revise IPA column for English (undo the strict phonetic coding of English
# phones created during stimulus recording).
with open(os.path.join(paramdir, 'english-ascii-to-ipa.json'), 'r') as _file:
    ipadict = json.load(_file)
cvtimes['ipa'] = cvtimes['ascii'].map(ipadict)
cvtimes.rename(columns={'CV-transition-time': 'cvtime'}, inplace=True)
# merge in durations
cvtimes['path'] = cvtimes['talker'] + '/' + cvtimes['consonant'] + '.wav'
cvtimes = cvtimes.merge(stimdurs[['wav_path', 'dur']], how='inner',
                        left_on=['path'], right_on=['wav_path'])
cvtimes = cvtimes[['talker', 'consonant', 'ascii', 'ipa', 'dur', 'cvtime']]
# sort
sort_order = (cvtimes[['ipa', 'cvtime']].groupby('ipa').aggregate('mean')
              .sort_values('cvtime').index)
cvtimes['sorter'] = pd.Categorical(cvtimes['ipa'], sort_order)
cvtimes.sort_values(['sorter', 'cvtime'], inplace=True)
cvtimes.reset_index(drop=True, inplace=True)
# assign colors
n_cols = cvtimes['ipa'].unique().shape[0]
cols = list(colors.keys())
cols = cols[::3] + cols[1::3] + cols[2::3]
col_list = (cols * (n_cols // len(colors) + 1))[:n_cols]
col_dict = {k: v for k, v in zip(cvtimes['ipa'].unique(), col_list)}
cvtimes['colors'] = cvtimes['ipa'].map(col_dict)
# plot
fig = plt.figure(figsize=(7, 5))
ax = fig.add_axes((0.05, 0.1, 0.9, 0.8))
fig.subplots_adjust
ax.hlines(y=cvtimes.index, xmin=(0 - cvtimes['cvtime']),
          xmax=(cvtimes['dur'] - cvtimes['cvtime']),
          colors=cvtimes['colors'].map(colors))
ax.axvline(color='w', linewidth=0.5, linestyle='-')
ax.axvline(0.1, color='k', linewidth=0.5, linestyle='--')
# text labels
ys = (cvtimes[['ascii', 'ipa']].groupby('ipa')
      .aggregate(lambda x: x.index.values.mean())).to_dict()['ascii']
x = ax.get_xlim()[0]
for label, y in ys.items():
    ax.annotate(label, (x, y), color=colors[col_dict[label]], xytext=(8, 0),
                textcoords='offset points', ha='center', va='center')
# axes
ax.yaxis.set_visible(False)
ax.set_xlabel('time (s)')
ax.set_title('Stimuli alignment and duration')

#sns.rugplot(0 - cvtimes['cvtime'], ax=ax)
#sns.rugplot(cvtimes['dur'] - cvtimes['cvtime'], ax=ax)

# save
if savefig:
    fig.savefig(os.path.join(outdir, 'stimuli-alignments.pdf'))
else:
    plt.ion()
    plt.show()
