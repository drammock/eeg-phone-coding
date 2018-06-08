#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-methods-diagram.py'
===============================================================================

This script plots a diagram of the analysis pipeline.
"""
# @author: drmccloy
# Created on Fri Oct 20 15:38:57 PDT 2017
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
import mne
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.misc import imread
from expyfun.io import read_wav
from aux_functions import (merge_features_into_df, plot_featmat, plot_confmat,
                           format_colorbar_percent)

# flags
onoff = 'off'
despine_kwargs = dict(top=True, bottom=True, left=True, right=True)
box_around = True
box_target = 'voiced'
arrowcolor = '0.5'

# figure params
paramdir = 'params'
with open(op.join(paramdir, 'manuscript-figure-params.yaml'), 'r') as f:
    figure_params = yaml.load(f)
    red = figure_params['red']
    blu = figure_params['blu']
    bad_color = figure_params['baddatacolor']
    good_color = figure_params['gooddatacolor']
    axislabelsize = figure_params['axislabelsize']
    ticklabelsize = figure_params['ticklabelsize']
    axislabelcolor = figure_params['axislabelcolor']
    ticklabelcolor = figure_params['ticklabelcolor']
    feature_order = figure_params['feat_order']
axislabelkwargs = dict(color=axislabelcolor, size=axislabelsize)
ticklabelkwargs = dict(color=ticklabelcolor, size=ticklabelsize)
# analysis params
analysis_paramfile = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
    bad_channels = analysis_params['bad_channels']
    brain_resp_dur = analysis_params['brain_resp_dur']
    n_components = analysis_params['dss']['n_components']
    feature_fnames = analysis_params['feature_fnames']
    canonical_phone_order = analysis_params['canonical_phone_order']['eng']

# init figure
plt.ioff()
plt.style.use({'font.serif': 'Charis SIL', 'font.family': 'serif'})
fig = plt.figure(figsize=(7.5, 6))


# # # # # # # # # # # #
# stimulus / EEG prep #
# # # # # # # # # # # #
stimdir = op.join('stimulus-generation', 'stimuli-final', 'subj-00')
wav_data, fs = read_wav(op.join(stimdir, 'block-00.wav'))
n_keep = 3    # how many stims to plot
n_start = 52  # which stim to start at
colors = ('0.75', red, '0.75')
# load trial info
df = pd.read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
df = merge_features_into_df(df, paramdir, 'all-features.tsv')
# purge unneeded columns & rows
df = df[['talker', 'syll', 'train', 'onset', 'offset', 'lang', 'ipa',
         'onset_sec', 'offset_sec']]
df = df.iloc[n_start:(n_start + n_keep + 1), :]
# create wav split points
df['on'] = np.stack([df['onset'], np.roll(df['offset'], 1)], axis=0
                    ).mean(axis=0).astype(int)
df['off'] = np.roll(df['on'], -1)
# undo "roll" for first stim onset, remove extra last stim
df.at[n_start, 'on'] = df.at[n_start, 'onset'] - fs / 5
df = df.iloc[:-1, :]
df[['onset', 'offset']] = df[['onset', 'offset']].astype(int)
# annotation setup
zeropad = int(df.at[n_start, 'on'])
nsamp = int(df['off'].iloc[-1])
minimum = wav_data[0, :nsamp].min()
maximum = wav_data[0, :nsamp].max()
df['tcode'] = df['talker'].str.split('-').map(lambda x: x[1].upper()
                                              if x[0] == 'eng' else x[0])
df['color'] = colors


# # # # # # # # # # # # # # # # #
# raw EEG traces w/ epoch lines #
# # # # # # # # # # # # # # # # #
pad = 0.1
subj, subj_code = 1, 'IJ'
include_stim_channel = False
# subset df (pandas loc indexing *includes* the last index, so -1)
df = df.loc[n_start:(n_start + n_keep - 1)]
# merge in CV transition time
wav_df = pd.read_csv(op.join(paramdir, 'wav-properties.tsv'), sep='\t')
cv_df = pd.read_csv(op.join(paramdir, 'cv-boundary-times.tsv'), sep='\t')
cv_df['key'] = cv_df['talker'] + '/' + cv_df['consonant'] + '.wav'
cv_df = cv_df.merge(wav_df, how='right', left_on='key', right_on='wav_path')
cv_df.rename(columns={'CV-transition-time': 'wav_cv'}, inplace=True)
cv_df['w_dur'] = cv_df['wav_nsamp'] / fs
cv_df['v_dur'] = cv_df['w_dur'] - cv_df['wav_cv']
cv_df.drop(['wav_path', 'key'], axis=1, inplace=True)
df = df.reset_index().merge(cv_df, how='left', left_on=['talker', 'syll'],
                            right_on=['talker', 'consonant']
                            ).set_index('index')
df.index.name = None
df.drop('consonant', axis=1, inplace=True)
# load event dict
basename = '{0:03}-{1}-'.format(subj, subj_code)
events = mne.read_events(op.join('eeg-data-clean', 'events',
                                 basename + 'cvalign-eve.txt'))
# read raw
fpath = op.join('eeg-data-clean', 'raws-with-projs', basename + 'raw.fif.gz')
raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)
raw.info['bads'] = bad_channels[subj_code]
raw.apply_proj()
# extract EEG time range
df['eeg_cv'] = events[n_start:(n_start + n_keep), 0] / raw.info['sfreq']
df['eeg_tmin'] = df['eeg_cv'] - cv_df['wav_cv'].max()
df['eeg_tmax'] = df['eeg_cv'] + cv_df['v_dur'].max() + brain_resp_dur
strt = raw.time_as_index(df['eeg_tmin'].iat[0] - pad)[0]
stop = raw.time_as_index(df['eeg_tmax'].iat[-1] + pad)[0]
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
eeg_axes = fig.add_axes((0, 0, 0.4, 0.95))
# plot EEG
for channel, color in zip(data, ch_colors):
    eeg_axes.plot(times, channel, color=color, linewidth=0.5)
eeg_axes.axis(onoff)
'''
# channel labels
eeg_axes.set_yticks(y_offsets)
eeg_axes.set_yticklabels(raw.ch_names, size=ticklabelsize)
for ticklabel, color in zip(eeg_axes.get_yticklabels(), ch_colors):
    ticklabel.set_color(color)
eeg_axes.tick_params(axis='y', length=0, pad=0)  # -15
# spines
eeg_axes.xaxis.set_visible(False)
for spine in eeg_axes.spines.values():
    spine.set_visible(False)
'''
# data-to-axes transform
x1 = eeg_axes.transData.transform
x2 = eeg_axes.transAxes.inverted().transform
top = x2(x1((0, y_offsets.max() - 3.)))[1]
columns = ['tcode', 'ipa', 'on', 'off', 'eeg_tmin', 'eeg_tmax', 'color']
for ix, talker, ipa, tmin, tmax, emin, emax, color in df[columns].itertuples():
    # highlight epochs
    if ix == n_start + (n_keep // 2):
        eeg_axes.axvspan(emin, emax, 0.02, top, facecolor=color, alpha=0.3,
                         clip_on=False)
    # plot audio
    this_wav = wav_data[0, int(tmin):int(tmax)]
    nsamp = this_wav.shape[0]
    scaler = 100 * (this_wav.max() - this_wav.min())
    y_offset = y_offsets.max() + 4
    this_wav = scaler * this_wav + y_offset
    this_times = np.linspace(emin, emax, nsamp)
    line = eeg_axes.plot(this_times, this_wav, color=color, linewidth=0.5)
    if ix == n_start + (n_keep // 2):
        line[0].set_zorder(20)
    # annotate
    ann_kwargs = dict(textcoords='offset points', va='baseline',
                      color=color, size=14, fontweight='bold',
                      clip_on=False, family='serif', xytext=(0, 6))
    xy = (this_times.mean(), maximum + y_offset)
    eeg_axes.annotate(ipa, xy=xy, ha='right', **ann_kwargs)  # consonant
    eeg_axes.annotate('ɑ', xy=xy, ha='left', **ann_kwargs)   # vowel
eeg_axes.axis(onoff)


# # # # # # # # # # # # # # # # #
# image of EEG sensors and head #
# # # # # # # # # # # # # # # # #
head = imread(op.join('figures', 'manuscript', 'eeg-sensors.png'), mode='RGBA')
head = np.fliplr(head)  # put ear on left
aspect = head.shape[1] / head.shape[0]
width = 0.16
height = width / aspect
x = eeg_axes.get_position().x1 - 0.05
head_axes = fig.add_axes((x, 1 - height, width, height))
head_axes.imshow(head)
head_axes.axis(onoff)
eeg_axes.set_zorder(2)


# # # # # # # # # #
# feature matrix  #
# # # # # # # # # #
top = 0.99
left = head_axes.get_position().x1 + 0.12
height = head_axes.get_position().height
width = 0.99 - left
feat_axes = fig.add_axes((left, top - height, width, height))
# colors
cmap = LinearSegmentedColormap.from_list(name='tol', N=2,
                                         colors=['0.85', '0.55'])
cmap.set_bad('1')
gridcol = '0.9'
gridlwd = 1.1
ticklabcol = 'k'
# load data
featsys = 'phoible_redux'
featmat = pd.read_csv(op.join(paramdir, feature_fnames[featsys]),
                      sep='\t', index_col=0, comment="#")
featmat = featmat.loc[canonical_phone_order]  # remove engma
featmat = featmat[feature_order[featsys]]     # quasi-consistent order
# plot
plot_featmat(featmat.T, ax=feat_axes, cmap=cmap, title='')
feat_axes.tick_params(length=0, which='both', labelsize=8)
# grid
feat_axes.grid(which='minor', axis='y', color=gridcol, linewidth=gridlwd)
feat_axes.grid(which='minor', axis='x', color=gridcol, linewidth=gridlwd)
# box
if box_around:
    # individual boxes around relevant cells
    for ipa in df['ipa']:
        x0 = featmat.index.tolist().index(ipa)
        y0 = featmat.columns.tolist().index(box_target)
        xy = np.array([x0, y0]) - 0.5
        rect = Rectangle(xy=xy, height=1, width=1, facecolor='none', zorder=5,
                         edgecolor='k', linewidth=2)
        feat_axes.add_artist(rect)
    '''
    # box around whole row
    xy = np.array([0, featmat.columns.tolist().index(box_target)]) - 0.5
    rect = Rectangle(xy=xy, height=1, width=featmat.shape[0],
                     facecolor='none', zorder=10,
                     edgecolor='k', linewidth=2, clip_on=False)
    feat_axes.add_artist(rect)
    '''


# # # # # # # # # # # #
# arrow: head to EEG  #
# # # # # # # # # # # #
arrow_dict = dict(arrowstyle='simple', fc=arrowcolor, ec='none',
                  connectionstyle='angle3,angleA=90,angleB=180')
p = ConnectionPatch(xyA=(0.5, 0), xyB=(0.2, -0.2), coordsA='axes fraction',
                    coordsB='axes fraction', axesA=head_axes,
                    axesB=head_axes, **arrow_dict)
head_axes.add_artist(p)
'''
top = head_axes.get_position().y0
left = eeg_axes.get_position().x1
width = head_axes.get_position().width / 2
height = width / 2
arrow_axes = fig.add_axes((left, top - height, width, height))
arrow_axes.annotate('', xy=(0, 0), xycoords='data',
                    xytext=(0.5, 1), textcoords='data',
                    arrowprops=arrow_dict)
arrow_axes.axis(onoff)
'''


# # # #
# DSS #
# # # #
top = feat_axes.get_position().y0 - 0.1
left = eeg_axes.get_position().x1 + 0.03
height = eeg_axes.get_position().height / 5
width = 0.2
dss_axes = fig.add_axes((left, top - height, width, height))
# load data
dss_fname = op.join('eeg-data-clean', 'dss', basename + 'cvalign-dss-data.npy')
dss = np.load(dss_fname)
data = dss[n_start:(n_start + n_keep), :n_components, :]
# scale data
dss_y_offsets = np.arange(data.shape[1])[::-1, np.newaxis]
scaler = np.full(dss_y_offsets.shape, 1e-2)
baselines = data[..., 0, np.newaxis]
offset_data = ((data - baselines) / scaler) + dss_y_offsets
# time vector
dss_fs = 100.  # downsampling happened during epoching
dss_t = np.linspace(0, (data.shape[-1] / dss_fs), data.shape[-1])

for (ix, color), epoch in zip(df[['color']].itertuples(), offset_data):
    if ix == n_start + (n_keep // 2):
        for channel in epoch:
            dss_axes.plot(dss_t, channel, color=color, linewidth=0.5)
        # channel labels
        '''
        dss_axes.set_yticks(dss_y_offsets)
        yticklabels = ['dss{}'.format(x) for x in np.arange(data.shape[1]) + 1]
        dss_axes.set_yticklabels(yticklabels, **ticklabelkwargs)
        dss_axes.tick_params(axis='y', length=0, pad=-3)
sns.despine(ax=dss_axes, **despine_kwargs)
dss_axes.set_xticks([])
'''
dss_axes.axis(onoff)


# # # # # # # # #
# unrolled DSS  #
# # # # # # # # #
bottom = dss_axes.get_position().y0
width = dss_axes.get_position().width
height = dss_axes.get_position().height / 2
left = 1 - width
trial_axes = fig.add_axes((left, bottom + 0.4 * height, width, height))
# plot unrolled DSS
y_offsets = np.arange(data.shape[0])[::-1, np.newaxis]
scaler = np.full(y_offsets.shape, 2e-2)
this_data = data.reshape(data.shape[0], -1)
baselines = this_data[..., 0, np.newaxis]
this_data = ((this_data - baselines) / scaler) + y_offsets
for (ix, color), trial in zip(df[['color']].itertuples(), this_data):
    trial_axes.plot(trial, color=color, linewidth=0.5)
# ticks
tl = featmat.loc[df['ipa'], box_target].reset_index()
tl[box_target] = tl[box_target].map(str)
ticklabels = tl.apply(lambda x: ': '.join(x), axis=1)
trial_axes.set_yticks(y_offsets)
trial_axes.set_yticklabels(ticklabels, size=ticklabelsize)
for ticklabel, color in zip(trial_axes.get_yticklabels(), df['color']):
    ticklabel.set_color(color)
# vlines
for ix in np.arange(data.shape[1]):
    if ix:
        trial_axes.axvline(x=(ix * data.shape[-1]), linewidth=0.5, alpha=0.5,
                           color='0.5', linestyle='dotted')
trial_axes.tick_params(axis='y', length=0, pad=-3)
sns.despine(ax=trial_axes, **despine_kwargs)
trial_axes.set_xticks([])


# # # # # # # # # # # # # # # # # # # # # # #
# arrows connecting featmat and trial axes  #
# # # # # # # # # # # # # # # # # # # # # # #
from_data = feat_axes.transData.transform
to_axes = trial_axes.transAxes.inverted().transform
conns1 = ['arc3,rad=0.55', 'arc3,rad=-0.05', 'arc3,rad=0.45']
conns2 = ['arc3,rad=-0.55', 'arc3,rad=0.05', 'arc3,rad=-0.45']
arrowstyles = ('-', '-|>', '-')

for ix, (ipa, conn1, conn2, arrowstyle) in enumerate(zip(df['ipa'], conns1,
                                                         conns2,
                                                         arrowstyles)):
    x0 = featmat.index.tolist().index(ipa)
    y0 = max(feat_axes.get_ylim()) + 2
    origin = to_axes(from_data((x0, y0)))
    dest = np.array((0, 1))
    halfway = (origin + dest) / 2
    # merge b and k at same point
    if ipa == 'k':
        midpt = halfway
    elif ipa == 'b':
        halfway = midpt
    p = ConnectionPatch(xyA=origin, xyB=halfway, coordsA='axes fraction',
                        coordsB='axes fraction', axesA=trial_axes,
                        axesB=trial_axes, arrowstyle='-', color=arrowcolor,
                        connectionstyle=conn1)
    trial_axes.add_artist(p)
    if ipa != 'b':
        q = ConnectionPatch(xyA=halfway, xyB=dest, coordsA='axes fraction',
                            coordsB='axes fraction', axesA=trial_axes,
                            axesB=trial_axes, arrowstyle=arrowstyle,
                            color=arrowcolor, connectionstyle=conn2)
        trial_axes.add_artist(q)
    if ipa == 'ð':
        trial_axes.annotate('training\nlabels', xy=halfway, xytext=(6, 0),
                            xycoords='axes fraction', va='baseline',
                            ha='left', textcoords='offset points')


# # # # # # # # # # #
# arrow: EEG to DSS #
# # # # # # # # # # #
arrow_dict = dict(arrowstyle='simple', fc=arrowcolor, ec='none')
dss_axes.annotate('', xy=(0, 0.5), xycoords='axes fraction',
                  xytext=(-0.2, 0.5), textcoords='axes fraction',
                  arrowprops=arrow_dict)
dss_axes.annotate('DSS', xy=(-0.1, 0.5), xycoords='axes fraction',
                  xytext=(0, 5), textcoords='offset points', ha='center',
                  va='baseline')


# # # # # # # # # # # #
# arrow: DSS to trial #
# # # # # # # # # # # #
from_data = dss_axes.transData.transform
to_axes = dss_axes.transAxes.inverted().transform
conns = ('arc3,rad=0.2', 'arc3,rad=0.15', 'arc3,rad=0.05', 'arc3,rad=-0.05',
         'arc3,rad=-0.15')
arrowstyles = ('-', '-', '-', '-', '-|>')
for y, conn, arrowstyle in zip(dss_y_offsets, conns, arrowstyles):
    yA = to_axes(from_data((1, y)))[1]
    p = ConnectionPatch(axesA=dss_axes, axesB=trial_axes,
                        coordsA='axes fraction', coordsB='axes fraction',
                        xyA=(1, yA), xyB=(-0.11, 0.5),
                        arrowstyle=arrowstyle, color=arrowcolor,
                        connectionstyle=conn)
    trial_axes.add_artist(p)


# # # # # # # # # # #
# confusion matrix  #
# # # # # # # # # # #
right = 0.95
bottom = 0.08
top = dss_axes.get_position().y0 - 0.05
height = top - bottom
conf_axes = fig.add_axes((right - height, bottom, height, height))
fname = ('row-ordered-eer-confusion-matrix-nonan-eng-cvalign-dss5-'
         '{}-average.tsv'.format(featsys))
fpath = op.join('processed-data-logistic', 'ordered-confusion-matrices', fname)
confmat = pd.read_csv(fpath, sep='\t', index_col=0)
ax = plot_confmat(confmat, ax=conf_axes, cmap='viridis',
                  norm=LogNorm(vmin=1e-5, vmax=1))
# axis labels
ax.set_ylabel('Stimulus phoneme')
ax.set_xlabel('Predicted phoneme')
conf_axes.yaxis.tick_right()
conf_axes.yaxis.set_label_position('right')
conf_axes.tick_params(which='both', labelsize=7)


# # # # # # #
# colorbar  #
# # # # # # #
right = conf_axes.get_position().x0 + 0.01
bottom = conf_axes.get_position().y0
top = conf_axes.get_position().y1
width = conf_axes.get_position().width / 18
cbar_axes = fig.add_axes((right - width, bottom, width, top - bottom))
cbar = fig.colorbar(conf_axes.images[0], cax=cbar_axes,
                    orientation='vertical')
cbar.outline.set_linewidth(0.2)
# scale on left
cbar_axes.yaxis.tick_left()
cbar_axes.yaxis.set_label_position('left')
cbar_axes.tick_params(which='both', labelsize=7, width=0.2, colors='0.5')
# colorbar ticks
cuts = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
tks = np.array([np.linspace(a, b, 9, endpoint=False)
                for a, b in zip(cuts[:-1], cuts[1:])
                ]).flatten().tolist() + [1]  # logspace ticks
cbar.set_ticks(tks)
# pretty-format the ticklabels
ticklabs = [l.get_text() for l in cbar_axes.get_yticklabels()]
ticklabs = format_colorbar_percent(ticklabs)
cbar.set_ticklabels(ticklabs)
cbar.set_label('Prediction scores', labelpad=0)


# # # # # # # # # # # # # #
# arrow: trial to confmat #
# # # # # # # # # # # # # #
arrow_dict = dict(arrowstyle='simple', fc=arrowcolor, ec='none')
trial_axes.annotate('', xy=(0.1, -0.8), xycoords='axes fraction',
                    xytext=(0.1, 0), textcoords='axes fraction',
                    arrowprops=arrow_dict)
trial_axes.annotate('classify &\naggregate', xy=(0.1, -0.4),
                    xycoords='axes fraction', xytext=(6, 0),
                    textcoords='offset points', ha='left', va='center')


# # # # # # # # # #
# subplot labels  #
# # # # # # # # # #
kwargs = dict(fontsize=14, fontweight='bold', ha='right', va='baseline',
              clip_on=False)
eeg_axes.text(0.1, 1, 'A', transform=eeg_axes.transAxes, **kwargs)
eeg_axes.text(0.1, 0.85, 'B', transform=eeg_axes.transAxes, **kwargs)
feat_axes.text(-0.25, 0.9, 'D', transform=feat_axes.transAxes, **kwargs)
trial_axes.text(0.95, 1, 'E', transform=trial_axes.transAxes, **kwargs)
kwargs.update(ha='left')
dss_axes.text(0, 1, 'C', transform=dss_axes.transAxes, **kwargs)
conf_axes.text(0, 1.05, 'F', transform=conf_axes.transAxes, **kwargs)

fig.savefig(op.join('figures', 'manuscript', 'fig-methods-diagram.pdf'))
