#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'check-snr.py'
===============================================================================

This script assesses the SNR of some epoched EEG recordings, by comparing
evoked power to power during the baseline period of each trial.
"""
# @author: drmccloy
# Created on Wed Jul 19 16:31:35 PDT 2017
# License: BSD (3-clause)

import yaml
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as op
from os import mkdir

# style setup
sns.set_style('darkgrid')
plt.rc('font', family='serif', serif='Linux Libertine O')
plt.rc('mathtext', fontset='custom', rm='Linux Libertine O',
       it='Linux Libertine O:italic', bf='Linux Libertine O:bold')

# LOAD PARAMS
paramdir = 'params'
analysis_paramfile = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    n_jobs = analysis_params['n_jobs']
    scheme = analysis_params['classification_scheme']
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

# FILE NAMING VARIABLES
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''

# BASIC FILE I/O
indir = op.join('eeg-data-clean', f'epochs{trunc}')
outdir = f'processed-data-{scheme}{trunc}'
plotdir = op.join('figures', 'snr')
if not op.isdir(plotdir):
    mkdir(plotdir)

# load master dataframe
master_df = pd.read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
master_df['subj'] = master_df['subj'].apply(int)
master_df['lang'] = master_df['talker'].apply(lambda x: x[:3])

# count total trials
n_trials = master_df.groupby('subj')['trial_id'].count()
n_trials.index = [{subjects[s]: s for s in subjects}[k + 1]
                  for k in n_trials.index]
n_trials.index.name = 'subj'
n_trials.name = 'n_trials'
# count total English-talker trials
eng_df = master_df.loc[master_df['lang'] == 'eng']
n_eng_trials = eng_df.groupby('subj')['trial_id'].count()
n_eng_trials.index = n_trials.index
n_eng_trials.name = 'n_eng_trials'

# init containers
snr = dict()
n_eng_epochs = dict()

# get SNR and number of retained English-talker epochs for each subj.
for subj_code, subj in subjects.items():
    # read Epochs
    basename = '{0:03}-{1}-'.format(subj, subj_code)
    # don't use 'cvalign-' (need baseline):
    fname = op.join(indir, basename + 'epo.fif.gz')
    epochs = mne.read_epochs(fname, verbose=False)
    baseline = epochs.copy().crop(*epochs.baseline)
    stim_evoked = epochs.copy().crop(tmin=epochs.baseline[1])
    evoked_power = stim_evoked.pick_types(eeg=True).get_data().var()
    baseline_power = baseline.pick_types(eeg=True).get_data().var()
    snr[subj_code] = 10 * np.log10(evoked_power / baseline_power)
    # which retained epochs are English?
    eps = [{v: k for k, v in epochs.event_id.items()}[e]
           for e in epochs.events[:, 2]]
    eng_eps = [e for e in eps if e.startswith('eng')]
    n_eng_epochs[subj_code] = len(eng_eps)

# convert to dataframe columns
snr = pd.DataFrame.from_dict(snr, orient='index')
snr.columns = ['snr']
n_eng_epochs = pd.DataFrame.from_dict(n_eng_epochs, orient='index')
n_eng_epochs.columns = ['n_eng_epochs']

# load external data (blinks, retained epochs)
bl = pd.read_csv(op.join('eeg-data-clean', 'blinks', 'blink-summary.tsv'),
                 sep='\t')
bl = bl.set_index('subj')
ep = pd.read_csv(op.join(indir, 'epoch-summary.tsv'), sep='\t')
ep = ep.set_index('subj')

# combine into one dataframe
df = pd.concat((snr, bl, ep, n_trials, n_eng_epochs, n_eng_trials), axis=1)
df.to_csv(op.join(outdir, 'blinks-epochs-snr.tsv'), sep='\t')

# prettify column names for plotting
new_names = dict(n_blinks='Number of blinks detected',
                 n_epochs='Number of retained epochs',
                 snr='SNR: 10×$\log_{10}$(evoked power / baseline power)')
df.rename(columns=new_names, inplace=True)
df.index.name = 'Subject'

# plot
axs = df.iloc[:, :3].plot.bar(subplots=True, sharex=True, legend=False)
ax = axs[2]
xvals = ax.xaxis.get_ticklocs()
maxlines = ax.hlines(df['n_trials'], xvals - 0.4, xvals + 0.4,
                     colors='k', linestyles='dashed', linewidths=1)
ax.annotate(s='total trials', va='center',
            xy=(1, maxlines.get_segments()[-1][-1][-1]), xytext=(2, 0),
            xycoords=('axes fraction', 'data'), textcoords='offset points')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for ax, ymax in zip(axs, [9, 2500, 5000]):
    ax.set_ylim([0, ymax])
plt.tight_layout()
plt.subplots_adjust(right=0.9)
plt.savefig(op.join(plotdir, f'subject-summary{trunc}.png'))

# supplementary figure
if not truncate:
    plt.savefig(op.join('figures', 'supplement', 'subject-summary.pdf'))
