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

# BASIC FILE I/O
indir = op.join('eeg-data-clean', 'epochs')
outdir = 'processed-data'
plotdir = op.join('figures', 'snr')
if not op.isdir(plotdir):
    mkdir(plotdir)

# LOAD PARAMS
paramdir = 'params'
analysis_paramfile = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    n_jobs = analysis_params['n_jobs']
del analysis_params

# load master dataframe
master_df = pd.read_csv(op.join(paramdir, 'master-dataframe.tsv'), sep='\t')
master_df['subj'] = master_df['subj'].apply(int)
n_trials = master_df.groupby('subj')['trial_id'].count()
n_trials.index = [{subjects[s]: s for s in subjects}[k + 1]
                  for k in n_trials.index]
n_trials.index.name = 'subj'
n_trials.name = 'n_trials'

# init containers
snr = dict()

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

# convert to dataframe columns
snr = pd.DataFrame.from_dict(snr, orient='index')
snr.columns = ['snr']

# load external data (blinks, retained epochs)
bl = pd.read_csv(op.join('eeg-data-clean', 'blinks', 'blink-summary.tsv'),
                 sep='\t')
bl = bl.set_index('subj')
ep = pd.read_csv(op.join(indir, 'epoch-summary.tsv'), sep='\t')
ep = ep.set_index('subj')

# combine into one dataframe
df = pd.concat((bl, ep, snr, n_trials), axis=1)
df.to_csv(op.join(outdir, 'blinks-epochs-snr.tsv'), sep='\t')

# prettify column names for plotting
new_names = dict(n_blinks='Number of blinks detected',
                 n_epochs='Number of retained epochs',
                 snr='SNR: 10*log(evoked power / baseline power)')
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
plt.savefig(op.join(plotdir, 'subject-summary.png'))
