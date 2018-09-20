#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'plot-dss-topomap.py'
===============================================================================

This script loads EEG data that has been processed with denoising source
separation (DSS) and plots scalp topomaps of the DSS components.
"""
# @author: drmccloy
# Created on Thu Aug  3 15:28:34 PDT 2017
# License: BSD (3-clause)

from os import path as op
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from aux_functions import merge_features_into_df

td_redux = True

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    feature_fnames = analysis_params['feature_fnames']
    truncate = analysis_params['eeg']['truncate']
del analysis_params

# FILE NAMING VARIABLES
cv = 'cvalign-' if align_on_cv else ''
trunc = '-truncated' if truncate else ''
nc = 'dss{}-'.format(n_comp) if do_dss else ''

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = op.join('figures', f'dss{trunc}')

# LOAD PARAMS FROM CSV
df_cols = ['subj', 'talker', 'syll', 'train', 'wav_idx', 'wav_path']
df_types = dict(subj=int, talker=str, syll=str, train=bool, wav_idx=int)
df = pd.read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
                 usecols=df_cols, dtype=df_types)
# choice doesn't matter; just need something to pass to the merge function:
feature_sys_fname = feature_fnames['jfh_dense']
df = merge_features_into_df(df, paramdir, feature_sys_fname)

# iterate over subjects
for ix, (subj_code, subj_num) in enumerate(subjects.items()):
    # load the data
    basename = '{0:03}-{1}-{2}'.format(subj_num, subj_code, cv)
    if td_redux:
        datafile_suffix = 'redux-{}data.npy'.format(nc if do_dss else 'epoch-')
        dss_data = np.load(op.join(indir, f'time-domain-redux{trunc}',
                           basename + datafile_suffix))
    else:
        datafile_suffix = 'dss-data.npy'
        dss_data = np.load(op.join(indir, f'dss{trunc}',
                                   basename + datafile_suffix))
    epochs = mne.read_epochs(op.join(indir, f'epochs{trunc}',
                                     basename + 'epo.fif.gz'), verbose=False)
    event_ids = epochs.events[:, -1]
    epochs_keys = list(epochs.event_id)

    # subset df for this subject
    this_df = df.loc[df['subj'] == (subj_num - 1)]  # subj. dict not 0-indexed
    this_df = this_df.iloc[epochs.selection, :]
    # get consonants in nice order
    eng_cons = sorted(this_df.loc[this_df['train'], 'ascii'].unique())
    for_cons = sorted(this_df.loc[np.logical_not(this_df['train']),
                                  'ascii'].unique())
    for_cons = sorted(set(for_cons) - set(eng_cons))
    # initialize figure
    fig, axs = plt.subplots(10, 6, figsize=(36, 24), sharex=True, sharey=True)
    # iterate over consonants
    for ix, this_ascii in enumerate(eng_cons + for_cons):
        this_cons_df = this_df.loc[this_df['ascii'] == this_ascii].copy()
        n_epochs = this_cons_df.shape[0]
        this_keys = this_cons_df['wav_path'].unique().astype(str)
        has_epochs = np.in1d(this_keys, epochs_keys)
        if not all(has_epochs):
            missing = this_keys[np.logical_not(has_epochs)]
            for m in missing:
                print('no epochs matching {}'.format(m))
        # find indices
        this_ev_id = [epochs.event_id[x] for x in this_keys]
        evoked = dss_data[np.in1d(event_ids, this_ev_id)].mean(axis=0)
        ax = axs.ravel()[ix]
        if td_redux:
            ax.plot(evoked.reshape(n_comp, -1).T, linewidth=0.5)
            ax.set_ylim(-0.0075, 0.0075)
        else:
            time = np.tile(epochs.times, (dss_data.shape[1], 1)).T
            ax.plot(time, evoked.T, linewidth=0.5)
            ax.set_ylim(-0.0025, 0.0025)
        ax.set_title('{} ({} epochs)'.format(this_ascii, n_epochs))
    fig.tight_layout()
    tdr = 'redux-' if td_redux else ''
    outfile = '{}dss{}-{}erp.pdf'.format(basename, n_comp, tdr)
    fig.savefig(op.join(outdir, outfile))
    plt.close(fig)
