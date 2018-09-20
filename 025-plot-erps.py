#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-erps.py'
===============================================================================

This script plots ERPs from labelled epochs.
"""
# @author: drmccloy
# Created on Tue May 30 11:01:12 2017
# License: BSD (3-clause)

import yaml
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op
from os import mkdir
from aux_functions import merge_features_into_df

pd.set_option('display.width', None)
np.set_printoptions(linewidth=130)
plt.ioff()

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    feature_fnames = analysis_params['feature_fnames']
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

# FILE NAMING VARIABLES
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''
cv = 'cvalign-' if align_on_cv else ''

# BASIC FILE I/O
indir = op.join('eeg-data-clean')
outdir = op.join('figures', f'erps{trunc}')
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS FROM CSV
df_cols = ['subj', 'talker', 'syll', 'train', 'wav_idx', 'wav_path']
df_types = dict(subj=int, talker=str, syll=str, train=bool, wav_idx=int)
df = pd.read_csv(op.join('params', 'master-dataframe.tsv'), sep='\t',
                 usecols=df_cols, dtype=df_types)
# choice doesn't matter; just need something to pass to the merge function:
feature_sys_fname = feature_fnames['jfh_dense']
df = merge_features_into_df(df, paramdir, feature_sys_fname)

# iterate over subjects
for subj_code, subj_num in subjects.items():
    # read epochs
    basename = '{0:03}-{1}-{2}'.format(subj_num, subj_code, cv)
    epochs = mne.read_epochs(op.join(indir, f'epochs{trunc}',
                                     basename + 'epo.fif.gz'), verbose=False)
    epochs.apply_proj()
    epochs_keys = list(epochs.event_id)
    # deal with dropped epochs
    events = mne.read_events(op.join(indir, 'events', basename + 'eve.txt'),
                             mask=None)[epochs.selection, -1]
    # subset df for this subject
    this_df = df.loc[df['subj'] == (subj_num - 1)]  # subj. dict not 0-indexed
    this_df = this_df.iloc[epochs.selection, :]
    # get consonants in nice order
    eng_cons = sorted(this_df.loc[this_df['train'], 'ascii'].unique())
    for_cons = sorted(this_df.loc[np.logical_not(this_df['train']),
                                  'ascii'].unique())
    for_cons = sorted(set(for_cons) - set(eng_cons))
    # initialize figure
    fig, axs = plt.subplots(10, 6, figsize=(36, 24))
    # iterate over consonants, putting english first
    for ix, this_ascii in enumerate(eng_cons + for_cons):
        this_cons_df = this_df.loc[this_df['ascii'] == this_ascii].copy()
        n_epochs = this_cons_df.shape[0]
        this_keys = this_cons_df['wav_path'].unique().astype(str)
        has_epochs = np.in1d(this_keys, epochs_keys)
        if not all(has_epochs):
            missing = this_keys[np.logical_not(has_epochs)]
            for m in missing:
                print('no epochs matching {}'.format(m))
        # plot ERP
        evoked = epochs[this_keys.tolist()].average()
        title = '{} ({} epochs)'.format(this_ascii, n_epochs)
        evoked.plot(spatial_colors=True, gfp=True, axes=axs.ravel()[ix],
                    titles=dict(eeg=title), show=False, selectable=False,
                    ylim=dict(eeg=[-10, 10]))
    fig.savefig(op.join(outdir, '{}erp.pdf'.format(basename)))
    plt.close(fig)
