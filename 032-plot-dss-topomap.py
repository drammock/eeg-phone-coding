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
import mne

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    align_on_cv = analysis_params['align_on_cv']
    do_dss = analysis_params['dss']['use']
    n_comp = analysis_params['dss']['n_components']
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

# FILE NAMING VARIABLES
cv = 'cvalign-' if align_on_cv else ''
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = op.join('figures', f'dss{trunc}')

# iterate over subjects
for ix, (subj_code, subj_num) in enumerate(subjects.items()):
    basename = '{0:03}-{1}-{2}'.format(subj_num, subj_code, cv)
    # read epochs
    epochs = mne.read_epochs(op.join(indir, f'epochs{trunc}',
                                     basename + 'epo.fif.gz'))
    epochs.info['projs'] = []
    # load DSS matrix & make into projectors
    dss_mat = np.load(op.join(indir, f'dss{trunc}', basename + 'dss-mat.npy'))
    dss_mat = dss_mat[:, :n_comp]  # only retain desired number of components
    for n, component in enumerate(dss_mat.T):
        desc = f'dss-{n+1}'
        data = dict(data=component, nrow=1, ncol=len(component),
                    row_names=desc, col_names=epochs.ch_names)
        dss_proj = mne.io.Projection(data=data, active=False, kind=1,
                                     desc=desc, explained_var=None)
        epochs.add_proj(dss_proj)
    lout = mne.channels.find_layout(epochs.info)
    fig = mne.viz.plot_projs_topomap(epochs.info['projs'], layout=lout,
                                     show=False)
    fig.canvas.set_window_title(subj_code)
    fig.savefig(op.join(outdir, '{}dss{}-topomap.pdf'.format(basename,
                                                             n_comp)))
