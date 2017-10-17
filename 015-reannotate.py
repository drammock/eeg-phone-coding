#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'reannotate-raws.py'
===============================================================================

This script opens and plots MNE raw files for (re)annotation. It also runs
blink detection before and after annotation, as a way of checking whether the
level of annotation is adequate to allow the blink detection algorithm to
perform reasonably well.
"""
# @author: drmccloy
# Created on Fri Apr  7 09:43:03 2017
# License: BSD (3-clause)

import yaml
import os.path as op
from os import mkdir
import mne
from mne.preprocessing import find_eog_events

# FLAGS
# set True to check behavior of blink projectors after 1st round of annotation
rerun = True
save = False

# BASIC FILE I/O
indir = 'eeg-data-clean'
outdir = op.join(indir, 'raws-annotated')
if not op.isdir(outdir):
    mkdir(outdir)

# LOAD PARAMS
paramdir = 'params'
analysis_paramfile = 'current-analysis-settings.yaml'
# analysis params
with open(op.join(paramdir, analysis_paramfile), 'r') as f:
    analysis_params = yaml.load(f)
subjects = analysis_params['subjects']
n_jobs = analysis_params['n_jobs']
skip = analysis_params['skip']
blink_chan = analysis_params['blink_channel']
ref_chan = analysis_params['eeg']['ref_channel']
del analysis_params

# iterate over subjects
for subj_code, subj_num in subjects.items():
    if subj_code in skip:
        continue
    # read Raws and events
    basename = '{0:03}-{1}-'.format(subj_num, subj_code)
    _dir = outdir if rerun else op.join(indir, 'raws')
    fname = op.join(_dir, basename + 'raw.fif.gz')
    raw = mne.io.read_raw_fif(fname, preload=True)
    # make a copy when setting reference channel
    raw_ref, _ = mne.io.set_eeg_reference(raw, ref_chan, copy=True)
    # filter
    picks = mne.pick_types(raw_ref.info, meg=False, eeg=True)
    raw_ref.filter(l_freq=0.1, h_freq=40., picks=picks, n_jobs='cuda')
    # see how well blink algorithm works before annotation
    blink_args = dict(ch_name=blink_chan[subj_code], reject_by_annotation=True)
    blink_events = find_eog_events(raw_ref, **blink_args)
    # create blink projector, so we can toggle it on and off during annotation
    blink_epochs = mne.Epochs(raw_ref, blink_events, event_id=998, tmin=-0.5,
                              tmax=0.5, proj=False, reject=None, flat=None,
                              baseline=None, picks=picks, preload=True,
                              reject_by_annotation=True)
    ssp_blink_proj = mne.compute_proj_epochs(blink_epochs, n_grad=0, n_mag=0,
                                             n_eeg=5, n_jobs=n_jobs,
                                             desc_prefix=None, verbose=None)
    raw_ref = raw_ref.add_proj(ssp_blink_proj)
    # interactive annotation: mark bad channels & transient noise during blocks
    raw_ref.plot(n_channels=33, duration=30, events=blink_events, block=True,
                 scalings=dict(eeg=50e-6))
    # compare blink algorithm performance after annotation
    new_blink_events = find_eog_events(raw_ref, **blink_args)
    print('#############################')
    print('old: {}'.format(blink_events.shape[0]))
    print('new: {}'.format(new_blink_events.shape[0]))
    print('#############################')
    if save:
        # copy the annotation data to the (unfiltered, unreferenced) Raw object
        raw.annotations = raw_ref.annotations
        # save annotated Raw object. If re-running, change overwrite to True
        raw.save(op.join(outdir, basename + 'raw.fif.gz'), overwrite=rerun)
