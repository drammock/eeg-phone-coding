# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'analyze-logs.py'
===============================================================================

This script analyzes the expyfun log to make sure there were no anomalies.
"""
# @author: drmccloy
# Created on Mon Feb 29 17:18:25 2016
# License: BSD (3-clause)


import numpy as np
import pandas as pd
from expyfun.io import read_tab
from expyfun.stimuli import get_tdt_rates
from glob import glob
from ast import literal_eval
from os import path as op
from os import mkdir

paramdir = 'params'
paramfile = 'master-dataframe.tsv'
logdir = 'expyfun-data-raw'
outdir = 'expyfun-data-clean'
outfile = 'clean-dataframe.tsv'
tabs = sorted(glob(op.join(logdir, '*.tab')))
logs = sorted(glob(op.join(logdir, '*.log')))

# load trial parameters
params = pd.read_csv(op.join(paramdir, paramfile), sep='\t')
# because pandas.read_csv(... dtype) argument doesn't work:
for col in ('subj', 'block', 'onset', 'offset', 'wav_idx', 'wav_nsamp'):
    params[col] = params[col].apply(int)
# convert string back to list
params['ttl_id'] = params['ttl_id'].apply(literal_eval)
# add a sequential "trial" integer to params
params['trial'] = params.groupby('subj')['subj'].transform(
    lambda x: np.arange(x.size))

df = None
header = ('subj', 'subj_code', 'trial', 'trial_id', 'start_time',
          'stop_time', 'ok_time')
for ix, tab in enumerate(tabs):
    table = []
    # parse experiment info from header
    with open(tab, 'r') as f:
        exp_dict = literal_eval(f.readline().strip('#').strip())
    subj_num = int(exp_dict['session']) - 1  # convert to zero-indexed
    subj_code = exp_dict['participant']
    # parse trial info
    list_of_dicts = read_tab(tab)
    for ixx, d in enumerate(list_of_dicts):
        row = [subj_num, subj_code, ixx, d['trial_id'][0][0], d['play'][0][1],
               d['stop'][0][1], d['trial_ok'][0][1]]
        extras = (d['keypress'], d['flip'], d['screen_text'])
        assert all([x == [] for x in extras])
        table.append(row)
    # convert to dataframe and parse trial_id into separate columns
    this_df = pd.DataFrame(table, columns=header)
    temp = this_df['trial_id'].apply(lambda x: x.split('_'))
    this_df['block'] = temp.apply(lambda x: int(x[0]))
    this_df['talker'] = temp.apply(lambda x: x[1])
    this_df['syll'] = temp.apply(lambda x: x[2])
    del temp
    # make sure we have the correct number of trials
    tester = params[params['subj'] == subj_num]
    assert tester.shape[0] == this_df.shape[0]
    del tester
    # append to master dataframe
    if df is None:
        df = this_df
    else:
        df = pd.concat(df, this_df)

df_all = params.merge(df, on=('subj', 'block', 'talker', 'syll', 'trial_id',
                              'trial'))
assert df_all.shape[0] == params.shape[0]  # true if data from all 12 subjs

# make sure timing makes sense: actual onset times are within 50 ms of expected
# onset times (will use actual in analysis, this just makes sure nothing went
# drastically wrong during experiment)
fs = get_tdt_rates()['25k']
assert all(df_all.groupby(['subj', 'block'])[['start_time', 'onset']].apply(
    lambda x: np.allclose(x['start_time'] - x['start_time'].min(),
                          x['onset'] / fs, atol=0.05, rtol=0)))

if not op.isdir(outdir):
    mkdir(outdir)
df_all.to_csv(op.join(outdir, outfile), sep='\t')
