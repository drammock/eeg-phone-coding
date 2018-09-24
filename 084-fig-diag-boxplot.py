#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os.path as op
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# FLAGS
target = 'manuscript'  # presentation or manuscript
individual = True      # individual or cross-subject?

if target == 'presentation':
    figure_paramfile = 'jobtalk-figure-params.yaml'
    outdir = op.join('figures', 'jobtalk')
    plt.style.use('dark_background')
else:
    figure_paramfile = 'manuscript-figure-params.yaml'
    outdir = op.join('figures', 'manuscript')

# figure params
paramdir = 'params'
with open(op.join(paramdir, figure_paramfile), 'r') as f:
    figure_params = yaml.load(f)
    col = figure_params['yel']
    avg = figure_params['grn']
    bad_color = figure_params['baddatacolor']
    good_color = figure_params['gooddatacolor']
    colordict = figure_params['colordict']
    axislabelsize = figure_params['axislabelsize']
    ticklabelsize = figure_params['ticklabelsize']
    ticklabelcolor = figure_params['ticklabelcolor']
    datacolor = figure_params['datacolor']
    bgcolor = figure_params['bgcolor']
    boxplot_color = figure_params['boxplot']
    median_color = figure_params['median']
    swarm_color = figure_params['swarm']
# analysis params
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    truncate = analysis_params['eeg']['truncate']
    trunc_dur = analysis_params['eeg']['trunc_dur']
del analysis_params

# FILE NAMING VARIABLES
trunc = f'-truncated-{int(trunc_dur * 1000)}' if truncate else ''

# STYLES
myriad = op.join(paramdir, 'matplotlib-font-myriad.yaml')

# I/O
if individual:
    fname = 'individually-row-ordered-matrix-diagonality-nonan-eer.tsv'
else:
    fname = ('cross-featsys-cross-subj-row-ordered-matrix-diagonality-'
             'nonan-eer.tsv')
fpath = op.join(f'processed-data-logistic{trunc}', 'matrix-correlations',
                fname)

# load data
diag_data = pd.read_csv(fpath, sep='\t', index_col=0)

# restrict feature systems that we plot (sparse only)
feature_systems = ['jfh_sparse', 'spe_sparse', 'phoible_redux']
feature_abbrevs = ['PSA', 'SPE', 'PHOIBLE']
data = diag_data[feature_systems].copy()
data.rename(columns={k: v for k, v in zip(feature_systems, feature_abbrevs)},
            inplace=True)

# remove "average" data to plot separately
avg_data = data.loc['average']
data.drop('average', axis='index', inplace=True)

# plot params
qrtp = dict(color='none', facecolor=boxplot_color)            # quartile box
whsp = dict(linewidth=0)                                      # whisker
medp = dict(color=median_color, linewidth=2)                  # median line
sigp = dict(color=col, linewidth=1.5)                         # signif. bracket
ptsp = dict(size=4.5, color=swarm_color, linewidth=0)         # data pts
boxp = dict(showcaps=False, showfliers=False, boxprops=qrtp, medianprops=medp,
            width=0.4, whiskerprops=whsp)

# init figure
plt.style.use(myriad)
fig, ax = plt.subplots(figsize=(3.25, 4))
fig.subplots_adjust(left=0.2, bottom=0.06, top=0.98, right=0.98)

# plot
ax = sns.boxplot(ax=ax, data=data, **boxp)
ax = sns.swarmplot(ax=ax, dodge=True, data=data, **ptsp)

# plot average
ax.plot(avg_data, linewidth=0, marker='_', color='k', markersize=24)

'''
# plot connecting lines
for row in data.itertuples(index=False):
    ax.plot(row, linewidth=0.5, linestyle='-', color='k', alpha=0.2)
'''

# stats
stats = pd.DataFrame(columns=['a', 'b', 't', 'p'])
for a, b in zip(data.columns, np.roll(data.columns, -1)):
    tval, pval = ttest_rel(data[a], data[b])
    index = '~'.join([a, b])
    stats = stats.append(pd.DataFrame(dict(a=a, b=b, t=tval, p=pval),
                         index=[index]))
bonferroni = 0.05 / stats.shape[0]
stats['signif'] = stats['p'] < bonferroni
stats['p_corr'] = stats['p'] * stats.shape[0]

# brackets
maxes = data.max()
pad = 0.015
depth = 0.01
brackets = list()
for _, a, b, s, p in stats[['a', 'b', 'signif', 'p_corr']].itertuples():
    if s:
        xa = data.columns.tolist().index(a)
        xb = data.columns.tolist().index(b)
        ya = maxes[a]
        yb = maxes[b]
        ym = max([ya, yb])
        yh = ym + pad + depth
        ax.text(x=((xa + xb) / 2), y=yh, s='*', color=col, size=14,
                ha='center', va='baseline')
        ax.plot([xa, xa, xb, xb], [ya+pad, yh, yh, yb+pad], **sigp)
        maxes[a] = max(maxes[a], yh)
        maxes[b] = max(maxes[b], yh)

# garnish
sns.despine(ax=ax)
ax.set_ylabel('Matrix diagonality', size=axislabelsize, labelpad=10)
ylims = (0.42, 0.88) if individual else (0.22, 0.76)
yticks = np.linspace(0.5, 0.8, 4) if individual else np.linspace(0.3, 0.7, 5)
ylims = np.array(ylims)
if trunc_dur == 0.1:
    ylims -= 0.2
    yticks -= 0.2
elif trunc_dur == 0.15:
    ylims -= np.array((0.9, 0.3))
    yticks = np.linspace(-0.4, 0.5, 10)
ax.set_ylim(*ylims)
ax.set_yticks(yticks)
ax.tick_params(axis='y', labelsize=ticklabelsize + 2)
ax.tick_params(axis='x', length=0, labelsize=axislabelsize)

individ = '-individ' if individual else ''
fname = f'fig-diagonality-barplot{individ}{trunc}.pdf'
fig.savefig(op.join(outdir, fname))
