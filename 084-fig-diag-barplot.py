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

# I/O
paramdir = 'params'
if individual:
    fname = 'individually-row-ordered-matrix-diagonality-nonan-eer.tsv'
else:
    fname = ('cross-featsys-cross-subj-row-ordered-matrix-diagonality-'
             'nonan-eer.tsv')
fpath = op.join('processed-data-logistic', 'matrix-correlations', fname)

# figure params
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
qrtp = dict(color='none', facecolor=bad_color)                # quartile box
whsp = dict(linewidth=0)                                      # whisker
medp = dict(color=bgcolor, linewidth=2)                       # median line
sigp = dict(color=col, linewidth=2)                           # signif. bracket
ptsp = dict(size=6, color=datacolor, linewidth=0)             # data pts
boxp = dict(showcaps=False, showfliers=False, boxprops=qrtp, medianprops=medp,
            width=0.4, whiskerprops=whsp)

# init figure
fig, ax = plt.subplots(figsize=(5, 6))
fig.subplots_adjust(left=0.15, bottom=0.05, top=0.95, right=0.95)

# plot
ax = sns.boxplot(ax=ax, data=data, **boxp)
ax = sns.swarmplot(ax=ax, dodge=True, data=data, **ptsp)

# plot average
ptsp.update(marker='*', color=avg)
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
ax.set_ylabel('Matrix diagonality', size=axislabelsize, labelpad=12)
ylims = (0.42, 0.88) if individual else (0.22, 0.76)
yticks = np.linspace(0.5, 0.8, 4) if individual else np.linspace(0.3, 0.7, 5)
ax.set_ylim(*ylims)
ax.set_yticks(yticks)
ax.tick_params(axis='y', labelsize=ticklabelsize + 2)
ax.tick_params(axis='x', length=0, labelsize=axislabelsize)

individ = '-individ' if individual else ''
fname = 'fig-diagonality-barplot{}.pdf'.format(individ)
fig.savefig(op.join(outdir, fname))
