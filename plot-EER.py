#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-EER.py'
===============================================================================

This script plots equal error rate as a grouped barplot.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)


from __future__ import division, print_function
import yaml
import numpy as np
import os.path as op
from pandas import read_csv
from expyfun.analyze import barplot
import matplotlib.pyplot as plt
plt.ioff()

# flags
savefig = True

# style setup
labelsize = 8
plt.rc('font', family='sans-serif', size=10, **{'sans-serif': 'M+ 1c'})
plt.rc('axes.spines', top=False, right=False, left=True, bottom=True)
plt.rc('ytick', right=False)
plt.rc('xtick', top=False)
colors = ['#77AADD', '#88CCAA', '#DDDD77', '#DDAA77', '#DD7788']

# file I/O
figdir = 'figures'
paramdir = 'params'
outdir = 'processed-data'
analysis_params = 'current-analysis-settings.yaml'

# load analysis params
with open(op.join(paramdir, analysis_params), 'r') as paramfile:
    params = yaml.load(paramfile)
clf_type = params['clf_type']
use_dss = params['dss']['use']
n_dss_channels_to_use = params['dss']['use_n_channels']
process_individual_subjs = params['process_individual_subjs']
fname_suffix = '-dss-{}'.format(n_dss_channels_to_use) if use_dss else ''
fname_id = '{}{}'.format(clf_type, fname_suffix)

# load data
subj_dict = np.load(op.join(paramdir, 'subjects.npz'))
langs = np.load(op.join(paramdir, 'langs.npy'))
eer_fname = 'equal-error-rates-{}.tsv'.format(fname_id)
eer = read_csv(op.join(outdir, eer_fname), sep='\t', index_col=0)
lang_names = dict(hin='Hindi', swh='Swahili', hun='Hungarian', nld='Dutch',
                  eng='English')
# sort so English is last
langs.sort()
langs = langs[::-1]
eer = eer[langs]
color_dict = {lang: col for lang, col in zip(langs, colors)}
# axis labels
bars = eer.values.ravel()
groups = np.arange(bars.size).reshape(-1, eer.columns.size)
group_names = dict(consonantal='conson.', sonorant='sonor.',
                   continuant='contin.',
                   delayedRelease='del. rel.', approximant='approx.',
                   nasal='nasal', lateral='lateral', labial='labial',
                   labiodental='labiodent.', coronal='coronal',
                   anterior='anterior', distributed='distrib.',
                   strident='strid.', dorsal='dorsal',
                   spreadGlottis='aspir.', periodicGlottalSource='voiced')
gn = [group_names[x] for x in eer.index]

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax, bar = barplot(bars, groups=groups, ax=ax, group_names=gn, bar_names=' ',
                  ylim=(0, 0.55), gap_size=0.6,
                  bar_kwargs=dict(color=[color_dict[l] for l in eer.columns]))
# make group names vertical
txt = [x for x in ax.get_children() if isinstance(x, plt.Text) and
       x.get_text() in group_names.values()]
for t in txt:
    t.set_rotation(90)
ax.axhline(0.5, color='k', linestyle=(0, (5, 7)), linewidth=1)
ax.set_ylabel('Equal error rate')
ax.legend(bar.get_children()[:5], (eer.columns), loc='lower right')
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # make room for vertical labels
if savefig:
    plt.savefig(op.join(figdir, 'eer-barplot-{}.pdf'.format(fname_id)))
else:
    plt.ion()
    plt.show()

# plot individ. subjs
if process_individual_subjs:
    fig, axs = plt.subplots(len(langs), len(subj_dict.keys()),
                            figsize=(48, 30))
    for s_ix, subj_id in enumerate(subj_dict.keys()):
        # load data
        fname = 'equal-error-rates-{}.tsv'.format(subj_id)
        eer = read_csv(op.join(outdir, subj_id, fname), sep='\t', index_col=0)
        for l_ix, lang in enumerate(langs):
            ax = axs[l_ix, s_ix]
            if lang in eer.columns:
                # axis labels
                bars = eer[lang].values
                gn = [group_names[x] for x in eer.index]
                ax, bar = barplot(bars, ax=ax, bar_names=gn,
                                  ylim=(0, 0.55), gap_size=0.3,
                                  bar_kwargs=dict(color=color_dict[lang]))
                ax.axhline(0.5, color='k', linestyle=(0, (5, 7)), linewidth=1)
                ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
            else:
                ax.xaxis.set_visible(False)
                for sp in ax.spines.keys():
                    ax.spines[sp].set_visible(False)
                ax.yaxis.set_tick_params(which='both', left=False,
                                         labelleft=False)
            if not s_ix:
                ax.set_ylabel(lang_names[lang], size=labelsize * 3)
            if not l_ix:
                ax.set_title(subj_id, size=labelsize * 3)
    plt.tight_layout()
    fig.subplots_adjust(top=0.94)
    fig.suptitle('Equal Error Rate (y) for each linguistic feature classifier '
                 '(x), grouped by language (rows) and by listener (columns)',
                 size=labelsize * 5)
if savefig:
    plt.savefig(op.join(figdir, 'eer-barplot-by-subj-{}.pdf'.format(fname_id)))
else:
    plt.ion()
    plt.show()
