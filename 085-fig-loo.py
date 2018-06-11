#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os.path as op
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from aux_functions import matrix_row_column_correlation, optimal_leaf_ordering


def joint_prob(panel):
    logged = -1 * panel.apply(np.log)
    summed = (-1 * logged.sum(axis=0)).swapaxes(0, 1)
    exped = summed.apply(np.exp)
    return exped


def do_olo(confmat):
    olo = optimal_leaf_ordering(confmat)
    dendrograms, linkages = olo['dendrograms'], olo['linkages']
    row_ord = dendrograms['row']['leaves']
    ordered_confmat = this_confmat.iloc[row_ord, row_ord]
    return ordered_confmat


# FLAGS
target = 'manuscript'  # presentation or manuscript
swarm = False

if target == 'presentation':
    outdir = op.join('figures', 'jobtalk')
    plt.style.use('dark_background')
else:
    outdir = op.join('figures', 'manuscript')

# LOAD PARAMS FROM YAML
paramdir = 'params'
analysis_param_file = 'current-analysis-settings.yaml'
with open(op.join(paramdir, analysis_param_file), 'r') as f:
    analysis_params = yaml.load(f)
    subjects = analysis_params['subjects']
    feature_systems = analysis_params['feature_systems']
    feature_mappings = analysis_params['feature_mappings']

for fsys in ('jfh_dense', 'spe_dense', 'phoible_sparse'):
    del feature_systems[fsys]
feature_abbrevs = dict(phoible_redux='PHOIBLE', jfh_sparse='PSA',
                       spe_sparse='SPE')

plt.style.use([op.join(paramdir, 'matplotlib-cb-colors.yaml'),
               op.join(paramdir, 'matplotlib-font-myriad.yaml')])

# I/O
datadir = 'processed-data-logistic'
indir = op.join(datadir, 'single-feat-confmats')

# plot params
qrtp = dict(color='none', facecolor='0.8')                    # quartile box
whsp = dict(linewidth=0)                                      # whisker
medp = dict(color='w', linewidth=2)                           # median line
ptsp = dict(size=4, color='k', linewidth=0)                   # data pts
boxp = dict(showcaps=False, showfliers=False, boxprops=qrtp, medianprops=medp,
            width=0.4, whiskerprops=whsp)

# init containers
feat_confmats = dict()
full_confmats = dict()
feat_diags = dict()
full_diags = dict()

for featsys in feature_systems:
    # set row/column order per the average matrices in the main manuscript
    fname = (f'row-ordered-eer-confusion-matrix-nonan-eng-cvalign-dss5-'
             f'{featsys}-average.tsv')
    fpath = op.join(datadir, 'ordered-confusion-matrices', fname)
    avg_confmat = pd.read_csv(fpath, sep='\t', index_col=0)
    phone_order = avg_confmat.index
    # this featsys info
    features = feature_systems[featsys]
    feat_map = feature_mappings[featsys]
    # containers
    feat_confmats[featsys] = dict()
    full_confmats[featsys] = dict()
    feat_diags[featsys] = dict()
    full_diags[featsys] = dict()
    # loop over subjects
    for subj in subjects:
        feat_confmats[featsys][subj] = dict()
        feat_diags[featsys][subj] = dict()
        for feat in features:
            # load single-feat confmats
            fname = ('eer-confusion-matrix-nonan-eng-cvalign-dss5-'
                     '{}-{}.tsv'.format(feat, subj))
            confmat = pd.read_csv(op.join(indir, fname), sep='\t', index_col=0)
            confmat = confmat.astype(float)
            feat_confmats[featsys][subj][feat] = confmat
        # full confmat
        confmat_3d = pd.Panel.from_dict(feat_confmats[featsys][subj])
        confmat_3d = confmat_3d.fillna(0.5)
        # collapse across features
        this_confmat = joint_prob(confmat_3d)
        # compute the optimal ordering
        ordered_confmat = do_olo(this_confmat)
        '''
        # put in consistent order
        ordered_confmat = this_confmat.loc[phone_order, phone_order]
        '''
        full_confmats[featsys][subj] = ordered_confmat
        # compute diagonality
        full_diags[featsys][subj] = \
            matrix_row_column_correlation(ordered_confmat)
        # leave-one-out
        for feat in features:
            this_confmat = confmat_3d.copy()
            this_confmat.drop(feat, inplace=True)
            this_confmat = joint_prob(this_confmat)
            # compute the optimal ordering
            ordered_confmat = do_olo(this_confmat)
            '''
            # put in consistent order
            ordered_confmat = this_confmat.loc[phone_order, phone_order]
            '''
            feat_confmats[featsys][subj][feat] = ordered_confmat
            # compute diagonality
            feat_diags[featsys][subj][feat] = \
                matrix_row_column_correlation(ordered_confmat)
        # reduce
        feat_confmats[featsys][subj] = \
            pd.Panel.from_dict(feat_confmats[featsys][subj])
    feat_diags[featsys] = pd.DataFrame.from_dict(feat_diags[featsys])
    full_diags[featsys] = pd.Series(full_diags[featsys], name='none')
    # rename
    feat_diags[featsys].rename(index=feat_map, inplace=True)
    # sort
    feat_order = (feat_diags[featsys].mean(axis=1)
                  .sort_values(ascending=False).index)
    subj_order = full_diags[featsys].sort_values(ascending=False).index
    feat_diags[featsys] = feat_diags[featsys].loc[feat_order, subj_order]
    full_diags[featsys] = full_diags[featsys][subj_order]
    # concatenate
    data = pd.concat((pd.DataFrame(full_diags[featsys]).T,
                      feat_diags[featsys]))

    # init figure
    fig, ax = plt.subplots(figsize=(6.75, 3))
    # plot
    ax = sns.boxplot(ax=ax, data=data.T, **boxp)
    if swarm:
        right_margin = 0.98
        ax = sns.swarmplot(ax=ax, data=data.T, dodge=True, **ptsp)
    else:  # connected lines
        right_margin = 0.87
        ax = data.plot(ax=ax, style='-', legend=False, linewidth=0.25,
                       zorder=20)
        xvals = [data.index.get_loc(x) for x in data.idxmin()]
        yvals = data.min()
        colors = [l.get_color() for l in ax.lines][-len(subjects):]
        handles = []
        for x, y, sid, c in zip(xvals, yvals, yvals.index, colors):
            ax.plot(x, y, '.', color=c, markersize=6, zorder=10)
            # fake data for legend
            handles.append(Line2D([], [], color=c, marker='.', markersize=8,
                                  linewidth=1.5, linestyle='-', label=sid))
        labels = [h.get_label() for h in handles]
        legend = ax.legend(handles, labels, title='Listener', loc='upper left',
                           bbox_to_anchor=(1.02, 1), borderaxespad=0,
                           edgecolor='none')
    # garnish
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('matrix diagonality', labelpad=7)
    # savefig
    fig.subplots_adjust(bottom=0.28, top=0.98, left=0.09, right=right_margin)
    fname = f'fig-loo-diag-{feature_abbrevs[featsys]}.pdf'
    fig.savefig(op.join(outdir, fname))
