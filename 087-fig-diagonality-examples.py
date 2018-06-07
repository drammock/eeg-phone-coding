#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


plt.style.use(['dark_background'])

# I/O
paramdir = 'params'

# figure params
figure_paramfile = 'jobtalk-figure-params.yaml'
with open(op.join(paramdir, figure_paramfile), 'r') as f:
    figure_params = yaml.load(f)
    col = figure_params['yel']
    bad_color = figure_params['baddatacolor']
    good_color = figure_params['gooddatacolor']
    colordict = figure_params['colordict']
    axislabelsize = figure_params['axislabelsize']
    ticklabelsize = figure_params['ticklabelsize']
    ticklabelcolor = figure_params['ticklabelcolor']

# synth data
diag_data = np.eye(23) / 23 + 1e-5
unif_data = np.full((23, 23), 23 ** -2 + 1e-5)
off_data = np.full((23, 23), 1e-5)
off_data[0, -1] = 23 / 2
off_data[-1, 0] = 23 / 2

# init figure
fig, axs = plt.subplots(3, 1, figsize=(5, 6))
fig.subplots_adjust(left=0.15, bottom=0.05, top=0.95, right=0.95)

# plot
for ax, data, ann in zip(axs, [diag_data, unif_data, off_data],
                         ['1', '0', '-1']):
    ax.imshow(data, norm=LogNorm(vmin=1e-5, vmax=1), cmap='viridis')
    ax.axis('off')
    ax.text(5, 18, ann, size=16, ha='center')


fig.savefig(op.join('figures', 'jobtalk', 'fig-diagonality-examples.pdf'))
