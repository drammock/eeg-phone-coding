#!/usr/bin/env python3

import yaml
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from aux_functions import matrix_row_column_correlation


def plot_matrix(mat, ax, **kwargs):
    ax.imshow(mat, **kwargs)
    ax.tick_params(axis='both', which='both', length=0,
                   labelbottom=False, labelleft=False,
                   grid_color='k', grid_alpha=0.25, grid_linewidth=0.5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True)


# setup
randseed = np.random.RandomState(seed=236887691)  # the 13 millionth prime
nrows = 4
ixs = np.atleast_2d(np.arange(1, 1 + nrows))

# example matrices
rand_mat = 0.2 + 0.6 * randseed.rand(nrows, nrows)  # between 0.2 & 0.8
i_mat = np.identity(nrows)
diagmat = np.diag(0.1 + 0.9 * randseed.rand(nrows))
tridiag = i_mat + 0.5 * (np.eye(nrows, k=1) + np.eye(nrows, k=-1))
# bad_mat = np.eye(nrows, k=(1 - nrows)) + np.eye(nrows, k=(nrows - 1))
antidiag = np.fliplr(np.diag(0.3 + 0.7 * randseed.rand(nrows)))
uniform = np.full_like(rand_mat, 0.5)
matrices = dict(identity=i_mat, tridiagonal=tridiag, antidiagonal=antidiag,
                random=rand_mat, diagonal=diagmat, uniform=uniform)
mat_norm = Normalize(vmin=0, vmax=1)

# weight matrices
uniform_weights = np.full((nrows, nrows), 1)
r_c_weights = ixs.T @ ixs
row_weights = np.tile(ixs.T, (1, nrows))
col_weights = np.tile(ixs, (nrows, 1))
weights = dict(rowcol=r_c_weights, row=row_weights, col=col_weights,
               rowsqu=(row_weights ** 2), colsqu=(col_weights ** 2),
               uniform=uniform_weights)
weight_norm = LogNorm(vmin=1, vmax=r_c_weights.max())

# figure setup
ticks = np.arange(nrows) - 0.5
height_frac = 0.9
width_frac = 0.8
width_buff = 0.3
left_margin = (1 - width_frac) / 3
subplot_width = width_frac  # (width_frac - width_buff) / ncol
common_kwargs = dict(family='serif', va='center')

# init containers
diag_vals = dict()

for matrix_name, matrix in matrices.items():
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_axes((0, 0, 1, 1))
    plot_matrix(matrix, ax, norm=mat_norm, cmap='gray_r')
    fig.savefig(op.join('figures', 'supplement',
                        f'{matrix_name}.pdf'))
    # explicit cast to float needed for clean write to yaml:
    diag_vals[matrix_name] = float(matrix_row_column_correlation(matrix))

for matrix_name, matrix in weights.items():
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_axes((0, 0, 1, 1))
    plot_matrix(matrix, ax, norm=weight_norm, cmap='gray_r')
    fig.savefig(op.join('figures', 'supplement',
                        f'weight-{matrix_name}.pdf'))

# save numbers
fname = op.join('figures', 'supplement', 'sample-diag-values.yaml')
with open(fname, 'w') as f:
    yaml.dump(diag_vals, f, default_flow_style=False)

'''
from mpl_toolkits.mplot3d import Axes3D  # noqa
ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), projection='3d')
X, Y = np.meshgrid(ixs, ixs)
ax.plot_surface(X, Y, Z=r_c_weights)
'''
