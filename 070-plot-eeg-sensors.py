#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'plot-eeg-sensors.py'
===============================================================================

This script plots a diagram of the analysis pipeline.
"""
# @author: drmccloy
# Created on Fri Oct 20 15:38:57 PDT 2017
# License: BSD (3-clause)

import os.path as op
import numpy as np
import mne
from mayavi import mlab

# basic file I/O
datadir = op.join('figures', 'eeg-head')

# make head with EEG sensors
mri_subj = 'AKCLEE_109'
trans = mne.read_trans(op.join(datadir, 'eric_sps_09-trans.fif'))
raw_fname = op.join(datadir, 'eric_sps_09_01_raw.fif')
raw = mne.io.read_raw_fif(raw_fname, preload=True, allow_maxshield=True)
# set up mayavi scene properties
bgcolor = (1, 1, 1)
mfig = mlab.figure(bgcolor=bgcolor, size=(1200, 1200))
mfig.scene.render_window.point_smoothing = True
mfig.scene.render_window.line_smoothing = True
mfig.scene.render_window.polygon_smoothing = True
mfig.scene.render_window.multi_samples = 8  # Try with 4 if too slow
# plot
mfig = mne.viz.plot_alignment(raw.info, trans=trans, subject=mri_subj,
                              dig=False, surfaces='seghead', meg=False,
                              eeg='projected', fig=mfig)

# customize plot
surface = mfig.children[0].children[0].children[0]
sensors = mfig.children[1].children[0].children[0]
surface.actor.property.color = (0.7, 0.7, 0.7)  # (0.91, 0.78, 0.69)
sensors.actor.property.color = (0.3, 0.4, 0.35)
sensors.actor.property.opacity = 1.
# camera
mfig.scene.camera.position = np.array([-0.13, 0.48, 0.2])
mfig.scene.camera.focal_point = np.array([0, 0, 0])
mfig.scene.camera.view_angle = 30
# lights
azimuths = [35, -35, 20, 0]
elevations = [30, -60, 5, 0]
intensities = [1, 0.5, 0.5, 0]
colors = [(0.9, 1, 1), (1, 1, 0.92), (1, 0.94, 1), (0, 0, 0)]
actives = [True, True, True, False]
for light, az, el, nt, co, ac in zip(mfig.scene.light_manager.lights, azimuths,
                                     elevations, intensities, colors, actives):
    light.activate = ac
    if ac:
        light.azimuth = az
        light.elevation = el
        light.intensity = nt
        light.color = co

# make background transparent
rgba_array = mlab.screenshot(figure=mfig, mode='rgba', antialiased=True)
bgcolor = bgcolor + (1,)
mask = np.all(rgba_array == bgcolor, axis=-1)
rgba_array[mask, -1] = 0
rgba_array = np.fliplr(rgba_array)  # put ear on left

# save figure
np.save(op.join(datadir, 'eeg-sensors.npy'), rgba_array, allow_pickle=False)
mlab.savefig(op.join(datadir, 'eeg-sensors.png'), figure=mfig)
mlab.close(mfig)
