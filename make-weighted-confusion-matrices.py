#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-weighted-confusion-matrices.py'
===============================================================================

This script combines a feature-based confusion matrix with weights from
EEG-trained classifiers.
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)

from __future__ import division, print_function
import json
import numpy as np
from os import path as op
from pandas import read_csv

# file I/O
paramdir = 'params'
outdir = 'processed-data'

# load list of languages
foreign_langs = np.load(op.join(paramdir, 'foreign-langs.npy'))

# load ASCII to IPA dictionary
with open(op.join(paramdir, 'ascii-to-ipa.json'), 'r') as ipafile:
    ipa = json.load(ipafile)

