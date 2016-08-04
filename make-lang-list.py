#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'make-lang-list.py'
===============================================================================

Simple helper to get the list of language abbreviations.
"""
# @author: drmccloy
# Created on Thu Aug  4 13:58:49 2016
# License: BSD (3-clause)

from numpy import save
from glob import glob
import os.path as op

# file i/o
paramdir = 'params'
outdir = 'processed-data'

# load each language's classification results
files = glob(op.join(outdir, 'classifier-probabilities-*.tsv'))
langs = [op.split(x)[1][-7:-4] for x in files]
save(op.join(paramdir, 'langs.npy'), langs)
