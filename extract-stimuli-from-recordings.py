# -*- coding: utf-8 -*-
"""
===============================================================================
Script ''
===============================================================================

This script does XXX.
"""
# @author: drmccloy
# Created on Wed Oct 21 18:01:30 2015
# License: BSD (3-clause)

import os
import os.path as op
from subprocess import call

indir = 'recordings-highpassed'
tgdir = 'textgrids'
outdir = 'stimuli'
subdirs = os.walk(indir).next()[1]  # list of only *immediate* subdirs of indir

for subdir in subdirs:
    inpath = op.join(indir, subdir)
    tgpath = op.join(tgdir, subdir)
    outpath = op.join(outdir, subdir)
    files = os.listdir(tgpath)
    textgrids = [f for f in files if f[-9:] == '.TextGrid']
    for tg in textgrids:
        wavfile = op.splitext(tg)[0] + '.wav'
        if not op.exists(outpath):
            os.makedirs(outpath)
        call(['praat', '--run', 'extract-labeled-intervals.praat',
              op.join(inpath, wavfile), op.join(tgpath, tg), outpath, '1'])
