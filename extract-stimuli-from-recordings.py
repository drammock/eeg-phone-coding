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
outdir = 'stimuli'
subdirs = os.walk(indir).next()[1]

for subdir in subdirs:
    sd = op.join(indir, subdir)
    files = os.listdir(sd)
    textgrids = [f for f in files if f[-9:] == '.TextGrid']
    for tg in textgrids:
        wavfile = op.splitext(tg)[0] + '.wav'
        outpath = op.join(outdir, subdir)
        if not op.exists(outpath):
            os.makedirs(outpath)
        call(['praat', 'extract-labeled-intervals.praat', op.join(sd, wavfile),
              op.join(sd, tg), outpath, '1'])
