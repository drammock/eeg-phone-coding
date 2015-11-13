# -*- coding: utf-8 -*-
"""
===============================================================================
Script ''
===============================================================================

This script does a highpass filter on a bunch of raw wordlist recordings and
puts them in a new directory tree sibling to the original recording tree.
"""
# @author: drmccloy
# Created on Tue Oct 13 14:32:11 2015
# License: BSD (3-clause)

import os
import os.path as op
import scipy.signal as ss
from expyfun.stimuli import read_wav, write_wav

lxmap = {'hungarian-male': 'hun-m', 'hungarian-female': 'hun-f',
         'chinese-male': 'cmn-m', 'chinese-female': 'cmn-f',
         'swahili-male': 'swh-m', 'swahili-female': 'swh-f',
         'english-male': 'eng-m', 'english-female': 'eng-f',
         'dutch-male': 'nld-m', 'dutch-female': 'nld-f',
         'hindi-male': 'hin-m', 'hindi-female': 'hin-f'}

cutoff = 50.
indir = 'recordings'
outdir = 'recordings-highpassed'
subdirs = os.walk(indir).next()[1]

for subdir in subdirs:
    sd = op.join(indir, subdir)
    wavfiles = os.listdir(sd)
    wavfiles[:] = [w for w in wavfiles if w[-4:] == '.wav']
    assert all([w[-4:] == '.wav' for w in wavfiles])
    for wavfile in wavfiles:
        wav, fs = read_wav(op.join(sd, wavfile))
        mono = wav[0] if wav.ndim > 1 else wav  # convert to mono
        b, a = ss.butter(4, cutoff / (fs/2.), btype='high')
        hp = ss.lfilter(b, a, mono)
        outsubdir = op.join(outdir, lxmap[subdir])
        if not op.exists(outsubdir):
            os.makedirs(outsubdir)
        write_wav(op.join(outsubdir, wavfile), hp, fs, overwrite=True)
