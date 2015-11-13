# -*- coding: utf-8 -*-
"""
===============================================================================
Script ''
===============================================================================

This script does XXX.
"""
# @author: drmccloy
# Created on Thu Oct 22 15:14:50 2015
# License: BSD (3-clause)

import os
import os.path as op
from expyfun.stimuli import rms, read_wav, write_wav

asciify = {u'ʃa': 'esha',
           u'n̪a': 'na',
           u'ɡa': 'ga',
           u'l̪a': 'la',
           u'r̪a': 'ra',
           u'd̪a': 'da',
           u't̪a': 'ta',
           u'ʀa': 'trilluvulara',
           u'x̟a': 'frontedxa',
           u'χa': 'xuvulara',
           u'ʂa': 'sra',
           u'ʋa': 'vua',
           u'ɲa': 'nja',
           u'ʁa': 'Rinva',
           u'ça': 'ccedilla',
           u'pʰa': 'pha',
           u'tʰa': 'tha',
           u'kʰa': 'kha',
           u'ʒa': 'ezha',
           u'ɟa': 'jstopa',
           u'ʈʂa': 'tsra',
           u'tsʰa': 'tsha',
           u'ʈʂʰa': 'tsrha',
           u't̠ʃa': 'tesha',
           u'ɕa': 'ccurla',
           u'd̠ʒa': 'dezha',
           u'tɕa': 'tccurla',
           u'tɕʰa': 'tccurlha',
           }

rms_out = 0.01
indir = 'stimuli'
outdir = 'stimuli-rms'
subdirs = os.walk(indir).next()[1]

for subdir in subdirs:
    inpath = op.join(indir, subdir)
    outpath = op.join(outdir, subdir)
    files = os.listdir(inpath)
    wavfiles = [f for f in files if f[-4:] == '.wav']
    for wavfile in wavfiles:
        syll = wavfile[:-4].decode('utf-8')
        wav, fs = read_wav(op.join(inpath, wavfile))
        rms_in = rms(wav)
        wavout = wav * rms_out / rms_in
        if not op.exists(outpath):
            os.makedirs(outpath)
        fname = asciify[syll] if syll in asciify.keys() else syll
        write_wav(op.join(outpath, fname + '.wav'), wavout, fs, overwrite=True)
