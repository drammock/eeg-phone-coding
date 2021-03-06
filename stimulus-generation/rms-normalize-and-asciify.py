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

asciify = {u'ʃa': 'esh',
           u'ɟa': 'j-bar',
           u'n̪a': 'n-dental',
           u'ɡa': 'g',
           u'l̪a': 'l-dental',
           u'r̪a': 'r-dental',
           u'd̪a': 'd-dental',
           u't̪a': 't-dental',
           u'ʀa': 'uvular-trill',
           u'x̟a': 'fronted-x',
           u'ɣa': 'gamma',
           u'χa': 'chi',
           u'ʂa': 's-retroflex',
           u'ʋa': 'nu',
           u'ɲa': 'n-palatal',
           u'ʁa': 'r-cap-inverted',
           u'ça': 'c-cedilla',
           u'ɕa': 'c-curl',
           u'ʒa': 'ezh',
           u'ŋa': 'engma',
           u'ᶮɟa': 'j-bar-prenasalized',
           u'ɓa': 'b-implosive',
           u'ɗa': 'd-implosive',
           u'ða': 'eth',
           u'θa': 'theta',
           u't̠ʃa': 'tesh',
           u'tʃa': 'tesh',
           u'ʈʂa': 'ts-retroflex',
           u'ᵐba': 'b-prenasalized',
           u'ⁿda': 'd-prenasalized',
           u'ᵑɡa': 'g-prenasalized',
           u'ⁿza': 'z-prenasalized',
           u'ᶬva': 'v-prenasalized',
           u'd̠ʒa': 'dezh',
           u'tɕa': 'tc-curl',
           u'pʰa': 'p-aspirated',
           u'tʰa': 't-aspirated',
           u'cʰa': 'c-aspirated',
           u'kʰa': 'k-aspirated',
           u't̠ʃʰa': 'tesh-aspirated',
           u'tʃʰa': 'tesh-aspirated',
           u'ʈʂʰa': 'ts-retroflex-aspirated',
           u'tsʰa': 'ts-aspirated',
           u'tɕʰa': 'tc-curl-aspirated',
           u'ɹa': 'r-turned',
           u'ɽa': 'flap-retroflex',
           u'ʈa': 't-retroflex',
           u'ɽʱa': 'flap-retroflex-breathy',
           u'ɡʱa': 'g-breathy',
           u'bʱa': 'b-breathy',
           u'd̪ʱa': 'd-dental-breathy',
           u'ɖa': 'd-retroflex',
           u'ɖʱa': 'd-retroflex-breathy',
           u'd̠ʒʱa': 'dezh-breathy',
           u'ʈʰa': 't-retroflex-aspirated',
           u't̪ʰa': 't-dental-aspirated',
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
        syll = wavfile[:-6].decode('utf-8')
        # retain token indices only for english syllables:
        syll_ix = wavfile[-6:-4] if subdir[:3] in ['eng'] else ''
        wav, fs = read_wav(op.join(inpath, wavfile))
        rms_in = rms(wav)
        wavout = wav * rms_out / rms_in
        if not op.exists(outpath):
            os.makedirs(outpath)
        fname = asciify[syll] if syll in asciify.keys() else syll[:-1]
        write_wav(op.join(outpath, fname + syll_ix + '.wav'), wavout, fs,
                  overwrite=True)
