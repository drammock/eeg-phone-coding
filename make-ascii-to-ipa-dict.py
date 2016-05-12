#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
===============================================================================
Script 'create-ascii-to-ipa-dict.py'
===============================================================================

This script creates a mapping between ASCII representations of phones and the
corresponding representations in IPA, and save it as a JSON file. The phone set
is project-specific but could be generalized...
"""
# @author: drmccloy
# Created on Wed Apr  6 12:43:04 2016
# License: BSD (3-clause)

import json
import numpy as np
import os.path as op
from pandas import read_csv

# file I/O
indir = 'params'
outdir = 'params'

# load data frame to get list of consonants in ASCII
df = read_csv(op.join(indir, 'master-dataframe.tsv'), sep='\t',
              usecols=['subj', 'syll'], dtype=dict(subj=int, syll=str))
df['cons'] = df['syll'].apply(lambda x: x[:-2].replace('-', '_')
                              if x.split('-')[-1] in ('0', '1', '2')
                              else x.replace('-', '_'))

# convert ASCII to IPA for feature lookup
ipa = dict(t_dental=u't̪', gamma=u'ɣ', t_dental_aspirated=u't̪ʰ',
           g_breathy=u'ɡʱ', t_retroflex_aspirated=u'ʈʰ',
           tesh_aspirated=u't̠ʃʰ', r_cap_inverted=u'ʁ', d_implosive=u'ɗ',
           tc_curl=u'tɕ', d_retroflex_breathy=u'ɖʱ', j_bar=u'ɟ',
           engma=u'ŋ', fronted_x=u'x̟', r_turned=u'ɹ', theta=u'θ',
           d_dental_breathy=u'd̪ʱ', ts_aspirated=u'tsʰ',
           b_prenasalized=u'ᵐb', ezh=u'ʒ', b_implosive=u'ɓ',
           s_retroflex=u'ʂ', b_breathy=u'bʱ', flap_retroflex_breathy=u'ɽʱ',
           z_prenasalized=u'ⁿz', g=u'ɡ', nu=u'ʋ', r_dental=u'r̪',
           p_aspirated=u'pʰ', g_prenasalized=u'ᵑɡ', c_cedilla=u'ç',
           dezh_breathy=u'd̠ʒʱ', ts_retroflex_aspirated=u'ʈʂʰ',
           t_aspirated=u'tʰ', l_dental=u'l̪', c_aspirated=u'cʰ', esh=u'ʃ',
           ts_retroflex=u'ʈʂ', k_aspirated=u'kʰ', eth=u'ð',
           v_prenasalized=u'ᶬv', n_palatal=u'ɲ', tesh=u't̠ʃ',
           n_dental=u'n̪', dezh=u'd̠ʒ', c_curl=u'ɕ',
           j_bar_prenasalized=u'ᶮɟ', d_retroflex=u'ɖ', uvular_trill=u'ʀ',
           tc_curl_aspirated=u'tɕʰ', d_dental=u'd̪', chi=u'χ',
           d_prenasalized=u'ⁿd', flap_retroflex=u'ɽ', t_retroflex=u'ʈ')
# fix some segment mismatches between dict and feature table
corrections = dict(fronted_x=u'x', v_prenasalized=u'ɱv', b_prenasalized=u'mb',
                   d_prenasalized=u'nd', j_bar_prenasalized=u'ɲɟ',
                   z_prenasalized=u'nz', g_prenasalized=u'ŋɡ', c_cedilla=u'ç',
                   d_retroflex_breathy=u'ɖ̤', d_dental_breathy=u'd̪̤',
                   g_breathy=u'ɡ̤', flap_retroflex_breathy=u'ɽ̤',
                   dezh_breathy=u'd̠ʒ̤')
ipa.update(corrections)
# add in consonants that are within ASCII
all_cons = np.unique(df['cons'].apply(str.replace, args=('-', '_'))
                     ).astype(unicode)
ascii_cons = all_cons[np.in1d(all_cons, ipa.keys(), invert=True)]
ipa.update({k: k for k in ascii_cons})
# save
with open(op.join(outdir, 'ascii-to-ipa.json'), 'w') as out:
    out.write(json.dumps(ipa))
