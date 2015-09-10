# -*- coding: utf-8 -*-
"""
===============================================================================
Script ''
===============================================================================

This script reads a CSV file and presents words from it on screen.
"""
# @author: Daniel McCloy (drmccloy@uw.edu)
# Created on Thu Sep 10 10:00:50 2015
# License: BSD (3-clause)

from pandas import read_csv
from expyfun import ExperimentController
from os import path as op

testing = True

fname = 'dutch.csv'
lang = op.splitext(fname)[0]
df = read_csv(fname, na_values=[])
df.fillna('', inplace=True)

# change this to show only a subset of words, e.g., for "do overs"
idx_to_show = range(df.shape[0])
#idx_to_show = [18, 5, 15]

# ExperimentController startup parameters
kwargs = dict(screen_num=0, participant='foo', session='001',
              full_screen=True)
if testing:
    kwargs['full_screen'] = False
    kwargs['window_size'] = [800, 600]

# screen position params
top = [0, 0.4]
mid = [0, -0.2]
bot = [0, -0.4]

with ExperimentController('wordlistPrompter', **kwargs) as ec:
    ec.wait_secs(1.)
    ec.screen_prompt('press space to begin', min_wait=0.2, live_keys=['space'],
                     pos=top, wrap=False, font_size=48)

    rows = list(df.iterrows())
    for idx in idx_to_show:
        _idx, (phone, syll, word, gloss) = rows[idx]
        phone = phone.decode('utf-8')
        # skip consonants that we don't have example words for
        if word == '':
            continue
        # prep screen text
        _b = word.find(syll)
        _e = _b + len(syll)
        _g = '{color (102, 255, 153, 255)}'  # color for target syllable
        _h = '{color (255, 255, 255, 255)}'  # color for rest of word
        _w = '{0}{2}{1}{3}{0}{4}'.format(_h, _g, word[:_b], word[_b:_e],
                                         word[_e:])
        _syl = ec.screen_text(syll, pos=top, font_size=60, wrap=False,
                              color=(0.4, 1.0, 0.6, 1.0))
        _wrd = ec.screen_text(_w, pos=mid, font_size=32, wrap=False)
        _gls = ec.screen_text('"{}"'.format(gloss), pos=bot, font_size=24,
                              wrap=False, color='gray')
        # main trial cycle
        ec.identify_trial(ec_id=_idx, ttl_id=[0, 0])
        ec.write_data_line(u'{} [{}]: {}'.format(lang, phone, word), _idx)
        ec.start_stimulus()
        ec.wait_one_press(min_wait=0.2, live_keys=['space'])
        ec.stop()
        ec.trial_ok()
