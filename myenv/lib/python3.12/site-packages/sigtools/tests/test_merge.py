#!/usr/bin/env python
# sigtools - Collection of Python modules for manipulating function signatures
# Copyright (C) 2013-2022 Yann Kaiser
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import repeated_test

from sigtools.signatures import merge, IncompatibleSignatures
from sigtools.support import s
from sigtools.tests.util import Fixtures, FixturesWithFutureAnnotations
from sigtools._util import OrderedDict as Od


class MergeTests(FixturesWithFutureAnnotations):
    def _test(self, result, exp_sources, *signatures, support_s, downgrade_sig):
        assert len(signatures) >= 2
        sigs = [support_s(sig, name='_' + str(i))
                for i, sig in enumerate(signatures, 1)]

        with self.maybe_with_downgrade_and_ignore_warnings(
            downgrade_sig,
            self.downgrade_sigs,
        ) as maybe_downgrade:
            sig = merge(*maybe_downgrade(sigs))
        exp_sig = support_s(result)
        exp_sources['+depths'] = dict(
            ('_' + str(i + 1), 0) for i in range(len(signatures)))

        self.assertSigsEqual(sig, exp_sig)
        if not downgrade_sig:
            self.assertSourcesEqual(sig.sources, exp_sources)

    posarg_default_erase = '', {}, '', '<a>=1'
    posarg_stars = '<a>', {2: 'a'}, '*args', '<a>'

    posarg_convert = '<a>', {1: 'a'}, '<a>', 'b'
    posarg_convert_left = '<b>', {2: 'b'}, 'a', '<b>'

    pokarg_default_erase = '', {}, '', 'a=1'

    pokarg_star_convert_pos = '<a>', {2: 'a'}, '*args', 'a'
    pokarg_star_convert_kwo = '*, a', {2: 'a'}, '**kwargs', 'a'
    pokarg_star_keep = 'a', {2: 'a'}, '*args, **kwargs', 'a'

    pokarg_rename = '<a>', {1: 'a'}, 'a', 'b'
    pokarg_rename_second = '<a>, <b>', {1: 'ab', 2: 'a'}, 'a, b', 'a, c'

    pokarg_found_kwo = '*, a', Od([(1, 'a'), (2, 'a')]), '*, a', 'a'
    pokarg_found_kwo_r = '*, a', Od([(2, 'a'), (1, 'a')]), 'a', '*, a'

    kwarg_default_erase = '', {}, '', '*, a=1'
    kwarg_stars = '*, a=1', {2: 'a'}, '**kwargs', '*, a=1'

    kwoarg_same = '*, a', {1: 'a', 2: 'a'}, '*, a', '*, a'
    posarg_same = '<a>', {1: 'a', 2: 'a'}, '<a>', '<a>'
    posarg_name = '<a>', {1: 'a'}, '<a>', '<b>'
    pokarg_same = 'a', {1: 'a', 2: 'a'}, 'a', 'a'

    default_same = 'a=1', {1: 'a', 2: 'a'}, 'a=1', 'a=1'
    default_diff = 'a=None', {1: 'a', 2: 'a'}, 'a=1', 'a=2'
    default_one = 'a', {1: 'a', 2: 'a'}, 'a=1', 'a'
    default_one_r = 'a', {1: 'a', 2: 'a'}, 'a', 'a=1'

    with repeated_test.options(downgrade_sig=False):
        annotation_both_diff = 'a', {1: 'a', 2: 'a'}, 'a:1', 'a:2'
        annotation_both_same = 'a:1', {1: 'a', 2: 'a'}, 'a:1', 'a:1'
        annotation_left = 'a:1', {1: 'a', 2: 'a'}, 'a:1', 'a'
        annotation_right = 'a:1', {1: 'a', 2: 'a'}, 'a', 'a:1'

    star_erase = '', {}, '*args', ''
    star_same = '*args', {1: ['args'], 2: ['args']}, '*args', '*args'
    star_name = '*largs', {1: ['largs']}, '*largs', '*rargs'
    star_same_pok = (
        'a, *args', Od([(1, ['a', 'args']), (2, ['a', 'args'])]),
        'a, *args', 'a, *args')
    star_extend = '<a>, *args', {1: ['a', 'args']}, '<a>, *args', '*args'
    star_extend_r = '<a>, *args', {2: ['a', 'args']}, '*args', '<a>, *args'

    stars_erase = '', {}, '**kwargs', ''
    stars_same = '**kwargs', {1: ['kwargs'], 2: ['kwargs']}, '**kwargs', '**kwargs'
    stars_name = '**lkwargs', {1: ['lkwargs']}, '**lkwargs', '**rkwargs'
    stars_extend = '*, a, **kwargs', {2: ['a', 'kwargs']}, '**kwargs', '*, a, **kwargs'
    stars_extend_r = '*, a, **kwargs', {1: ['a', 'kwargs']}, '*, a, **kwargs', '**kwargs'

    def test_omit_sources(self):
        s1 = s('a, *args, **kwargs')
        s2 = s('a, *args, **kwargs')
        ret = merge(s1, s2)
        self.assertSigsEqual(ret, s('a, *args, **kwargs'))

    three = '*, a, b, c, **k', {1: 'a', 2: 'b', 3: 'ck'}, '*, a, **k', '*, b, **k', '*, c, **k'


class MergeRaiseTests(Fixtures):
    def _test(self, *signatures):
        assert len(signatures) >= 2
        sigs = [s(sig) for sig in signatures]
        self.assertRaises(IncompatibleSignatures, merge, *sigs)

    posarg_raise = '', '<a>'
    pokarg_raise = '', 'a'

    kwarg_raise = '*, a', ''
    kwarg_r_raise = '', '*, a'
