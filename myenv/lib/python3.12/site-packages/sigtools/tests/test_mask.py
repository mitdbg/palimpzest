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

from sigtools import signatures
from sigtools.tests.util import FixturesWithFutureAnnotations


class MaskTests(FixturesWithFutureAnnotations):
    def _test(self, expected_str, sig_str, num_args=0, named_args=(),
                   hide_varargs=False, hide_varkwargs=False,
                   hide_args=False, hide_kwargs=False,
                   *, support_s, downgrade_sig):
        expected_sig = support_s(expected_str)

        in_sig = support_s(sig_str)

        with self.maybe_with_downgrade_and_ignore_warnings(downgrade_sig, self.downgrade_sig) as maybe_downgrade:
            sig = signatures.mask(
                maybe_downgrade(in_sig), num_args, *named_args,
                hide_varargs=hide_varargs, hide_varkwargs=hide_varkwargs,
                hide_args=hide_args, hide_kwargs=hide_kwargs)

        self.assertSigsEqual(sig, expected_sig)

        if not downgrade_sig:
            expected_src = {'func': expected_sig.parameters}
            self.assertSourcesEqual(sig.sources, expected_src, func='func')

    hide_pos = '<b>', '<a>, <b>', 1
    hide_pos_pok = 'c', '<a>, b, c', 2

    eat_into_varargs = '*args', 'a, *args', 2

    name_pok_last = 'a, b', 'a, b, c', 0, 'c'
    name_pok = 'a, *, c', 'a, b, c', 0, 'b'
    name_pok_last_hide_va = 'a, b', 'a, b, c, *args', 0, 'c'
    name_pok_hide_va = 'a, *, c', 'a, b, c, *args', 0, 'b'

    name_kwo = 'a', 'a, *, b', 0, 'b'

    name_varkwargs = '**kwargs', '**kwargs', 0, 'a'
    name_varkwargs_hide = '', '**kwargs', 0, 'a', False, True

    hide_varargs = 'a, b, *, c', 'a, b, *args, c', 0, '', True, False
    eat_into_varargs_hide = '', 'a, *args', 2, '', True, False

    hide_varargs_absent = '', '', 0, '', True, False
    hide_varkwargs_absent = '', '', 0, '', False, True

    hide_args = '*, x', 'a, /, b, *, x', 0, '', False, False, True, False
    hide_args_starargs = (
        '*, x', 'a, /, b, *args, x', 0, '', False, False, True, False)

    hide_kwargs = (
        'a, /, *args', 'a, /, b, *args, c, **kwargs',
        0, '', False, False, False, True)


@repeated_test.with_options(downgrade_sig=repeated_test.skip_option)
class MaskRaiseTests(FixturesWithFutureAnnotations):
    def _test(self, sig_str, num_args, named_args=(),
              hide_varargs=False, hide_varkwargs=False,
              *, support_s):
        sig = support_s(sig_str)
        self.assertRaises(
            ValueError, signatures.mask,
            sig, num_args, *named_args,
            hide_varargs=hide_varargs, hide_varkwargs=hide_varkwargs)

    no_pos_1 = '', 1
    no_pos_2 = '<a>', 2

    no_pok_2 = 'a', 2
    no_pos_pok_3 = '<a>, b', 3

    key_is_pos = '<a>', 0, 'a'
    key_absent = '', 0, 'a'

    key_twice = 'a', 0, 'aa'
    pos_and_key = 'a', 1, 'a'
