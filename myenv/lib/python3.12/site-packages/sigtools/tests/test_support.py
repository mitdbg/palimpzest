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


import sys
import unittest

from repeated_test import options

from sigtools import support, _specifiers
from sigtools.tests.util import Fixtures, python_has_future_annotations


def remove_spaces(s):
    return s.strip().replace(' ', '')


force_modifiers = {
        'use_modifiers_annotate': True,
        'use_modifiers_posoargs': True,
        'use_modifiers_kwoargs': True,
}


class RoundTripTests(Fixtures):
    def _test(self, sig_str, old_fmt=None):
        def assert_signature_matches_original(sig):
            self._assert_equal_ignoring_spaces(f'({sig_str})', str(sig), f'({old_fmt})')

        # Using inspect.signature
        sig = support.s(sig_str)
        assert_signature_matches_original(sig)

        # Roundtrip and use forged signature
        pf_sig = _specifiers.forged_signature(support.func_from_sig(sig))
        assert_signature_matches_original(pf_sig)

        # Using sigtools.modifiers and and inspect.signature
        m_sig = support.s(sig_str, **force_modifiers)
        assert_signature_matches_original(m_sig)

        # Roundtrip and use forged signature
        pmf_sig = _specifiers.forged_signature(support.func_from_sig(m_sig))
        assert_signature_matches_original(pmf_sig)

    def _assert_equal_ignoring_spaces(self, expected, actual, expected_old_fmt=None):
        try:
            self.assertEqual(
                remove_spaces(expected),
                remove_spaces(actual),
            )
        except AssertionError:
            if expected_old_fmt is None: raise
            self.assertEqual(
                remove_spaces(expected_old_fmt),
                remove_spaces(actual),
            )

    empty = '',

    pok = 'a, b',
    pos = 'a, /, b', '<a>, b'
    pos_old = '<a>, b', 'a, /, b'

    default = 'a=1',
    varargs = '*args',
    varkwargs = '**kwargs',

    kwo = '*args, a',
    kwo_novarargs = '*, a',
    kwo_order = 'a, b=1, *args, c, d, e, f=4',

    defaults = 'a, b=1, *, c, d=1',
    default_after_star = 'a, b, *, c, d=1, e=2',

    annotate = 'a:1, /, b:2, *c:3, d:4, **e:5', '<a>:1, b:2, *c:3, d:4, **e:5'

    def test_return_annotation(self):
        self._assert_equal_ignoring_spaces('() -> 2', str(support.s('', 2)))
        self._assert_equal_ignoring_spaces('() -> 3', str(support.s('', ret=3)))
        self._assert_equal_ignoring_spaces('(a:4) -> 5', str(support.s('a:4', 5)))
        self._assert_equal_ignoring_spaces('(b:6) -> 7', str(support.s('b:6', ret=7)))

    def test_locals(self):
        obj = object()
        sig = support.s('a:o', locals={'o': obj})
        self.assertIs(obj, sig.parameters['a'].annotation)

    def test_name(self):
        func = support.f('a, b, c', name='test_name')
        self._assert_equal_ignoring_spaces(func.__name__, 'test_name')

    @unittest.skipUnless(python_has_future_annotations, "requires python with optional deferred annotations")
    def test_deferred_annotations(self):
        deferred = support.s('a: 1', future_features=("annotations",))
        manually_deferred = support.s('a: "1"')
        self._assert_equal_ignoring_spaces(str(manually_deferred), str(deferred))

class FuncCodeTests(Fixtures):
    def _test(self, sig, expected_code, kwargs={}, *, min_version=None, max_version=None):
        if min_version is not None and sys.version_info < min_version:
            self.skipTest(f"Python version too low for this test ({sys.version_info} < {min_version})")
        if max_version is not None and sys.version_info >= max_version:
            self.skipTest(f"Python version too high for this test ({sys.version_info} >= {min_version})")
        actual_code = support.func_code(*support.read_sig(sig, **kwargs))
        self.assertEqual(remove_spaces(expected_code), remove_spaces(actual_code))

    empty = "", """
        def func():
            return {}
    """

    posoarg_py38 = "a, /", """
        def func(a, /):
            return {'a': a}
    """, options(min_version=(3,8))

    posoarg_pre_py38 = "a, /", """
        @modifiers.posoargs('a')
        def func(a):
            return {'a': a}
    """, options(max_version=(3,8))

    posoarg_force_modifiers = "a, /", """
        @modifiers.posoargs('a')
        def func(a):
            return {'a': a}
    """, force_modifiers

    posoarg_chevron_py38 = "<a>", """
        def func(a, /):
            return {'a': a}
    """, options(min_version=(3,8))

    posoarg_chevron_and_slash_py38 = "<a>, b, /, c", """
        def func(a, b, /, c):
            return {'a': a, 'b': b, 'c': c}
    """, options(min_version=(3,8))

    posoarg_chevron_others_py38 = "<a>, b, *, c", """
        def func(a, /, b, *, c):
            return {'a': a, 'b': b, 'c': c}
    """, options(min_version=(3,8))

    posoarg_chevron_pre_py38 = "<a>", """
        @modifiers.posoargs('a')
        def func(a):
            return {'a': a}
    """, options(max_version=(3,8))

    posoarg_chevron_force_modifiers = "<a>", """
        @modifiers.posoargs('a')
        def func(a):
            return {'a': a}
    """, force_modifiers

    kwoargs = "*, a", """
        def func(*, a):
            return {'a': a}
    """

    kwoargs_force_modifiers = "*, a", """
        @modifiers.kwoargs('a')
        def func(a):
            return {'a': a}
    """, force_modifiers

    defaults = "a=1, /, b=2, *, c=3", """
        def func(a=1, /, b=2, *, c=3):
            return {'a': a, 'b': b, 'c': c}
    """, options(min_version=(3,8))

    defaults_starting_after_vararg = "a, *args, b=2", """
        def func(a, *args, b=2):
            return {'a': a, 'b': b, 'args': args}
    """

    defaults_starting_after_star = "a, *, b=2", """
        def func(a, *, b=2):
            return {'a': a, 'b': b}
    """

    varargs_varkwargs = "*args, **kwargs", """
        def func(*args, **kwargs):
            return {'args': args, 'kwargs': kwargs}
    """

    annotations = "a: 1", """
        def func(a: 1):
            return {'a': a}
    """

    annotations_modifiers = "a: 1", """
        @modifiers.annotate(a=1)
        def func(a):
            return {'a': a}
    """, force_modifiers

    return_annotation = "a", """
        def func(a) -> 1:
            return {'a': a}
    """, {'ret': 1}

    return_annotation_modifiers = "a", """
        @modifiers.annotate(1)
        def func(a):
            return {'a': a}
    """, {**force_modifiers, 'ret': 1}

    return_and_parameter_annotation_modifiers = "a: 2", """
        @modifiers.annotate(1, a=2)
        def func(a):
            return {'a': a}
    """, {**force_modifiers, 'ret': 1}
    kwo_after_default = "a, b=1, *args, c, d, e=4", """
        @modifiers.kwoargs('c', 'd', 'e')
        def func(a, c, d, b=1, e=4, *args):
            return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'args': args}
    """, force_modifiers
