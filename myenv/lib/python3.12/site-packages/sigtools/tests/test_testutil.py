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


import unittest

from sigtools._util import OrderedDict as od
from sigtools.support import s
from sigtools.tests import util as tutil


class UtilTests(unittest.TestCase):
    def test_conv_first_posarg(self):
        self.assertEqual(s(''), tutil.conv_first_posarg(s('')))
        self.assertEqual(
            s('one, /, two, *three, four, **five'),
            tutil.conv_first_posarg(s('one, two, *three, four, **five')))


class TransformExpectedSourcesTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(tutil.transform_exp_sources({}), {'+depths': {}})

    def test_tf(self):
        self.assertEqual(
            tutil.transform_exp_sources({'func': 'abc', 'func2': 'def'}),
            { 'a': ['func'], 'b': ['func'], 'c': ['func'],
              'd': ['func2'], 'e': ['func2'], 'f': ['func2'],
              '+depths': {'func': 0, 'func2': 1} })

    def test_order(self):
        self.assertEqual(
            tutil.transform_exp_sources(od([('func', 'abc'), ('func2', 'abc')])),
            { 'a': ['func', 'func2'],
              'b': ['func', 'func2'],
              'c': ['func', 'func2'],
              '+depths': {'func': 0, 'func2': 1} })

    def test_implicit(self):
        self.assertEqual(
            tutil.transform_exp_sources({0: 'abc', 'func1': 'def'}, subject='func2'),
            { 'a': ['func2'], 'b': ['func2'], 'c': ['func2'],
              'd': ['func1'], 'e': ['func1'], 'f': ['func1'],
              '+depths': {'func2': 0, 'func1': 1} })

    def test_implicit_missing(self):
        with self.assertRaises(ValueError):
            tutil.transform_exp_sources({0: 'abc', 'func2': 'def'})

    def test_conv(self):
        self.assertEqual(
            tutil.transform_exp_sources({1: 'abc', 2: 'def'}),
            { 'a': ['_1'], 'b': ['_1'], 'c': ['_1'],
              'd': ['_2'], 'e': ['_2'], 'f': ['_2'],
              '+depths': {'_1': 0, '_2': 1} })

    def test_func_name(self):
        def f1():
            raise NotImplementedError
        self.assertEqual(
            tutil.transform_exp_sources({f1: 'abc', 'func2': 'def'}),
            { 'a': ['f1'], 'b': ['f1'], 'c': ['f1'],
              'd': ['func2'], 'e': ['func2'], 'f': ['func2'],
              '+depths': {'f1': 0, 'func2': 1} })

    def test_implicit_func(self):
        def f1():
            raise NotImplementedError
        self.assertEqual(
            tutil.transform_exp_sources({0: 'abc', 'func2': 'def'}, subject=f1),
            { 'a': ['f1'], 'b': ['f1'], 'c': ['f1'],
              'd': ['func2'], 'e': ['func2'], 'f': ['func2'],
              '+depths': {'f1': 0, 'func2': 1} })

    def test_copy_depth(self):
        def f1():
            raise NotImplementedError
        self.assertEqual(
            tutil.transform_exp_sources(
                {'+depths': {f1: 0, 'f2': 0}, f1: 'a', 'f2': 'b'}),
            { 'a': ['f1'] , 'b': ['f2'], '+depths': {'f1': 0, 'f2': 0} })

    def test_depth_list(self):
        def f1():
            raise NotImplementedError
        self.assertEqual(
            tutil.transform_exp_sources(
                {'+depths': ['f2', f1], f1: 'a', 'f2': 'b'}),
            { 'a': ['f1'] , 'b': ['f2'], '+depths': {'f2': 0, 'f1': 1} })


class TransformRealSourcesTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(tutil.transform_real_sources({}), {})

    def test_tf(self):
        def f1():
            raise NotImplementedError
        def f2():
            raise NotImplementedError
        self.assertEqual(
            tutil.transform_real_sources({'a': [f1, f2], 'b': [f1], 'c': [f2]}),
            {'a': ['f1', 'f2'], 'b': ['f1'], 'c': ['f2']})

    def test_depth(self):
        def f1():
            raise NotImplementedError
        def f2():
            raise NotImplementedError
        self.assertEqual(
            tutil.transform_real_sources(
                {'+depths': {f1: 0, f2: 1},
                'a': [f1, f2], 'b': [f1], 'c': [f2]}),
            {'a': ['f1', 'f2'], 'b': ['f1'], 'c': ['f2'],
             '+depths': {'f1': 0, 'f2': 1}})


class SignatureTestsTests(tutil.SignatureTests):
    def test_sigs_equal(self):
        self.assertSigsEqual(s('one'), s('one'))
        self.assertSigsEqual(s('*, one'), s('*, one'))

        with self.assertRaises(AssertionError):
            self.assertSigsEqual(s('one'), s('two'))
        with self.assertRaises(AssertionError):
            self.assertSigsEqual(s('one'), s('*, one'))

    def test_sigs_equal_conv_first(self):
        self.assertSigsEqual(s('self, /, one'), s('self, one'),
                             conv_first_posarg=True)
        self.assertSigsEqual(s('self, one'), s('self, /, one'),
                             conv_first_posarg=True)
        self.assertSigsEqual(s('self, /, *, one'), s('self, *, one'),
                             conv_first_posarg=True)
        self.assertSigsEqual(s('self, *, one'), s('self, /, *, one'),
                             conv_first_posarg=True)

        with self.assertRaises(AssertionError):
            self.assertSigsEqual(s('self, /, one'), s('self, two'),
                                 conv_first_posarg=True)
        with self.assertRaises(AssertionError):
            self.assertSigsEqual(s('self, /, one'), s('self, *, one'),
                                 conv_first_posarg=True)

    @unittest.skipIf(*tutil.python_doesnt_have_future_annotations)
    def test_sigs_equal_evaluated_annotation_different(self):
        with self.assertRaisesRegex(AssertionError, "^.*(one: 2).*(one: 1).*$"):
            self.assertSigsEqual(
                s("one: a", globals={"a": 1}, future_features=["annotations"]),
                s("one: a", globals={"a": 2}, future_features=["annotations"]),
            )

    def test_assertIs(self):
        self.assertIs(*([],)*2)
        with self.assertRaises(AssertionError):
            self.assertIs([], [])

    def test_src_eq(self):
        def f1():
            raise NotImplementedError
        def f2():
            raise NotImplementedError
        self.assertSourcesEqual(
            {'a': [f1], 'b': [f2], 'c': [f1], '+depths': {f1: 0, f2: 1}},
            {'f1': 'ac', 'f2': 'b'})
        with self.assertRaises(AssertionError):
            self.assertSourcesEqual(
                {'a': [f1], 'b': [f2], 'c': [f1], '+depths': {f1: 0, f2: 1}},
                {'f1': 'a', 'f2': 'b'})
