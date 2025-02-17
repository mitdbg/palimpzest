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


import unittest

from sigtools.tests import sphinxextfixt, util


app = object()


class SphinxExtTests(unittest.TestCase):
    def setUp(self):
        try:
            from sigtools import sphinxext
        except SyntaxError: # sphinx does not work on py32
            raise unittest.SkipTest("Sphinx could not be imported.")
        else:
            self.sphinxext = sphinxext

    def test_forge(self):
        r = self.sphinxext.process_signature(
            app, 'function', 'sigtools.tests.sphinxextfixt.outer',
            sphinxextfixt.outer, {}, '(c, *args, **kwargs)', None)
        self.assertEqual(('(c, a, b)', ''), r)

    def test_method_forge(self):
        r = self.sphinxext.process_signature(
            app, 'method', 'sigtools.tests.sphinxextfixt.AClass.outer',
            sphinxextfixt.AClass.outer, {}, '(c, *args, **kwargs)', None)
        self.assertEqual(('(c, a, b)', ''), r)

    def test_modifiers(self):
        r = self.sphinxext.process_signature(
            app, 'function', 'sigtools.tests.sphinxextfixt.kwo',
            sphinxextfixt.AClass.outer, {}, '(a, b, c=1, d=2)', None)
        self.assertEqual(('(a, b, *, c=1, d=2)', ''), r)

    def test_autoforward(self):
        r = self.sphinxext.process_signature(
            app, 'function', 'sigtools.tests.sphinxextfixt.autoforwards',
            sphinxextfixt.autoforwards, {}, '(d, *args, **kwargs)', None)
        self.assertEqual(('(d, a, b)', ''), r)

    def test_attribute(self):
        r = self.sphinxext.process_signature(
            app, 'attribute', 'sigtools.tests.sphinxextfixt.AClass.class_attr',
            sphinxextfixt.AClass.class_attr, {}, None, None)
        self.assertEqual((None, None), r)

    def test_inst_attr(self):
        r = self.sphinxext.process_signature(
            app, 'attribute', 'sigtools.tests.sphinxextfixt.AClass.abc',
            None, {}, None, None)
        self.assertEqual((None, None), r)

    @unittest.skipUnless(*util.python_has_optional_future_annotations)
    def test_attrs_class(self):
        r = self.sphinxext.process_signature(
            app, '', 'sigtools.tests.sphinxextfixt.AttrsClass',
            None, {}, None, None
        )
        self.assertEqual(('(one, *, two)', ''), r)
