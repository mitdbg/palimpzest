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


from sigtools.tests.util import SignatureTests
from sigtools.wrappers import Combination
from sigtools.support import s, f
from sigtools.specifiers import signature

class CombinationTests(SignatureTests):
    def test_result(self):
        def func1(arg, **kwargs): return arg + kwargs.pop('a')
        def func2(arg, **kwargs): return arg + kwargs.pop('b')
        def func3(arg, **kwargs): return arg + kwargs.pop('c')
        c = Combination(func1, func2, func3)
        self.assertEqual('0123', c('0', a='1', b='2', c='3'))

    def test_sig(self):
        self.assertSigsEqual(
            s('arg, *, a, b, c, **kwargs'),
            signature(Combination(
                f('arg, *, a, **kwargs'),
                f('arg, *, b, **kwargs'),
                f('arg, *, c, **kwargs')))
            )

    def test_repr(self):
        c = Combination('a', 'b', 'c')
        self.assertEqual("sigtools.wrappers.Combination('a', 'b', 'c')",
                         repr(c))

    def test_extend(self):
        def func1(arg, **kwargs): raise NotImplementedError
        def func2(arg, **kwargs): raise NotImplementedError
        def func3(arg, **kwargs): raise NotImplementedError
        def func4(arg, **kwargs): raise NotImplementedError
        c1 = Combination(func1, func2)
        c2 = Combination(func3, c1, func4)
        self.assertEqual(c2.functions, [func3, func1, func2, func4])
        self.assertSigsEqual(s('arg, **kwargs'), signature(c2))
