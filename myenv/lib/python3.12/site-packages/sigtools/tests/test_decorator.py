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


from sigtools import wrappers, support, signatures
from sigtools.tests.util import tup, Fixtures


def getclosure(obj):
    try:
        return obj.__closure__
    except AttributeError:
        return obj.func_closure


def getcode(obj):
    try:
        return obj.__code__
    except AttributeError:
        return obj.func_code


class WrapperTests(Fixtures):
    def _test(self, func, sig_str, args, kwargs, ret, exp_sources, decorators):
        """Tests .wrappers.decorator

        Checks its reported signature, that it chains functions correctly
        and that it its decorators are in the signature sources
        """
        sig = signatures.signature(func)
        self.assertSigsEqual(sig, support.s(sig_str))
        self.assertEqual(func(*args, **kwargs), ret)
        self.assertSourcesEqual(sig.sources, exp_sources,
                                func, depth_order=True)
        self.assertEqual(
            list(wrappers.wrappers(func)),
            [getclosure(d)[getcode(d).co_freevars.index('func')].cell_contents
             for d in decorators])

    @wrappers.decorator
    def _deco_all(func, a, b, *args, **kwargs):
        return a, b, func(*args, **kwargs)

    def test_decorator_repr(self):
        repr(self._deco_all)

    @tup('a, b, j, k, l', (1, 2, 3, 4, 5), {}, (1, 2, (3, 4, 5)),
         {0: 'jkl', '_deco_all':'ab', '+depths': ['_deco_all', 'func']},
         [_deco_all])
    @_deco_all
    def func(j, k, l):
        return j, k, l

    @_deco_all
    def _method(self, n, o):
        return self, n, o

    def test_bound_wrapped_repr(self):
        repr(self._method)

    def test_bound(self):
        self._test(
            self._method, 'a, b, n, o',
            (1, 2, 3, 4), {}, (1, 2, (self, 3, 4)),
            {0: 'no', '_deco_all': 'ab', '+depths': ['_deco_all', '_method']},
            [self._deco_all])

    @staticmethod
    @_deco_all
    def _static(d, e, f):
        raise NotImplementedError

    def test_wrapped_repr(self):
        repr(self._static)

    @wrappers.decorator
    def _deco_pos(func, p, q, *args, **kwargs):
        return p, func(q, *args, **kwargs)

    @tup('p, q, ma, mb', (1, 2, 3, 4), {}, (1, (2, 3, 4)),
         {0: ['ma', 'mb'], '_deco_pos': 'pq',
          '+depths': ['_deco_pos', 'masked']},
         [_deco_pos])
    @_deco_pos
    def masked(mq, ma, mb):
        return mq, ma, mb

    @_deco_all
    @_deco_pos
    def _chain(ca, cb, cc):
        return ca, cb, cc

    chain = (
        _chain, 'a, b, p, q, cb, cc',
        (1, 2, 3, 4, 5, 6), {}, (1, 2, (3, (4, 5, 6))),
        {0: ['cb', 'cc'], '_deco_all': 'ab', '_deco_pos': 'pq',
         '+depths': ['_deco_all', '_deco_pos', '_chain']},
        [_deco_all, _deco_pos])

    def _deco_classic(func):
        def wrapper(d, e, *args, **kwargs): return d, e, func(*args, **kwargs)
        wrapper.__name__ = '_deco_classic'
        wrapper.__wrapped__ = func
        return wrapper

    @tup('a, b, d, e, j, k, l',
         (1, 2, 3, 4, 5, 6, 7), {}, (1, 2, (3, 4, (5, 6, 7))),
         {'partial_': 'jkl', '_deco_all': 'ab', '_deco_classic': 'de',
          '+depths': ['_deco_all', '_deco_classic', 'partial_']},
         [_deco_all])
    @_deco_all
    @_deco_classic
    def partial_(j, k, l):
        return j, k, l
