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


import sys

from sigtools import modifiers, specifiers, support, _util, signatures
from sigtools.tests.util import Fixtures, SignatureTests, tup

import unittest


# bulk of the testing happens in test_merge and test_embed

not_py33 = sys.version_info < (3,3)


def _func(*args, **kwargs):
    raise NotImplementedError

def _free_func(x, y, z):
    raise NotImplementedError

class _cls(object):
    method = _func
_inst = _cls()
_im_type = type(_inst.method)



class MiscTests(unittest.TestCase):
    def test_sigtools_signature(self):
        import sigtools
        self.assertEqual(sigtools.signature, specifiers.signature)


class ForwardsTest(Fixtures):
    def _test(self, outer, inner, args, kwargs,
                    expected, expected_get, exp_src, exp_src_get):
        outer_f = support.f(outer, name='outer')
        inner_f = support.f(inner, name='inner')
        forw = specifiers.forwards_to_function(inner_f, *args, **kwargs)(outer_f)

        sig = specifiers.signature(forw)
        self.assertSigsEqual(sig, support.s(expected))

        self.assertSourcesEqual(sig.sources, {
                'outer': exp_src[0], 'inner': exp_src[1],
                '+depths': ['outer', 'inner']})

        sig_get = specifiers.signature(_util.safe_get(forw, object(), object))
        self.assertSigsEqual(sig_get, support.s(expected_get))

        self.assertSourcesEqual(sig_get.sources, {
                'outer': exp_src_get[0], 'inner': exp_src_get[1],
                '+depths': ['outer', 'inner']})

    a = (
        'a, *p, b, **k', 'c, *, d', (), {},
        'a, c, *, b, d', 'c, *, b, d', ['ab', 'cd'], ['b', 'cd'])
    b = (
        'a, *p, b, **k', 'a, c, *, b, d', (1, 'b'), {},
        'a, c, *, b, d', 'c, *, b, d', ['ab', 'cd'], ['b', 'cd'])

    def test_call(self):
        outer = support.f('*args, **kwargs')
        inner = support.f('a, *, b')
        forw = specifiers.forwards_to_function(inner)(outer)
        instance = object()
        forw_get_prox = _util.safe_get(forw, instance, object)
        self.assertEqual(
            forw_get_prox(1, b=2),
            {'args': (instance, 1), 'kwargs': {'b': 2}})

def sig_equal(self, obj, sig_str, exp_src):
    sig = specifiers.signature(obj)
    self.assertSigsEqual(sig, support.s(sig_str),
                         conv_first_posarg=True)
    self.assertSourcesEqual(sig.sources, exp_src, obj)

class _Coop(object):
    @modifiers.kwoargs('cb')
    def method(self, ca, cb, *cr, **ck):
        raise NotImplementedError

class ForwardsAttributeTests(Fixtures):
    _test = sig_equal

    class _Base(object):
        def __init__(self, decorated=None):
            self.decorated = decorated
            self.coop = _Coop()

        @modifiers.kwoargs('b')
        def inner(self, a, b):
            raise NotImplementedError

        @specifiers.forwards_to(_free_func)
        def ft(self, *args, **kwargs):
            raise NotImplementedError

        @specifiers.forwards_to_method('inner')
        def ftm(self, *args, **kwargs):
            raise NotImplementedError

        @specifiers.forwards_to_ivar('decorated')
        def fti(self, *args, **kwargs):
            raise NotImplementedError

        @specifiers.forwards_to_method('ftm')
        @modifiers.kwoargs('d')
        def ftm2(self, c, d, *args2, **kwargs2):
            raise NotImplementedError

        @modifiers.kwoargs('m')
        def fts(self, l, m):
            raise NotImplementedError

        @modifiers.kwoargs('o')
        def afts(self, n, o):
            raise NotImplementedError

        @specifiers.forwards_to_method('ftm2')
        @modifiers.kwoargs('q')
        def chain_fts(self, p, q, *args, **kwargs):
            raise NotImplementedError

        @specifiers.forwards_to_method('ftm2')
        @modifiers.kwoargs('s')
        def chain_afts(self, r, s, *args, **kwargs):
            raise NotImplementedError

        @specifiers.forwards_to_method('coop.method')
        @modifiers.kwoargs('bc')
        def ccm(self, ba, bb, bc, *args, **kwargs):
            raise NotImplementedError

    @_Base
    @modifiers.kwoargs('b')
    def _base_inst(a, b):
        raise NotImplementedError

    @specifiers.apply_forwards_to_super('afts', 'chain_afts')
    class _Derivate(_Base):
        @specifiers.forwards_to_method('inner')
        def ftm(self, e, *args, **kwoargs):
            raise NotImplementedError

        @specifiers.forwards_to_super()
        def fts(self, s, *args, **kwargs):
            super() # pragma: no cover

        def afts(self, asup, *args, **kwargs):
            raise NotImplementedError

        @specifiers.forwards_to_super()
        def chain_fts(self, u, *args, **kwargs):
            super() # pragma: no cover

        def chain_afts(self, v, *args, **kwargs):
            raise NotImplementedError

    @_Derivate
    @modifiers.kwoargs('y')
    def _sub_inst(x, y):
        raise NotImplementedError

    base_func = _base_inst.ft, 'x, y, z', {'_free_func': 'xyz'}

    base_method = _base_inst.ftm, 'a, *, b', {'inner': 'ab'}
    base_method2 = _base_inst.ftm2, 'c, a, *, d, b', {
        'ftm2': 'cd', 'inner': 'ab', '+depths': ['ftm2', 'ftm', 'inner']}

    base_method_cls = _Base.ftm, 'self, *args, **kwargs', {
        'ftm': ['self', 'args', 'kwargs']}

    base_ivar = _base_inst.fti, 'a, *, b', {'_base_inst': 'ab'}

    base_coop = _base_inst.ccm, 'ba, bb, ca, *cr, bc, cb, **ck', {
        'ccm': ['ba', 'bb', 'bc'], 'method': ['ca', 'cb', 'cr', 'ck'] }

    sub_method = _sub_inst.ftm, 'e, a, *, b', {'ftm': 'e', 'inner': 'ab'}

    sub_method2 = _sub_inst.ftm2, 'c, e, a, *, d, b', {
        'ftm2': 'cd', 'ftm': 'e', 'inner': 'ab'}

    sub_ivar = _sub_inst.fti, 'x, *, y', {'_sub_inst': 'xy'}

    @tup('a, y=None, z=None', {0: 'a', '_free_func': 'yz'})
    @specifiers.forwards_to_function(_free_func, 1, partial=True)
    def forwards_to_partial(a, *args, **kwargs):
        raise NotImplementedError

    def _test_raw_source(self, obj, exp_sig, exp_src):
        sig = specifiers.signature(obj)
        self.assertSigsEqual(sig, support.s(exp_sig), conv_first_posarg=True)
        self.assertEqual(sig.sources, exp_src)

    def test_fts(self):
        if sys.version_info >= (3,3):
            sup = super(self._Derivate, self._sub_inst).fts
            sig = specifiers.signature(self._sub_inst.fts)
            self.assertSigsEqual(sig, support.s('s, l, *, m'))
            self._test_raw_source(self._sub_inst.fts, 's, l, *, m', {
                    '+depths': {self._sub_inst.fts: 0, sup: 1},
                    's': [self._sub_inst.fts],
                    'l': [sup], 'm': [sup]
                })

    def test_sub_afts_cls(self):
        fun = self._Derivate.afts
        self._test_raw_source(
            fun, 'self, asup, *args, **kwargs', {
            '+depths': {fun: 0},
            'self': [fun], 'asup': [fun], 'args': [fun], 'kwargs': [fun]
        })

    def test_sub_afts(self):
        fun = self._sub_inst.afts
        sup = super(self._Derivate, self._sub_inst).afts
        self._test_raw_source(
            fun, 'asup, n, *, o',
            {'+depths': {fun: 0, sup: 1},
             'asup': [fun], 'n': [sup], 'o': [sup]})

    def test_chain_fts(self):
        if sys.version_info < (3,3):
            return

        fun = self._Derivate.chain_fts
        self._test_raw_source(
            fun, 'self, u, *args, **kwargs',
            {'+depths': {fun: 0},
             'self': [fun], 'u': [fun], 'args': [fun], 'kwargs': [fun]})

        inst = self._sub_inst
        sup = super(self._Derivate, self._sub_inst)
        self._test_raw_source(
            inst.chain_fts, 'u, p, c, e, a, *, d, b, q', {
                '+depths': {inst.chain_fts: 0, sup.chain_fts: 1,
                            inst.ftm2: 2, inst.ftm: 3, inst.inner:4},
                'u': [inst.chain_fts],
                'p': [sup.chain_fts], 'q': [sup.chain_fts],
                'c': [inst.ftm2], 'd': [inst.ftm2],
                'e': [inst.ftm],
                'a': [inst.inner], 'b': [inst.inner]
            })

    def test_chain_afts_cls(self):
        fun = self._Derivate.chain_afts
        self._test_raw_source(
            fun, 'self, v, *args, **kwargs',
            {'+depths': {fun: 0},
             'self': [fun], 'v': [fun], 'args': [fun], 'kwargs': [fun]})

    def test_chain_afts(self):
        inst = self._sub_inst
        sup = super(self._Derivate, self._sub_inst)
        self._test_raw_source(
            inst.chain_afts, 'v, r, c, e, a, *, d, b, s', {
                '+depths': {
                    inst.chain_afts: 0, sup.chain_afts: 1, inst.ftm2: 2,
                            inst.ftm: 3, inst.inner: 4},
                'v': [inst.chain_afts],
                'r': [sup.chain_afts], 's': [sup.chain_afts],
                'c': [inst.ftm2], 'd': [inst.ftm2],
                'e': [inst.ftm],
                'a': [inst.inner], 'b': [inst.inner]})

    def test_transform(self):
        class _callable(object):
            def __call__(self):
                raise NotImplementedError

        class Cls(object):
            @specifiers.forwards_to_method('__init__', emulate=True)
            def __new__(cls):
                raise NotImplementedError

            def __init__(self):
                raise NotImplementedError

            abc = None
            if sys.version_info >= (3,):
                abc = specifiers.forwards_to_method('__init__', emulate=True)(
                    _callable()
                    )
        Cls.abc
        Cls.__new__
        self.assertEqual(type(Cls.__dict__['__new__'].__wrapped__),
                         staticmethod)
        Cls.__new__
        self.assertEqual(type(Cls.__dict__['__new__'].__wrapped__),
                         staticmethod)

    def test_emulation(self):
        func = specifiers.forwards_to_method('abc', emulate=False)(_func)
        self.assertTrue(_func is func)

        func = specifiers.forwards_to_method('abc')(_func)
        self.assertTrue(_func is func)

        class Cls(object):
            func = specifiers.forwards_to_method('abc')(_func)
        func = getattr(Cls.func, '__func__', func)
        self.assertTrue(_func is func)
        self.assertTrue(_func is Cls().func.__func__)

        class Cls(object):
            def func(self, abc, *args, **kwargs):
                raise NotImplementedError
            def abc(self, x):
                raise NotImplementedError
        method = Cls().func
        func = specifiers.forwards_to_method('abc')(method)
        self.assertTrue(isinstance(func, specifiers._ForgerWrapper))
        self.assertEqual(func.__wrapped__, method)
        self.assertRaises(
            AttributeError,
            specifiers.forwards_to_method('abc', emulate=False), Cls().func)
        exp = support.s('abc, x')
        self.assertSigsEqual(signatures.signature(func), exp)
        self.assertSigsEqual(specifiers.signature(func), exp)

        class Emulator(object):
            def __init__(self, obj, forger):
                self.obj = obj
                self.forger = forger

        func = specifiers.forwards_to_function(func, emulate=Emulator)(_func)
        self.assertTrue(isinstance(func, Emulator))

        @specifiers.forwards_to_function(_func, emulate=True)
        def func(x, y, *args, **kwargs):
            return x + y
        self.assertEqual(5, func(2, 3))

    def test_super_fail(self):
        class Cls(object):
            def m(self):
                raise NotImplementedError
            def n(self):
                raise NotImplementedError
        class Sub(Cls):
            @specifiers.forwards_to_super()
            def m(self, *args, **kwargs):
                raise NotImplementedError
            @specifiers.forwards_to_super()
            def n(self, *args, **kwargs):
                super(Sub, self).n(*args, **kwargs) # pragma: no cover
        self.assertRaises(ValueError, specifiers.signature, Sub().m)
        if sys.version_info < (3,):
            self.assertRaises(ValueError, specifiers.signature, Sub().n)


class ForgerFunctionTests(SignatureTests):
    def test_deco(self):
        @specifiers.forger_function
        def forger(obj):
            return support.s('abc')
        @forger()
        def forged():
            raise NotImplementedError
        self.assertSigsEqual(support.s('abc'), specifiers.signature(forged))

    def test_forger_sig(self):
        @specifiers.forger_function
        def forger(p1, p2, p3, obj):
            raise NotImplementedError
        self.assertSigsEqual(
            support.s('p1, p2, p3, *, emulate=None'),
            specifiers.signature(forger))

    def test_directly_applied(self):
        def forger(obj):
            return support.s('abc')
        def forged():
            raise NotImplementedError
        specifiers.set_signature_forger(forged, forger)
        self.assertSigsEqual(support.s('abc'), specifiers.signature(forged))

    def test_forger_lazy(self):
        class Flag(Exception): pass
        @specifiers.forger_function
        def forger(obj):
            raise Flag
        @forger()
        def forged():
            pass
        self.assertSigsEqual(forged(), None)
        self.assertRaises(Flag, specifiers.signature, forged)

    def test_orig_sig(self):
        @specifiers.forger_function
        def forger(obj):
            return None
        @forger()
        def forged(alpha):
            raise NotImplementedError
        self.assertSigsEqual(support.s('alpha'), specifiers.signature(forged))

    def test_as_forged(self):
        sig = support.s('this, isa, test')
        @specifiers.forger_function
        def forger(obj):
            return sig
        class MyClass(object):
            __signature__ = specifiers.as_forged
            def __init__(self):
                forger()(self)
            def __call__(self):
                raise NotImplementedError
        self.assertSigsEqual(signatures.signature(MyClass()), sig)

    def test_as_forged_forwards(self):
        def function(a, b, c):
            raise NotImplementedError
        class MyClass(object):
            __signature__ = specifiers.as_forged
            def __init__(self):
                specifiers.forwards_to_function(function)(self)
            def __call__(self, x, *args, **kwargs):
                raise NotImplementedError
        self.assertSigsEqual(signatures.signature(MyClass()),
                             support.s('x, a, b, c'))

    def test_dunder_call(self):
        sig = support.s('dunder, call')
        @specifiers.forger_function
        def forger(obj):
            return sig
        class MyClass(object):
            def __init__(self):
                forger()(self)
            def __call__(self):
                raise NotImplementedError
        self.assertSigsEqual(specifiers.signature(MyClass()), sig)

    def test_as_forged_dunder_call_method(self):
        class MyClass(object):
            __signature__ = specifiers.as_forged
            @specifiers.forwards_to_method('method')
            def __call__(self, x, *args, **kwags):
                raise NotImplementedError
            def method(self, a, b, c):
                raise NotImplementedError
        exp = support.s('x, a, b, c')
        self.assertSigsEqual(signatures.signature(MyClass()), exp)
        self.assertSigsEqual(specifiers.signature(MyClass()), exp)

    def test_forger_priority_over_autoforwards_hint(self):
        def make_func():
            def real_inner(x, y, z):
                pass
            def inner(i, j):
                raise NotImplementedError
            @specifiers.forwards_to_function(inner)
            @modifiers.kwoargs('a')
            def outer(a, *args, **kwargs):
                real_inner(*args, **kwargs)
            return outer
        func = make_func()
        self.assertSigsEqual(
            support.s('i, j, *, a'),
            specifiers.signature(func))
        func(1, 2, 3, a=4)
