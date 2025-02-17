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

from functools import partial, wraps
from mock import patch
import ast

from sigtools import support, modifiers, specifiers, signatures, _util, _autoforwards
from sigtools.tests.util import Fixtures, tup


_wrapped = support.f('x, y, *, z', name='_wrapped')


def func(*args, **kwargs):
    pass


class AutoforwardsMarkerReprs(Fixtures):
    def _test(self, obj, exp_repr):
        self.assertEqual(repr(obj), exp_repr)

    _af = _autoforwards

    name = _af.Name('spam'), "<name 'spam'>"
    attribute = _af.Attribute(_af.Name('ham'), 'eggs'), "<attribute <name 'ham'>.eggs>"
    arg = _af.Arg('eggs'), "<argument 'eggs'>"

    unknown_unsourced = _af.Unknown(), '<irrelevant>'
    unknown_marker = (
        _af.Unknown(_af.Name('spam')),
        "<unknown until runtime: <name 'spam'>>")
    unknown_ast = (
        _af.Unknown(ast.Pass()),
        "<unknown until runtime: Pass()>")
    unknown_ast_list = (
        _af.Unknown([ast.Pass(), ast.Pass()]),
        "<unknown until runtime: [Pass(), Pass()]>")


class AutoforwardsTests(Fixtures):
    def _test(self, func, expected, sources, incoherent=False):
        sig = specifiers.signature(func)
        self.assertSigsEqual(sig, support.s(expected))
        self.assertSourcesEqual(sig.sources, sources, func)
        if not incoherent:
            support.assert_func_sig_coherent(
                func, check_return=False, check_invalid=False)

    @tup('a, b, x, y, *, z',
         {'global_': ('a', 'b'), '_wrapped': ('x', 'y', 'z')})
    def global_(a, b, *args, **kwargs):
        return _wrapped(*args, **kwargs)

    def _make_closure():
        wrapped = _wrapped
        def wrapper(b, a, *args, **kwargs):
            return wrapped(*args, **kwargs)
        return wrapper
    closure = (
        _make_closure(),
        'b, a, x, y, *, z', {'wrapper': 'ba', '_wrapped': 'xyz'})

    @tup('a, b, y', {'args': 'ab', '_wrapped': 'y'})
    def args(a, b, *args, **kwargs):
        return _wrapped(a, *args, z=b, **kwargs)

    @tup('a, b, *, z', {'using_other_varargs': 'ab', '_wrapped': 'z'})
    def using_other_varargs(a, b, **kwargs):
        return _wrapped(a, *b, **kwargs)

    # def test_external_args(self):
    #     def l1():
    #         a = None
    #         def l2(**kwargs):
    #             nonlocal a
    #             _wrapped(*a, **kwargs)
    #         return l2
    #     self._test_func(l1(), '*, z')

    @tup('x, y, /, *, kwop', {'kwo': ['kwop'], '_wrapped': 'xy'})
    @modifiers.kwoargs('kwop')
    def kwo(kwop, *args):
        _wrapped(*args, z=kwop)

    @tup('a, b, y, *, z', {'subdef': 'ab', '_wrapped': 'yz'})
    def subdef(a, b, *args, **kwargs):
        def func():
            _wrapped(42, *args, **kwargs)
        func()

    @tup('a, b, y, *, z', {'subdef_lambda': 'ab', '_wrapped': 'yz'})
    def subdef_lambda(a, b, *args, **kwargs):
        (lambda: _wrapped(42, *args, **kwargs))()

    @tup('a, b, x, y, *, z', {0: 'ab', '_wrapped': 'xyz'})
    def rebind_in_subdef(a, b, *args, **kwargs):
        def func():
            args = 1,
            kwargs = {'z': 2}
            _wrapped(42, *args, **kwargs)
        _wrapped(*args, **kwargs)
        func()

    @tup('a, b, x, y, *, z', {'rebind_subdef_param': 'ab', '_wrapped': 'xyz'})
    def rebind_subdef_param(a, b, *args, **kwargs):
        def func(*args, **kwargs):
            _wrapped(42, *args, **kwargs)
        _wrapped(*args, **kwargs)
        func(2, z=3)

    @tup('a, b, *args, **kwargs',
         {'rebind_subdef_lambda_param': ['a', 'b', 'args', 'kwargs']})
    def rebind_subdef_lambda_param(a, b, *args, **kwargs):
        f = lambda *args, **kwargs: _wrapped(*args, **kwargs)
        f(1, 2, z=3)

    # @tup('a, b, x, y, *, z', {0: 'ab', '_wrapped': 'xyz'})
    # def nonlocal_already_executed(a, b, *args, **kwargs):
    #     def make_ret2(args, kwargs):
    #         def ret2():
    #             _wrapped(*args, **kwargs)
    #     make_ret2(args, kwargs)
    #     def ret1():
    #         nonlocal args, kwargs
    #         args = ()
    #         kwargs = {}

    def test_nonlocal_outside(self):
        x = _wrapped
        def l1(*args, **kwargs):
            nonlocal x
            x(*args, **kwargs)
        self._test(l1, 'x, y, *, z', {_wrapped: 'xyz'})

    def test_partial(self):
        def _wrapper(wrapped, a, *args, **kwargs):
            return wrapped(*args, **kwargs)
        func = partial(_wrapper, _wrapped)
        sig = specifiers.signature(func)
        self.assertSigsEqual(sig, support.s('a, x, y, *, z'))
        self.assertEqual(sig.sources, {
            'a': [_wrapper],
            'x': [_wrapped], 'y': [_wrapped], 'z': [_wrapped],
            '+depths': {func: 0, _wrapper: 1, _wrapped: 2}
        })
        support.assert_func_sig_coherent(
            func, check_return=False, check_invalid=False)

    @staticmethod
    @modifiers.kwoargs('wrapped')
    def _wrapped_kwoarg(a, wrapped, *args, **kwargs):
        return wrapped(*args, **kwargs) # pragma: no cover

    def test_partial_kwo(self):
        """When given keyword arguments, functools.partial only makes them
        defaults. The full signature is therefore not fully determined, since
        the user can replace wrapped and change the meaning of *args, **kwargs.

        The substitution could be made in good faith that the user wouldn't
        change the value of the parameter, but this would potentially cause
        confusing documentation where a function description says remaining
        arguments will be forwarded to the given function, while the signature
        in the documentation only shows the default target's arguments.
        """
        func = partial(AutoforwardsTests._wrapped_kwoarg, wrapped=_wrapped)
        expected = support.s('a, *args, wrapped=w, **kwargs',
                             locals={'w': _wrapped})
        self.assertSigsEqual(specifiers.signature(func), expected)


    _wrapped_attr = staticmethod(support.f('d, e, *, f'))

    @tup('a, d, e, *, f', {0: 'a', 'func': 'def'})
    def global_attribute(a, *args, **kwargs):
        AutoforwardsTests._wrapped_attr(*args, **kwargs)

    def test_instance_attribute(self):
        class A(object):
            def wrapped(self, x, y):
                pass
            def method(self, a, *args, **kwargs):
                self.wrapped(a, *args, **kwargs)
        a = A()
        self._test(a.method, 'a, y', {0: 'a', 'wrapped': 'y'})

    def test_multiple_method_calls(self):
        class A(object):
            def wrapped_1(self, x, y):
                pass
            def wrapped_2(self, x, y):
                pass
            def method(self, a, *args, **kwargs):
                self.wrapped_1(a, *args, **kwargs)
                self.wrapped_2(a, *args, **kwargs)
        self._test(A().method, 'a, y', _util.OrderedDict([
               (0, 'a'), ('method', 'a'),
               ('wrapped_1', 'y'), ('wrapped_2', 'y'),
               ('+depths', {'method': 0, 'wrapped_1': 1, 'wrapped_2': 1})]))

    @staticmethod
    @modifiers.kwoargs('b')
    def _deeparg_l1(l2, b, *args, **kwargs):
        l2(*args, **kwargs)

    @staticmethod
    @modifiers.kwoargs('c')
    def _deeparg_l2(l3, c, *args, **kwargs):
        l3(*args, **kwargs)

    @tup('x, y, *, a, b, c, z', {
            0: 'a', '_deeparg_l1': 'b', '_deeparg_l2': 'c', _wrapped: 'xyz'})
    @modifiers.kwoargs('a')
    def deeparg(a, *args, **kwargs):
        AutoforwardsTests._deeparg_l1(
            AutoforwardsTests._deeparg_l2, _wrapped,
            *args, **kwargs)

    @staticmethod
    @modifiers.kwoargs('l2')
    def _deeparg_kwo_l1(l2, b, *args, **kwargs):
        l2(*args, **kwargs)

    @staticmethod
    @modifiers.kwoargs('l3')
    def _deeparg_kwo_l2(l3, c, *args, **kwargs):
        l3(*args, **kwargs)

    @tup('a, b, c, x, y, *, z', {
        0: 'a', '_deeparg_kwo_l1': 'b', '_deeparg_kwo_l2': 'c', _wrapped: 'xyz'})
    def deeparg_kwo(a, *args, **kwargs):
        AutoforwardsTests._deeparg_kwo_l1(
            *args, l2=AutoforwardsTests._deeparg_kwo_l2, l3=_wrapped, **kwargs)

    @tup('a, x, y, *, z', {0: 'a', _wrapped: 'xyz'})
    def call_in_args(a, *args, **kwargs):
        func(_wrapped(*args, **kwargs))

    @tup('a, x, y, *, z', {0: 'a', _wrapped: 'xyz'})
    def call_in_kwargs(a, *args, **kwargs):
        func(kw=_wrapped(*args, **kwargs))

    @tup('a, x, y, *, z', {0: 'a', _wrapped: 'xyz'})
    def call_in_varargs(a, *args, **kwargs):
        func(*_wrapped(*args, **kwargs))

    @tup('a, x, y, *, z',
         {0: 'a', _wrapped: 'xyz', '+depths': ['call_in_varkwargs', '_wrapped']})
    def call_in_varkwargs(a, *args, **kwargs):
        func(**_wrapped(*args, **kwargs))

    def test_functools_wrapped(self):
        @wraps(_wrapped)
        def func(a, *args, **kwargs):
            _wrapped(1, *args, **kwargs)
        sig = specifiers.signature(func)
        self.assertSigsEqual(sig, support.s('a, y, *, z'))
        self.assertEqual(sig.sources, {
            '+depths': {func: 0, _wrapped: 1},
            'a': [func],
            'y': [_wrapped], 'z': [_wrapped]
        })
        support.assert_func_sig_coherent(
            func, check_return=False, check_invalid=False)

    def test_decorator_wraps(self):
        def decorator(function):
            @wraps(function)
            @modifiers.autokwoargs
            def _decorated(a, b=2, *args, **kwargs):
                function(1, *args, **kwargs)
            return _decorated
        func = decorator(_wrapped)
        sig = specifiers.signature(func)
        self.assertSigsEqual(sig, support.s('a, y, *, b=2, z'))
        self.assertEqual(sig.sources, {
            '+depths': {func: 0, _wrapped: 1},
            'a': [func], 'b': [func],
            'y': [_wrapped], 'z': [_wrapped]
        })
        support.assert_func_sig_coherent(
            func, check_return=False, check_invalid=False)

    @tup('a, b, *args, z',
         {'unknown_args': ['a', 'b', 'args'], '_wrapped': 'z'})
    def unknown_args(a, b, *args, **kwargs):
        args = (1, 2)
        return _wrapped(*args, **kwargs)

    # @tup('a, b, c, x, y, *, z', {0: 'ab', 'sub': 'c', '_wrapped': 'xyz'})
    # def use_subdef(a, b, *args, **kwargs):
    #     def sub(c, *args, **kwargs):
    #         _wrapped(*args, **kwargs)
    #     sub(1, *args, **kwargs)

    @tup('a, b, x=None, y=None, *, z=None', {0: 'ab', '_wrapped': 'xyz'})
    def pass_to_partial(a, b, *args, **kwargs):
        partial(_wrapped, *args, **kwargs)

    @tup('a, b, y=None', {0: 'ab', '_wrapped': 'y'})
    def pass_to_partial_with_args(a, b, *args, **kwargs):
        partial(_wrapped, a, *args, z=b, **kwargs)

    @tup('x, y, *, z', {'_wrapped': 'xyz'})
    def kwargs_passed_to_func_after(*args, **kwargs):
        _wrapped(*args, **kwargs)
        func(kwargs)

    @tup('x, y, *, z', {'_wrapped': 'xyz'})
    def args_passed_to_func(*args, **kwargs):
        func(args)
        _wrapped(*args, **kwargs)


not_callable = None


class UnresolvableAutoforwardsTests(Fixtures):
    def _test(self, func, ensure_incoherent=True):
        self.assertSigsEqual(
            specifiers.signature(func),
            signatures.signature(func))
        if ensure_incoherent:
            with self.assertRaises(AssertionError):
                support.assert_func_sig_coherent(
                    func, check_return=False, check_invalid=False)

    @tup(False)
    def missing_global(a, b, *p, **k):
        return doesntexist(*p, **k) # pyflakes: silence

    @tup()
    def builtin(a, b, *args, **kwargs):
        return iter(*args, **kwargs)

    def test_get_from_object(self):
        class A(object):
            def wrapped(self, x, y):
                pass
            def method(self, a, *p, **k):
                self.wrapped(a, *p, **k)
        method = _util.safe_get(A.__dict__['method'], object(), type(A))
        self._test(method, ensure_incoherent=False)

    def test_unset_attribute(self):
        class A(object):
            def method(self, a, *p, **k):
                self.wrapped(a, *p, **k)
        a = A()
        self._test(a.method, ensure_incoherent=False)

    @tup(False)
    def attribute_on_unset(*a, **k):
        doesntexist.method(*a, **k) # pyflakes: silence

    @tup()
    def constant(a, *p, **k):
        None(*p, **k)

    @tup()
    def not_callable(a, *p, **k):
        not_callable(*p, **k)

    def test_no_sig(self):
        obj = object()
        sig = support.s('a, *p, **k')
        def sig_replace(obj_):
            if obj is obj_:
                raise ValueError("no sig for obj")
            else:
                return sig
        def func(a, *p, **k):
            obj(*p, **k)
        with patch.multiple(_util.funcsigs, signature=sig_replace):
            self.assertSigsEqual(specifiers.signature(func), sig)

    @tup()
    def nonforwardable(*args):
        _wrapped(*args)

    def test_super_with_args(self):
        class Base(object):
            def method(self, x, y, z):
                pass
        class Derived(Base):
            def method(self, a, *args, **kwargs):
                super(Derived, self).method(*args, **kwargs)
        class MixIn(Base):
            def method(self, b, *args, **kwargs):
                super(MixIn, self).method(*args, **kwargs)
        class MixedIn(Derived, MixIn):
            pass
        for cls in [Derived, MixedIn]:
            with self.subTest(cls=cls.__name__):
                self._test(cls().method)

    @tup()
    def kwargs_passed_to_func(**kwargs):
        func(kwargs)
        _wrapped(**kwargs)

    @tup()
    def kwargs_method_called(**kwargs):
        kwargs.update({})
        _wrapped(**kwargs)

    @tup()
    def kwargs_item_added(**kwargs):
        kwargs['ham'] = 'spam'
        _wrapped(**kwargs)

    @tup(False)
    def kwargs_item_removed(**kwargs):
        del kwargs['ham']
        _wrapped(**kwargs)

    @tup()
    def kwargs_item_popped(**kwargs):
        kwargs.pop('ham', 'default')
        _wrapped(**kwargs)

    @tup(False)
    def kwargs_item_accessed(**kwargs):
        kwargs['ham']
        _wrapped(**kwargs)

    @tup()
    def rebind_subdef_nonlocal(a, b, *args, **kwargs):
        def func():
            nonlocal args, kwargs
            args = 2,
            kwargs = {'z': 3}
            _wrapped(42, *args, **kwargs)
        func()
        _wrapped(*args, **kwargs)

    @tup()
    def nonlocal_backchange(a, b, *args, **kwargs):
        def ret1():
            _wrapped(*args, **kwargs)
        def ret2():
            nonlocal args, kwargs
            args = ()
            kwargs = {}
        ret2()
        ret1()

    @tup()
    def nonlocal_deep(a, *args, **kwargs):
        def l1():
            def l2():
                nonlocal args, kwargs
                args = ()
                kwargs = {}
            l2()
        l1()
        _wrapped(*args, **kwargs)

    def test_missing_freevar(self):
        def make_closure():
            var = 1
            del var
            def func(a, *p, **k):
                var(*p, **k) # pyflakes: silence
            return func
        self._test(make_closure(), ensure_incoherent=False)

    def test_deleted(self):
        def makef(**kwargs):
            def func():
                _wrapped(**kwargs) # pyflakes: silence
            del kwargs
            return func
        self._test(makef, ensure_incoherent=False)

    def test_super_without_args(self):
        class Base:
            def method(self, x, y, *, z):
                pass
        class Derived(Base):
            def method(self, *args, a, **kwargs):
                super().method(*args, **kwargs)
        class MixIn(Base):
            def method(self, *args, b, **kwargs):
                super().method(*args, **kwargs)
        class MixedIn(Derived, MixIn):
            pass
        for cls in [Derived, MixedIn]:
            with self.subTest(cls=cls.__name__):
                self._test(cls().method)


class UnresolvableAutoforwardsWithSourcesTests(Fixtures):
    def _test(self, func, expected, expected_src):
        sig = specifiers.signature(func)
        self.assertSigsEqual(sig, support.s(expected))
        self.assertSourcesEqual(sig.sources, expected_src, func)
        with self.assertRaises(AssertionError):
            support.assert_func_sig_coherent(
                func, check_return=False, check_invalid=False)

    @tup('v, w, *a, **k', {0: 'vwak'})
    def double_starargs(v, w, *a, **k):
        _wrapped(*a, *a)

    @tup('v, w, *a, **k', {0: 'vwak'})
    def double_kwargs(v, w, *a, **k):
        _wrapped(**k, **w)
