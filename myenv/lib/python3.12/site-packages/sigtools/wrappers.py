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

"""
`sigtools.wrappers`: Combine multiple functions
-----------------------------------------------

The functions here help you combine multiple functions into a new callable
which will automatically advertise the correct signature.

"""

from functools import partial, update_wrapper, wraps

from sigtools import _util, signatures, specifiers

class Combination(object):
    """Creates a callable that passes the first argument through each
    callable, using the result of each pass as the argument to the next
    """
    def __init__(self, *functions):
        funcs = self.functions = []
        for function in functions:
            if isinstance(function, Combination):
                funcs.extend(function.functions)
            else:
                funcs.append(function)
        specifiers.set_signature_forger(self, self.get_signature,
                                        emulate=False)

    def __call__(self, arg, *args, **kwargs):
        for function in self.functions:
            arg = function(arg, *args, **kwargs)
        return arg

    def get_signature(self, obj):
        return signatures.merge(
            signatures.signature(self),
            *(specifiers.signature(func) for func in self.functions)
            )

    def __repr__(self):
        return '{0.__module__}.{0.__name__}({1})'.format(
            type(self), ', '.join(repr(f) for f in self.functions)
            )


def decorator(func):
    """Turns a function into a decorator.

    The function received the decorated function as first argument.

    ::

        from sigtools import wrappers

        @wrappers.decorator
        def my_decorator(func, *args, deco_param=False, **kwargs):
            ... # Do stuff with deco_param
            return func(*args, **kwargs)

        @my_decorator
        def my_function(func_param):
            ...

        my_function('value for func_param', deco_param=True)

        from sigtools import specifiers
        print(specifiers.signature(my_function))
        # (func_param, *, deco_param=False)

    Unlike `wrapper_decorator`, ``decorator`` does not require you to specify
    how your function uses ``*args, **kwargs`` and lets automatic signature
    discovery figure it out.

    .. note:: Signature reporting will not work in interactive sessions, as per
       :ref:`autofwd limits`.
    """
    @wraps(func)
    def _decorate(decorated):
        return _SimpleWrapped(func, decorated)
    return _decorate


class _SimpleWrapped(object):
    def __init__(self, wrapper, wrapped):
        update_wrapper(self, wrapped)
        self.func = partial(wrapper, wrapped)
        self.wrapper = wrapper
        self._sigtools__wrappers = wrapper,
        self.__wrapped__ = wrapped
        try:
            del self._sigtools__forger
        except AttributeError:
            pass
        try:
            del self.__signature__
        except AttributeError:
            pass

    __signature__ = specifiers.as_forged

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __get__(self, instance, owner):
        return type(self)(
            self.wrapper,
            _util.safe_get(self.__wrapped__, instance, owner))

    def __repr__(self):
        return '<{0!r} wrapped with {1!r}>'.format(
                self.__wrapped__, self.wrapper)


@specifiers.forwards_to_function(specifiers.forwards, 2)
def wrapper_decorator(*args, **kwargs):
    """Turns a function into a decorator that wraps callables with
    that function.

    Consult `.signatures.forwards`'s documentation for :ref:`help picking the correct values for the parameters <forwards-pick>`.

    The wrapped function is passed as first argument to the wrapper.

    As an example, here we create a ``@print_call`` decorator which wraps the
    decorated function and prints a line everytime the function is called::

        >>> from sigtools import modifiers, wrappers
        >>> @wrappers.wrapper_decorator
        ... @modifiers.autokwoargs
        ... def print_call(func, _show_return=True, *args, **kwargs):
        ...     print('Calling {0.__name__}(*{1}, **{2})'.format(func, args, kwargs))
        ...     ret = func(*args, **kwargs)
        ...     if _show_return:
        ...         print('Return: {0!r}'.format(ret))
        ...     return ret
        ...
        >>> print_call
        <decorate with <<function print_call at 0x7f28d721a950> with signature print_cal
        l(func, *args, _show_return=True, **kwargs)>>
        >>> @print_call
        ... def as_list(obj):
        ...     return [obj]
        ...
        >>> as_list
        <<function as_list at 0x7f28d721ad40> decorated with <<function print_call at 0x
        7f28d721a950> with signature print_call(func, *args, _show_return=True, **kwargs
        )>>
        >>> from inspect import signature
        >>> print(signature(as_list))
        (obj, *, _show_return=True)
        >>> as_list('ham')
        Calling as_list(*('ham',), **{})
        Return: ['ham']
        ['ham']
        >>> as_list('spam', _show_return=False)
        Calling as_list(*('spam',), **{})
        ['spam']

    """
    if not kwargs and len(args) == 1 and callable(args[0]):
        return _WrapperDecorator((), {}, args[0])
    return partial(_WrapperDecorator, args, kwargs)

class _WrapperDecorator(object):
    def __init__(self, f_args, f_kwargs, wrapper):
        self.f_args = f_args
        self.f_kwargs = f_kwargs
        self.wrapper = wrapper

    def wrap(self, wrapped):
        return _Wrapped(self, self.wrapper, wrapped)

    __call__ = wrap

    def __repr__(self):
        return '<wrap with {0!r}>'.format(self.wrapper)

class _Wrapped(object):
    def __init__(self, deco, wrapper, wrapped):
        func = partial(wrapper, wrapped)
        update_wrapper(self, wrapped)
        self.func = func
        self.wrapper = wrapper
        self._sigtools__wrappers = wrapper,
        self.decorator = deco
        self.__wrapped__ = wrapped
        try:
            del self._sigtools__forger
        except AttributeError:
            pass
        try:
            del self.__signature__
        except AttributeError:
            pass

    __signature__ = specifiers.as_forged

    def _sigtools__forger(self, obj):
        return specifiers.forwards(
            self.func, self.__wrapped__,
            *self.decorator.f_args, **self.decorator.f_kwargs)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __get__(self, instance, owner):
        return type(self)(
            self.decorator, self.wrapper,
            _util.safe_get(self.__wrapped__, instance, owner))

    def __repr__(self):
        return '<{0!r} wrapped with {1!r}>'.format(
                self.__wrapped__, self.wrapper)

def wrappers(obj):
    """For introspection purposes, returns an iterable that yields each
    wrapping function of obj(as done through `wrapper_decorator`, outermost
    wrapper first.

    Continuing from the `wrapper_decorator` example::

        >>> list(wrappers.wrappers(as_list))
        [<<function print_call at 0x7f28d721a950> with signature print_call(func, *args,
         _show_return=True, **kwargs)>]

    """
    while True:
        try:
            wrappers = obj._sigtools__wrappers
        except AttributeError:
            return
        for wrapper in wrappers:
            yield wrapper
        obj = obj.__wrapped__
