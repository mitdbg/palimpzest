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
`sigtools.modifiers`: Modify the effective signature of the decorated callable
------------------------------------------------------------------------------

The functions in this module can be used as decorators to mark and enforce some
parameters to be keyword-only (`kwoargs`) or annotate (`annotate`) them, just
like you can :ref:`using Python 3 syntax <def>`. You can also mark and enforce
parameters to be positional-only (`posoargs`). `autokwoargs` helps you quickly
make your parameters with default values become keyword-only.

"""

from functools import partial, update_wrapper

from sigtools import _util, _specifiers, _signatures

__all__ = ['annotate', 'kwoargs', 'autokwoargs', 'posoargs']


class _PokTranslator(_util.OverrideableDataDesc):
    __slots__ = ['__self__', 'func', 'posoarg_names', 'kwoarg_names', 'kwopos', '__signature__']

    def __new__(cls, func=None, posoargs=(), kwoargs=(), **kwargs):
        if func is None:
            return partial(_PokTranslator, posoargs=posoargs,
                           kwoargs=kwoargs, **kwargs)
        if posoargs or kwoargs:
            return super(_PokTranslator, cls).__new__(cls)
        return func

    def __init__(self, func, posoargs=(), kwoargs=(), **kwargs):
        update_wrapper(self, func)
        try:
            self.__self__ = func.__self__
        except AttributeError:
            pass
        try:
            del self._sigtools__forger
        except AttributeError:
            pass
        super(_PokTranslator, self).__init__(**kwargs)
        self.func = func
        self.posoarg_names = set(posoargs)
        self.kwoarg_names = set(kwoargs)
        if isinstance(func, _PokTranslator):
            self._merge_other(func)
        self._prepare()

    def _merge_other(self, other):
        self.func = other.func
        self.posoarg_names |= other.posoarg_names
        self.kwoarg_names |= other.kwoarg_names

        from sigtools import wrappers
        self.custom_getter = wrappers.Combination(
            self.custom_getter, other.custom_getter)

    def _prepare(self):
        intersection = self.posoarg_names & self.kwoarg_names
        if intersection:
            raise ValueError(
                'Parameters marked as both positional-only and keyword-only: '
                + ' '.join(repr(name) for name in intersection))
        to_use = self.posoarg_names | self.kwoarg_names

        sig = _specifiers.forged_signature(self.func, auto=False)
        params = []
        kwoparams = []
        kwopos = self.kwopos = []
        found_pok = found_kws = False
        for i, param in enumerate(sig.parameters.values()):
            if param.kind == param.POSITIONAL_OR_KEYWORD:
                if param.name in self.posoarg_names:
                    if found_pok:
                        raise ValueError(
                            '{0.name!r} was requested to become a positional-'
                            'only parameter, but comes after a regular '
                            'parameter'.format(param))
                    params.append(
                        param.replace(kind=param.POSITIONAL_ONLY))
                    to_use.remove(param.name)
                elif param.name in self.kwoarg_names:
                    kwoparams.append(
                        param.replace(kind=param.KEYWORD_ONLY))
                    kwopos.append((i, param))
                    to_use.remove(param.name)
                else:
                    found_pok = True
                    params.append(param)
            else: # not a POK param
                if param.name in to_use:
                    if param.kind == param.POSITIONAL_ONLY and param.name in self.posoarg_names:
                        to_use.remove(param.name)
                    elif param.kind == param.KEYWORD_ONLY and param.name in self.kwoarg_names:
                        to_use.remove(param.name)
                    else:
                        raise ValueError(
                            '{0.name!r} is not of kind POSITIONAL_OR_KEYWORD, but:'
                            ' {0.kind}'.format(param))
                if param.kind == param.VAR_KEYWORD:
                    found_kws = True
                    params.extend(kwoparams)
                params.append(param)
        if not found_kws:
            params.extend(kwoparams)
        if to_use:
            raise ValueError("Parameters not found: " + ' '.join(to_use))
        self.__signature__ = sig.replace(
            parameters=params,
            sources=_signatures.copy_sources(sig.sources, {self.func:self}))

    def _sigtools__autoforwards_hint(self, func):
        ast = _util.get_ast(self.func)
        if ast is None:
            return None
        sig = self.__signature__
        return self.func, ast, sig

    def __call__(self, *args, **kwargs):
        intersect = self.posoarg_names.intersection(kwargs)
        if intersect:
            raise TypeError(
                'Named arguments refer to positional-only parameters: {0}'
                .format(' '.join(repr(name) for name in intersect))
                )
        args = list(args) # we might need list.insert
        missing = []
        for pos, param in self.kwopos:
            if param.name in kwargs:
                if pos < len(args):
                    args.insert(pos, kwargs.pop(param.name))
            elif param.default == param.empty:
                missing.append(param.name)
            elif pos < len(args):
                args.insert(pos, param.default)
        if missing:
            raise TypeError('{0}() is missing the following required '
                            'keyword-only arguments: {1}'.format(
                            self.func.__name__, ', '.join(missing)))
        return self.func(*args, **kwargs)

    def parameters(self):
        return {
            'posoargs': self.posoarg_names,
            'kwoargs': self.kwoarg_names,
            }

    def __repr__(self):
        return (
            '<{0.func!r} with arg translation>'
            .format(self))

@_PokTranslator(kwoargs=('start',))
def kwoargs(start=None, *kwoarg_names):
    """Marks the given parameters as keyword-only, avoiding the use of
    python3 syntax.

    These two functions are equivalent::

        def py3_func(spam, *, ham, eggs='chicken'):
            return spam, ham, eggs

        @kwoargs('ham', 'eggs')
        def py23_func(spam, ham, eggs='chichen'):
            return spam, ham, eggs

    :param str start: If given and is the name of a parameter, it and all
        parameters after it are made keyword-only
    :param str kwoarg_names: Names of the parameters to convert

    :raises: `ValueError` if end or one of posoarg_names isn't in the
        decorated function's signature.
    """
    assert all(isinstance(s, str) for s in kwoarg_names), \
        "argument names must be strings; forgot to put () after @kwoargs?"
    if start is not None:
        return partial(_kwoargs_start, start, kwoarg_names)
    if not kwoarg_names:
        return _util.noop
    return partial(_PokTranslator, kwoargs=kwoarg_names)
# my syntax highlighter is broken """

def _kwoargs_start(start, _kwoargs, func, *args, **kwargs):
    kwoarg_names = set(_kwoargs)
    found = False
    sig = _specifiers.forged_signature(func, auto=False).parameters.values()
    for param in sig:
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            if found or param.name == start:
                found = True
                kwoarg_names.add(param.name)
        elif param.kind != param.POSITIONAL_ONLY:
            break # no more POKs now
    if not found:
        raise ValueError('{0!r} not found in {1.__name__}{2}'.format(
            start, func, sig))
    return _PokTranslator(
        func, kwoargs=kwoarg_names,
        get=partial(_kwoargs_start, start, _kwoargs))

@kwoargs('end')
def posoargs(end=None, *posoarg_names):
    """Marks the given parameters as positional-only.

    If the resulting function is passed any named arguments that references a
    positional parameter, `TypeError` will be raised.

        >>> from sigtools.modifiers import posoargs
        >>> @posoargs('ham')
        ... def func(ham, spam):
        ...     pass
        ...
        >>> func('ham', 'spam')
        >>> func('ham', spam='spam')
        >>> func(ham='ham', spam='spam')
        Traceback (most recent call last):
          File "<input>", line 1, in <module>
          File "./sigtools/modifiers.py", line 94, in __call__
            .format(' '.join(repr(name) for name in intersect))
        TypeError: Named arguments refer to positional-only parameters: 'ham'

    :param str end: If given and is the name of a parameter, it and all
        parameters leading to it are made positional-only.
    :param str posoarg_names: Names of the parameters to convert

    :raises: `ValueError` if end or one of posoarg_names isn't in the
        decorated function's signature.
    """
    assert all(isinstance(s, str) for s in posoarg_names), \
        "argument names must be strings"
    if end is not None:
        return partial(_posoargs_end, end, posoarg_names)
    if not posoarg_names:
        return _util.noop
    return partial(_PokTranslator, posoargs=posoarg_names)

def _posoargs_end(end, _posoargs, func, *args, **kwargs):
    posoarg_names = set(_posoargs)
    found = False
    sig = _specifiers.forged_signature(func, auto=False).parameters.values()
    for param in sig:
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            if not found:
                posoarg_names.add(param.name)
            if param.name == end:
                found = True
        elif param.kind != param.POSITIONAL_ONLY:
            break # no more POKs now
    if not found:
        raise ValueError('{0!r} not found in {1.__name__}{2}'.format(
            end, func, sig))
    return _PokTranslator(
        func, posoargs=posoarg_names,
        get=partial(_posoargs_end, end, _posoargs))

@kwoargs('exceptions')
def autokwoargs(func=None, exceptions=()):
    """Marks all arguments with default values as keyword-only.

    :param sequence exceptions: names of parameters not to convert

    ::

        >>> from sigtools.modifiers import autokwoargs
        >>> @autokwoargs(exceptions=['c'])
        ... def func(a, b, c=3, d=4, e=5):
        ...     pass
        ...
        >>> from inspect import signature
        >>> print(signature(func))
        (a, b, c=3, *, d=4, e=5)

    """
    if func is not None:
        if callable(func):
            return _autokwoargs(exceptions, func)
        else:
            raise ValueError("exceptions must be passed by name")
    else:
        return partial(_autokwoargs, exceptions)

def _autokwoargs(exceptions, func):
    sig = _specifiers.forged_signature(func, auto=False)
    args = []
    exceptions = set(exceptions)
    for param in sig.parameters.values():
        if (
                param.kind == param.POSITIONAL_OR_KEYWORD
                and param.default != param.empty
            ):
            try:
                exceptions.remove(param.name)
            except KeyError:
                args.append(param.name)
    if exceptions:
        raise ValueError(
            "parameters referred to by 'exceptions' not present: "
            + ' '.join(repr(name) for name in exceptions))
    return kwoargs(*args)(func)

class annotate(object):
    """Annotates a function, avoiding the use of python3 syntax

    These two functions are equivalent::

        def py3_func(spam: 'ham', eggs: 'chicken'=False) -> 'return':
            return spam, eggs

        @annotate('return', spam='ham', eggs='chicken')
        def py23_func(spam, eggs=False):
            return spam, eggs

    :param _annotate__return_annotation: The annotation to attach for return
        value
    :param annotations: The annotations to attach for each parameter
    :raises: `ValueError` if a parameter to be annotated does not exist
        on the function
    """

    def __init__(self, __return_annotation=_util.UNSET, **annotations):
        self.ret = __return_annotation
        self.annotations = annotations
        self.to_use = set(annotations)

    def __call__(self, obj):
        func = obj
        poks = []
        while isinstance(func, _PokTranslator):
            poks.append(func)
            func = func.func
        sig = _specifiers.forged_signature(func, auto=False)
        parameters = []
        to_use = self.to_use.copy()
        for name, parameter in sig.parameters.items():
            if name in self.annotations:
                annotation = self.annotations[name]
                upgraded_annotation = _signatures.UpgradedAnnotation.preevaluated(annotation)
                parameters.append(parameter.replace(
                    annotation=annotation,
                    upgraded_annotation=upgraded_annotation,
                ))
                to_use.remove(name)
            else:
                parameters.append(parameter)
        if to_use:
            raise ValueError(
                'the following parameters to be annotated '
                'were not found in {0}: {1}'
                .format(func.__name__, ', '.join(to_use)))
        if self.ret is _util.UNSET:
            sig = sig.replace(parameters=parameters)
        else:
            sig = sig.replace(parameters=parameters,
                              return_annotation=self.ret,
                              upgraded_return_annotation=_signatures.UpgradedAnnotation.preevaluated(self.ret))
        func.__signature__ = sig
        for pok in reversed(poks):
            pok._prepare()
        return obj

    def __repr__(self):
        return '{0}.annotate({1}{2})'.format(
            _util.qualname(type(self)),
            '' if self.ret is _util.UNSET else '{0!r}, '.format(self.ret),
            ', '.join('{0[0]}={0[1]!r}'.format(item)
                      for item in sorted(self.annotations.items())))


_finalized = True
