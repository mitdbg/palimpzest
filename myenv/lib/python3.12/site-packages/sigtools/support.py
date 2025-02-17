# vim: set fileencoding=utf-8
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
`sigtools.support`: Utilities for use in interactive sessions and unit tests
----------------------------------------------------------------------------

"""

import __future__
import re
import sys
import itertools
from warnings import warn

from sigtools import _util, modifiers, signatures, specifiers

__all__ = [
    's', 'f', 'read_sig', 'func_code', 'make_func', 'func_from_sig',
    'make_up_callsigs', 'bind_callsig', 'sort_callsigs',
    'assert_func_sig_coherent',
    ]

try:
    zip = itertools.izip
except AttributeError:
    pass

re_paramname = re.compile(
    r'^'
    r'\s*([^:=]+)'      # param name
    r'\s*(?::(.+?))?'    # annotation
    r'\s*(?:=(.+))?'   # default value
    r'$')
re_posoarg = re.compile(r'^<(.*)>$')

def read_sig(sig_str, ret=_util.UNSET, *,
             use_modifiers_annotate=False,
             use_modifiers_posoargs=sys.version_info < (3, 8),
             use_modifiers_kwoargs=False,
    ):
    """Reads a string representation of a signature and returns a tuple
    `func_code` can understand."""

    names = []
    return_annotation = ret
    annotations = {}
    posoarg_n = []
    kwoarg_n = []
    params = []
    found_star = False
    varargs = None
    varkwargs = None
    chevron_index = None
    default_index = None
    for i, param in enumerate(sig_str.split(',')):
        if not param:
            continue
        arg, annotation, default = re_paramname.match(param).groups()
        insert = arg
        is_posoarg = re_posoarg.match(arg)
        if is_posoarg:
            name = arg = is_posoarg.group(1)
            insert = arg
            if use_modifiers_posoargs:
                posoarg_n.append(name)
            else:
                chevron_index = i
        else:
            name = arg.lstrip('*')
        if annotation:
            if use_modifiers_annotate:
                annotations[name.lstrip('*')] = annotation
            else:
                insert = f"{insert}: {annotation}"

        if default:
            if default_index is None:
                if found_star:
                    default_index = i - 1
                else:
                    default_index = i
            insert = f'{insert}={default}'

        if arg == '/':
            if use_modifiers_posoargs:
                posoarg_n.extend(names)
            else:
                params.append("/")
                chevron_index = None
        elif arg.startswith('*'):
            found_star = True
            if chevron_index is not None and not use_modifiers_posoargs:
                params.insert(chevron_index + 1, "/")
                chevron_index = None
            if name:
                params.append(insert)
                if arg.startswith('**'):
                    varkwargs = name
                else:
                    varargs = name
            elif not use_modifiers_kwoargs:
                params.append("*")
        elif found_star:
            if not use_modifiers_kwoargs:
                params.append(insert)
                names.append(name)
            else:
                kwoarg_n.append(arg)
                names.append(name)
                if not default and default_index is not None:
                    params.insert(default_index, insert)
                    default_index += 1
                else:
                    if params and params[-1].startswith('*'):
                        params.insert(-1, insert)
                    else:
                        params.append(insert)
        else:
            params.append(insert)
            names.append(name)
    if chevron_index is not None and not use_modifiers_posoargs:
        params.insert(chevron_index + 1, "/")
    if varargs:
        names.append(varargs)
    if varkwargs:
        names.append(varkwargs)
    return (
        names, return_annotation, annotations, posoarg_n, kwoarg_n,
        ', '.join(params), use_modifiers_annotate)

def func_code(names, return_annotation, annotations, posoarg_n,
              kwoarg_n, params, use_modifiers_annotate,
              pre='', name='func',
    ):
    """Formats the code to construct a function to `read_sig`'s design."""
    code = [pre]
    if use_modifiers_annotate and return_annotation is not _util.UNSET and annotations:
        annotation_args = ', '.join(
            f'{key}={value}'
            for key, value in annotations.items())
        code.append(f'@modifiers.annotate({return_annotation}, {annotation_args})')
    elif use_modifiers_annotate and return_annotation is not _util.UNSET:
        code.append(f'@modifiers.annotate({return_annotation})')
    elif annotations:
        annotation_args = ', '.join(
            f'{key}={value}'.format(key, value)
            for key, value in annotations.items())
        code.append(f'@modifiers.annotate({annotation_args})')
    if posoarg_n:
        posoarg_names = ', '.join(f"'{name}'" for name in posoarg_n)
        code.append(f'@modifiers.posoargs({posoarg_names})')
    if kwoarg_n:
        kwoargs_name = ', '.join(f"'{name}'" for name in kwoarg_n)
        code.append(f'@modifiers.kwoargs({kwoargs_name})')
    if not use_modifiers_annotate and return_annotation is not _util.UNSET:
        annotation_syntax = f" -> {return_annotation}"
    else:
        annotation_syntax = ""
    code.append(f'def {name}({params}){annotation_syntax}:')
    return_keyvalues = ', '.join(f'{name!r}: {name}' for name in names)
    code.append(f'    return {{{return_keyvalues}}}')
    return '\n'.join(code)

def make_func(source, globals=None, locals=None, name='func', future_features=()):
    """Executes the given code and returns the object named func from
    the resulting namespace."""
    flags = 0
    for feature in future_features:
        flags |= getattr(__future__, feature).compiler_flag
    code = compile(source, "<sigtools.support>", "exec", flags)
    func_globals = { "modifiers": modifiers, **(globals or {}) }
    exec(code, func_globals, locals)
    def_space = locals if locals is not None else func_globals
    return def_space[name]

def f(sig_str, ret=_util.UNSET, *, pre='', globals=None, locals=None, name='func', future_features=(), **kwargs):
    """Creates a dummy function that has the signature represented by
    ``sig_str`` and returns a tuple containing the arguments passed,
    in order.

    .. warning::
        The contents of the arguments are eventually passed to `exec`.
        Do not use with untrusted input.

    ::

        >>> from sigtools.support import f
        >>> import inspect
        >>> func = f('a, b=2, *args, c:"annotation", **kwargs')
        >>> print(inspect.signature(func))
        (a, b=2, *args, c:'annotation', **kwargs)
        >>> func(1, c=3)
        {'b': 2, 'a': 1, 'kwargs': {}, 'args': ()}
        >>> func(1, 2, 3, 4, c=5, d=6)
        {'b': 2, 'a': 1, 'kwargs': {'d': 6}, 'args': (3, 4)}
    """
    return make_func(
        func_code(
            *read_sig(sig_str, ret=ret, **kwargs),
            name=name,
            pre=pre,
        ),
        globals=globals,
        locals=locals,
        name=name,
        future_features=future_features,
    )

def s(*args, **kwargs):
    """Creates a signature from the given string representation of one.

    .. warning::
        The contents of the arguments are eventually passed to `exec`.
        Do not use with untrusted input.

    ::

        >>> from sigtools.support import s
        >>> sig = s('a, b=2, *args, c:"annotation", **kwargs')
        >>> sig
        <inspect.Signature object at 0x7f15e6055550>
        >>> print(sig)
        (a, b=2, *args, c:'annotation', **kwargs)
    """
    return specifiers.signature(f(*args, **kwargs))

def func_from_sig(sig):
    """Creates a dummy function from the given signature object

    .. warning::
        The contents of the arguments are eventually passed to `exec`.
        Do not use with untrusted input.
    """
    ret, sep, sig_str = str(sig).rpartition(' -> ')
    ret = ret if sep else _util.UNSET
    return f(sig_str[1:-1], ret)

def make_up_callsigs(sig, extra=2):
    """Figures out reasonably as many ways as possible to call a callable
    with the given signature."""
    pospars, pokpars, varargs, kwopars, varkwargs = signatures.sort_params(sig)

    names = [
        arg.name for arg in itertools.chain(
            pospars, pokpars, kwopars.values()
        )]
    for i in range(extra):
        names.append('__make_up_callsigs__extra_{0}'.format(i))

    args = [
        tuple(names[:i])
        for i in range(len(names) + 1)
        ]

    if varargs:
        names.append(varargs.name)
    if varkwargs:
        names.append(varkwargs.name)
    kwargs = [
        dict((name, name) for name in names_)
        for names_ in itertools.chain.from_iterable(
            itertools.combinations(names, i)
            for i in range(len(names) + 1)
            )
        ]

    ret = list(itertools.product(args, kwargs))
    return ret

def bind_callsig(sig, args, kwargs):
    """Returns a dict with each parameter name from ``sig`` mapped to
    values from ``args``, ``kwargs`` as if a function with ``sig``
    was called with ``(*args, **kwargs)``.

    Similar to `inspect.Signature.bind`."""
    assigned = {}

    varkwargs = next(
        (param for param in sig.parameters.values()
        if param.kind == param.VAR_KEYWORD), None)
    if varkwargs:
        assigned[varkwargs.name] = {}

    params = iter(sig.parameters.values())
    args_ = iter(args)
    i = 0
    for (i, posarg), param in zip(enumerate(args_, 1), params):
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            assigned[param.name] = posarg
        elif param.kind == param.VAR_POSITIONAL:
            assigned[param.name] = (posarg,) + tuple(args_)
            break
        else:
            raise TypeError('too many positional arguments')
    else:
        if args[:i] != args:
            raise TypeError('too many positional arguments')

    for key, value in kwargs.items():
        if key in sig.parameters:
            param = sig.parameters[key]
            if param.kind == param.POSITIONAL_ONLY:
                raise TypeError('{0!r} is positional-only'.format(key))
            elif param.kind in (param.POSITIONAL_OR_KEYWORD,
                                param.KEYWORD_ONLY):
                if key in assigned:
                    raise TypeError('{0!r} was specified twice'.format(key))
                assigned[key] = value
                continue
        if varkwargs:
            assigned[varkwargs.name][key] = value
        else:
            raise TypeError('unknown parameter {0!r}'.format(key))

    for param in sig.parameters.values():
        if param.name not in assigned:
            if param.kind == param.VAR_POSITIONAL:
                assigned[param.name] = ()
            elif param.default != param.empty:
                assigned[param.name] = param.default
            else:
                raise TypeError('omitted required parameter {0!r}'.format(
                    param.name))

    return assigned


DEBUG_STDLIB=False


def sort_callsigs(sig, callsigs):
    """Determines which ways to call ``sig`` in ``callsigs`` are valid or not.

    :returns:
        Two lists: ``(valid, invalid)``.

        ``valid``
            ``(args, kwargs, bound)`` in which ``bound`` is the dict
            returned by `bind_callsig`. It will be equal to the
            return value of a function with ``sig`` returned by
            `f`
        ``ìnvalid``
            ``(args, kwargs)``
    """
    valid = []
    invalid = []

    for args, kwargs in callsigs:
        try:
            bound = bind_callsig(sig, args, kwargs)
        except TypeError:
            if DEBUG_STDLIB:
                try:
                    sig.bind(*args, **kwargs)
                except TypeError:
                    pass
                else:
                    warn('{0}.bind(*{1}, **{2}) didn\'t raise TypeError'
                         .format(sig, args, kwargs))
            invalid.append((args, kwargs))
        else:
            valid.append((args, kwargs, bound))
            if DEBUG_STDLIB:
                try:
                    sig.bind(*args, **kwargs)
                except TypeError as e:
                    warn('{0}.bind(*{1}, **{2}) raised TypeError: {3}'
                         .format(sig, args, kwargs, e))

    return valid, invalid

def assert_func_sig_coherent(func, check_return=True, check_invalid=True):
    """Tests if a function is coherent with its signature.

    :param bool check_return: Check if the return value is correct
        (see `sort_callsigs`)
    :param bool check_invalid: Make sure call signatures invalid for the
        signature are also invalid for the passed callable.
    :raises: AssertionError
    """
    sig = specifiers.signature(func)

    sig_exceptions = (TypeError, ValueError) if hasattr(sys, 'pypy_version_info') else TypeError

    valid, invalid = sort_callsigs(sig, make_up_callsigs(sig, extra=2))

    for args, kwargs, expected_ret in valid:
        try:
            ret = func(*args, **kwargs)
        except sig_exceptions:
            raise AssertionError(
                '{0}{1} <- *{2}, **{3} raised TypeError'
                .format(_util.qualname(func), sig, args, kwargs))
        else:
            if check_return and expected_ret != ret:
                raise AssertionError(
                    '{0}{1} <- *{2}, **{3} returned {4} instead of {5}'
                    .format(_util.qualname(func), sig, args, kwargs,
                            ret, expected_ret))

    if check_invalid:
        for args, kwargs in invalid:
            try:
                func(*args, **kwargs)
            except sig_exceptions:
                pass
            else:
                raise AssertionError(
                    '{0}{1} <- *{2}, **{3} did not raise TypeError as expected'
                    .format(_util.qualname(func), sig, args, kwargs))

