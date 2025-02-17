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


from sigtools import _signatures, _util


def forged_signature(obj, auto=True, args=(), kwargs={}):
    """Retrieves the full signature of ``obj``, either by taking note of
    decorators from this module, or by performing automatic signature
    discovery.

    If ``auto`` is true, the signature will be automatically refined based on
    how ``*args`` and ``**kwargs`` are used throughout the function.

    If ``args`` and/or ``kwargs`` are specified, they are used by automatic
    signature discovery as arguments passed into the function. This is
    useful if the function calls something passed in as a parameter.

    You can use ``emulate=True`` as an argument to the decorators from this
    module if you wish them to work with `inspect.signature` or its
    `funcsigs<funcsigs:signature>` backport directly.

    ::

        >>> from sigtools import specifiers
        >>> import inspect
        >>> def inner(a, b):
        ...     return a + b
        ...
        >>> # Relying on automatic discovery
        >>> def outer(c, *args, **kwargs):
        ...     return c * inner(*args, **kwargs)
        >>> print(inspect.signature(outer))
        (c, *args, **kwargs)
        >>> print(specifiers.signature(outer, auto=False))
        (c, *args, **kwargs)
        >>> print(specifiers.signature(outer))
        (c, a, b)
        >>>
        >>> # Using a decorator from this module
        >>> @specifiers.forwards_to_function(inner)
        ... def outer(c, *args, **kwargs):
        ...     return c * inner(*args, **kwargs)
        ...
        >>> print(inspect.signature(outer))
        (c, *args, **kwargs)
        >>> print(specifiers.signature(outer), auto=False)
        (c, a, b)
        >>> print(specifiers.signature(outer))
        (c, a, b)
        >>>
        >>> # Using the emulate argument for compatibility with inspect
        >>> @specifiers.forwards_to_function(inner, emulate=True)
        ... def outer(c, *args, **kwargs):
        ...     return c * inner(*args, **kwargs)
        >>> print(inspect.signature(outer))
        (c, a, b)
        >>> print(specifiers.signature(outer), auto=False)
        (c, a, b)
        >>> print(specifiers.signature(outer))
        (c, a, b)

    :param bool auto: Enable automatic signature discovery.
    :param sequence args: Positional arguments passed to the function.
    :param mapping: Named arguments passed to the function.

    .. seealso:
        :ref:`autofwd limits`
    """
    subject = _util.get_introspectable(obj, af_hint=auto)
    forger = getattr(subject, '_sigtools__forger', None)
    if forger is not None:
        ret = forger(obj=subject)
        if ret is not None:
            return _signatures.UpgradedSignature._upgrade_with_warning(ret)
    if auto:
        try:
            subject._sigtools__autoforwards_hint
        except AttributeError:
            pass
        else:
            h = subject._sigtools__autoforwards_hint(subject)
            if h is not None:
                try:
                    ret = _autoforwards.autoforwards_ast(
                        *h, args=args, kwargs=kwargs)
                except _autoforwards.UnknownForwards:
                    pass
                else:
                    return _signatures.UpgradedSignature._upgrade_with_warning(ret)
            subject = _util.get_introspectable(subject, af_hint=False)
        try:
            ret = _signatures.UpgradedSignature._upgrade_with_warning(
                _autoforwards.autoforwards(subject, args, kwargs)
            )
        except _autoforwards.UnknownForwards:
            pass
        else:
            return _signatures.UpgradedSignature._upgrade_with_warning(ret)
    return _signatures.UpgradedSignature._upgrade_with_warning(_signatures.signature(obj))


from sigtools import _autoforwards
