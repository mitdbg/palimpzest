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
`sigtools.sphinxext`: Extension to make Sphinx use signature objects
--------------------------------------------------------------------

`sphinx.ext.autodoc` can only automatically discover the signatures of basic
callables. This extension makes it use `sigtools.specifiers.signature` on the
callable instead.

Enable it by appending ``'sigtools.sphinxext'`` to the ``extensions`` list
in your Sphinx ``conf.py``

"""

from sphinx.ext import autodoc

from sigtools import specifiers, _util


class _cls(object):
    def method(self):
        raise NotImplementedError
instancemethod = type(_cls().method)
del _cls


def process_signature(app, what, name, obj, options,
                      sig, return_annotation):
    try:
        parent, obj = fetch_dotted_name(name)
    except AttributeError:
        return sig, return_annotation
    if isinstance(obj, instancemethod): # python 2 unbound methods
        obj = obj.__func__
    if isinstance(parent, type) and callable(obj):
        obj = _util.safe_get(obj, object(), type(parent))
    try:
        sig = specifiers.signature(obj).evaluated()
    except (TypeError, ValueError):
        # inspect.signature raises ValueError if obj is callable but it can't
        # determine a signature, eg. built-in objects
        return sig, return_annotation
    ret_annot = sig.return_annotation
    if ret_annot != sig.empty:
        sret_annot = '{0!r}'.format(ret_annot)
        sig = sig.replace(return_annotation=sig.empty)
    else:
        sret_annot = ''
    return str(sig), sret_annot

def fetch_dotted_name(name):
    assert name
    post_import = name.split('.')[1:]
    while name:
        name = name.rpartition('.')[0]
        try:
            mod = __import__(name)
        except ImportError as exc:
            imp_exc = exc
        else:
            return fetch_deep_attr(mod, post_import)
    raise imp_exc

def fetch_deep_attr(obj, attrs):
    assert attrs
    for attr in attrs:
        parent, obj = obj, getattr(obj, attr)
    return parent, obj

class SignatureDocumenter(autodoc.FunctionDocumenter):
    objtype = 'signature'
    directivetype = 'function'
    priority = -1

    option_spec = {
        'index': autodoc.bool_option,
        }

    def __init__(self, *args, **kwargs):
        super(SignatureDocumenter, self).__init__(*args, **kwargs)
        self.options.noindex = not self.options.index

    def add_content(self, *args, no_docstring=True, **kwargs):
        return super(SignatureDocumenter, self).add_content(
            *args, **kwargs)

    def get_doc(self):
        return []

def setup(app):
    app.connect('autodoc-process-signature', process_signature)
    app.add_autodocumenter(SignatureDocumenter)
