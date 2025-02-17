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
`sigtools.signatures`: Signature object manipulation
----------------------------------------------------

The functions here are high-level operations that produce a signature from
other signature objects, as opposed to dealing with each parameter
individually. They are most notably used by the decorators from
`sigtools.specifiers` to compute combined signatures.

"""

from sigtools._signatures import (
    signature,
    IncompatibleSignatures,
    UpgradedSignature, UpgradedParameter, UpgradedAnnotation,
    sort_params, apply_params,
    merge, embed, mask, forwards
    )

__all__ = [
    'signature',
    'merge', 'embed', 'mask', 'forwards', 'IncompatibleSignatures',
    'UpgradedSignature', 'UpgradedParameter', 'UpgradedAnnotation',
    'sort_params', 'apply_params',
    ]
