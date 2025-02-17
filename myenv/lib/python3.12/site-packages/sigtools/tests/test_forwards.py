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


from repeated_test import options

from sigtools.signatures import forwards
from sigtools.support import s

from sigtools.tests.util import FixturesWithFutureAnnotations


class ForwardsTests(FixturesWithFutureAnnotations):
    def _test(self, exp_sig, exp_src, outer, inner,
                    *, num_args=0, named_args=(),
                    hide_args=False, hide_kwargs=False,
                    use_varargs=True, use_varkwargs=True,
                    partial=False,
                    support_s, downgrade_sig
              ):
        outer_sig = support_s(outer, name='o')
        inner_sig = support_s(inner, name='i')

        with self.maybe_with_downgrade_and_ignore_warnings(
            downgrade_sig,
            self.downgrade_sig
        ) as maybe_downgrade:
            sig = forwards(
                        maybe_downgrade(outer_sig), maybe_downgrade(inner_sig),
                        num_args, *named_args,
                        hide_args=hide_args, hide_kwargs=hide_kwargs,
                        use_varargs=use_varargs, use_varkwargs=use_varkwargs,
                        partial=partial)
            self.assertSigsEqual(sig, s(exp_sig))
            if not downgrade_sig:
                self.assertSourcesEqual(sig.sources, {
                        'o': exp_src[0], 'i': exp_src[1],
                        '+depths': ['o', 'i']})

    a = 'a, b', ['a', 'b'], 'a, *args, **kwargs', 'b'

    pass_pos = 'a, c', ['a', 'c'], 'a, *p, **k', 'b, c', options(num_args=1)
    pass_kw = 'a, *, c', ['a', 'c'], 'a, *p, **k', 'b, c', options(named_args='b')

    dont_use_varargs = 'a, *p, b', ['ap', 'b'], 'a, *p, **k', 'b', options(use_varargs=False)

    through_kw = (
        'a, b, *, z', ['ab', 'z'], 'a, b, **k', 'x, y, *, z', options(num_args=2, hide_args=True))

    kwo = 'x, y, /, *, k', ['k', 'xy'], '*args, k', 'x, y, *, z', options(named_args='z')

    par = (
        'a, *, b, y=None, **z', ['ab', 'yz'], 'a, *p, b, **k', 'x, *, y, **z',
        options(num_args=1, partial=True))
