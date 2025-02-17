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
import inspect
import unittest
import warnings
from functools import partial

from sigtools._signatures import (
    sort_params, apply_params, IncompatibleSignatures, signature,
    UpgradedSignature, UpgradedParameter, _upgrade_parameters_with_warning,
    UpgradedAnnotation,
)
from sigtools.support import s, f
from sigtools._util import OrderedDict

from sigtools.tests.util import SignatureTests, Fixtures, python_doesnt_have_future_annotations, python_has_annotations


class SourceTests(Fixtures):
    def _test(self, sig_str):
        func = f(sig_str)
        sig = signature(func)
        srcs = dict((p, [func]) for p in sig.parameters)
        srcs['+depths'] = {func: 0}
        self.assertEqual(sig.sources, srcs)

    f1 = 'a, b, c',
    f2 = 'a, /, b, *, c',
    f3 = 'a, b, *args, c, **kwargs',
    f4 = '',


class PartialSigTests(Fixtures):
    def _test(self, obj, exp_sig, exp_src=None):
        sig = signature(obj)
        self.assertSigsEqual(sig, s(exp_sig))
        if exp_src is None:
            exp_src = dict((p, [obj.func]) for p in sig.parameters)
            exp_src['+depths'] = {obj: 0, obj.func: 1}
        self.assertEqual(sig.sources, exp_src)

    _func1 = f('a, b, c, *args, d, e, **kwargs')

    pos = partial(_func1, 1), 'b, c, *args, d, e, **kwargs'
    kwkw = partial(_func1, d=1), 'a, b, c, *args, e, d=1, **kwargs'

    kwposlast_1 = partial(_func1, c=1), 'a, b, *, d, e, c=1, **kwargs'
    kwposlast_2 = partial(_func1, b=1), 'a, *, d, e, c, b=1, **kwargs'

    def _kw_create(_func1):
        par = partial(_func1, f=1)
        src = dict((p, [_func1]) for p in ['a', 'b', 'c', 'args',
                                           'd', 'e', 'kwargs'])
        src['f'] = [par]
        src['+depths'] = {_func1: 1, par: 0}
        return par, 'a, b, c, *args, d, e, f=1, **kwargs', src
    kw_create = _kw_create(_func1)


class SortParamsTests(SignatureTests):
    def test_empty(self):
        sig = s('')
        self.assertEqual(sort_params(sig), ([], [], None, {}, None))

    def test_collectargs(self):
        sig = s('*args, **kwargs')
        ret = sort_params(sig)
        self.assertEqual(ret[0], [])
        self.assertEqual(ret[1], [])
        self.assertEqual(ret[2].name, 'args')
        self.assertEqual(ret[3], {})
        self.assertEqual(ret[4].name, 'kwargs')

    def test_pos(self):
        sig = s('a, b, /')
        ret = sort_params(sig)
        self.assertEqual(len(ret[0]), 2)
        self.assertEqual(ret[0][0].name, 'a')
        self.assertEqual(ret[0][1].name, 'b')
        self.assertEqual(ret[1:], ([], None, {}, None))

    def test_pok(self):
        sig = s('a, b')
        ret = sort_params(sig)
        self.assertEqual(len(ret[1]), 2)
        self.assertEqual(ret[1][0].name, 'a')
        self.assertEqual(ret[1][1].name, 'b')
        self.assertEqual(ret[0], [])
        self.assertEqual(ret[2:], (None, {}, None))

    def test_order_kwo(self):
        sig = s('*, one, two, three, four, five, six, seven')
        ret = sort_params(sig)
        self.assertEqual(ret[:3], ([], [], None))
        self.assertEqual(
            list(ret[3]),
            ['one', 'two', 'three', 'four', 'five', 'six', 'seven'])
        self.assertEqual(ret[4], None)


def p(sig_str):
    sig = s(sig_str)
    return next(iter(sig.parameters.values()))


class ApplyParamsTests(SignatureTests):
    def test_empty(self):
        self.assertSigsEqual(
            s(''),
            apply_params(s(''), [], [], None, {}, None)
            )

    def test_all(self):
        self.assertSigsEqual(
            s('one, two, /, three, four, *five, six, seven, **eight'),
            apply_params(
                s(''),
                [p('one, /'), p('two, /')],
                [p('three'), p('four')],
                p('*five'),
                OrderedDict(
                    (('six', p('*, six')), ('seven', p('*, seven')))),
                p('**eight')
                )
            )

    def test_roundtrip(self):
        sig_str = 'one, two, /, three, four, *five, six, seven, **eight'
        sig = s(sig_str)
        self.assertEqual(apply_params(s(''), *sort_params(sig)), sig)


class ExcTests(Fixtures):
    def _test(self, sig_str, sigs_strs, expected):
        sig = s(sig_str)
        sigs = [s(sig_str_) for sig_str_ in sigs_strs]
        exc = IncompatibleSignatures(sig, sigs)
        self.assertEqual(expected, str(exc))

    one = 'a, b', ['c, d'], '(c, d) (a, b)'
    two = 'a, b', ['c, d', 'e, f'], '(c, d) (e, f) (a, b)'


class UpgradedSignatureTests(SignatureTests):
    def test_upgrade_with_warning(self):
        sig = s("abc")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            UpgradedSignature._upgrade_with_warning(sig)

        downgraded_sig = self.downgrade_sig(sig)
        with self.assertWarns(DeprecationWarning):
            UpgradedSignature._upgrade_with_warning(downgraded_sig)

    def test_none_parameters(self):
        self.assertSigsEqual(
            UpgradedSignature(),
            s(""),
        )

    def test_replace_keep_parameters(self):
        sig = s("a, b, c")
        replaced = sig.replace(
            return_annotation=1,
            upgraded_return_annotation=UpgradedAnnotation.preevaluated(1)
        )
        self.assertSigsEqual(
            replaced,
            s("a, b, c", 1)
        )

    @unittest.skipIf(*python_doesnt_have_future_annotations)
    def test_evaluated(self):
        self.assertSigsEqual(
            s("one: a", "ret", globals={"a": 1, "ret": 2}, future_features=["annotations"]).evaluated(),
            s("one: 1", 2),
        )


class UpgradedParameterTests(SignatureTests):
    def test_upgrade_with_warning(self):
        param = UpgradedParameter(
            name="name",
            kind=UpgradedParameter.POSITIONAL_OR_KEYWORD,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _upgrade_parameters_with_warning([param])

        downgraded_param = inspect.Parameter("name", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        with self.assertWarns(DeprecationWarning):
            _upgrade_parameters_with_warning([param, downgraded_param])


class UpgradedAnnotationTests(SignatureTests):
    def test_upgrade_empty(self):
        self.assertEqual(
            UpgradedAnnotation.upgrade(UpgradedParameter.empty, None, 'param'),
            UpgradedAnnotation.preevaluated(UpgradedParameter.empty)
        )
        self.assertEqual(
            UpgradedAnnotation.upgrade(inspect.Parameter.empty, None, 'param'),
            UpgradedAnnotation.preevaluated(UpgradedParameter.empty)
        )

    @unittest.skipIf(*python_doesnt_have_future_annotations)
    def test_upgrade_postponed_annotation(self):
        func = f("", globals={"value": "the value"}, future_features=["annotations"])
        self.assertEqual(
            UpgradedAnnotation.upgrade('value', func, 'param'),
            UpgradedAnnotation.preevaluated("the value"),
        )

    @unittest.skipIf(python_has_annotations, "py version doesn't have preevaluated annotations")
    def test_upgrade_preevaluated_annotation(self):
        func = f("")
        self.assertEqual(
            UpgradedAnnotation.upgrade('the value', func, 'param'),
            UpgradedAnnotation.preevaluated("the value"),
        )

    def test_upgrade_without_function_warns(self):
        with self.assertWarns(DeprecationWarning):
            value = UpgradedAnnotation.upgrade('a value', None, 'param')
        self.assertEqual(value, UpgradedAnnotation.preevaluated(UpgradedParameter.empty))

    def test_not_equal(self):
        self.assertNotEqual(
            UpgradedAnnotation.preevaluated(UpgradedParameter.empty),
            UpgradedAnnotation.preevaluated("some value"),
            msg="empty is not equal to preevaluated value"
        )
        self.assertNotEqual(
            UpgradedAnnotation.preevaluated("some value"),
            UpgradedAnnotation.preevaluated("some other value"),
            msg="two different pre-evaluated values"
        )
        self.assertNotEqual(
            UpgradedAnnotation.preevaluated("some value"),
            "some value",
            msg="unrelated types"
        )

    def test_empty_repr(self):
        self.assertEqual(
            repr(UpgradedAnnotation.preevaluated(UpgradedParameter.empty)),
            "EmptyAnnotation",
        )
