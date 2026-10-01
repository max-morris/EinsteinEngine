#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
#
#  This file is part of the Einstein Engine (EinsteinEngine).
#
#  EinsteinEngine is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  EinsteinEngine is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The transcendental surcharge must apply to applied calls, not only to classes.

sqrt and cbrt are built as powers. The generic cost model charges them like any other
fractional power. The CarpetX one, which emits them as sqrt() and cbrt(), matches them
by exponent and charges them like a call.
"""

import unittest
from typing import Any

import sympy as sy

from EinsteinEngine.common.sympywrap import atan, cbrt, cos, cosh, cot, coth, csc, csch, erf, exp, log, \
    mk_symbol, sec, sech, sin, sinh, sqrt, tan, tanh
from EinsteinEngine.emit.code.cpp_carpetx.cpp_carpetx_complexity import CppCarpetXComplexityVisitor
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef
from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend
from EinsteinEngine.frontend.dsl.f90.vanilla_f90_frontend import VanillaF90Module
from EinsteinEngine.generators.sympy_complexity import SympyComplexityVisitor, TRANSCENDENTAL_COST


def _visitor() -> SympyComplexityVisitor:
    return SympyComplexityVisitor(lambda _s: False)


def _carpetx_visitor() -> CppCarpetXComplexityVisitor:
    return CppCarpetXComplexityVisitor(lambda _s: False)


class TestTranscendentalSurcharge(unittest.TestCase):
    def test_transcendental_functions_cost_the_surcharge_plus_arguments(self) -> None:
        v = _visitor()
        x = mk_symbol("x")
        # A local symbol costs 1.
        for fn in (sin, cos, tan, cot, sec, csc, atan,
                   sinh, cosh, tanh, coth, sech, csch,
                   exp, log, erf):
            with self.subTest(fn=fn):
                self.assertEqual(v.complexity(fn(x)), TRANSCENDENTAL_COST + 1)

    def test_other_function_has_no_surcharge(self) -> None:
        v = _visitor()
        self.assertEqual(v.complexity(sy.Abs(mk_symbol("x"))), 1)

    def test_generic_model_charges_sqrt_and_cbrt_as_powers(self) -> None:
        v = _visitor()
        x = mk_symbol("x")
        # Backends such as F90 emit these as general powers, so the exponent is charged.
        for fn in (sqrt, cbrt):
            with self.subTest(fn=fn):
                self.assertEqual(v.complexity(fn(x)), TRANSCENDENTAL_COST + 2)

    def test_carpetx_charges_sqrt_and_cbrt_the_surcharge_plus_base(self) -> None:
        v = _carpetx_visitor()
        x = mk_symbol("x")
        # These are x**(1/2) and x**(1/3). The exponent is not charged.
        for fn in (sqrt, cbrt):
            with self.subTest(fn=fn):
                self.assertEqual(v.complexity(fn(x)), TRANSCENDENTAL_COST + 1)

    def test_other_fractional_powers_keep_the_general_cost(self) -> None:
        x = mk_symbol("x")
        # These are emitted as pow(), so the exponent is still charged.
        for v in (_visitor(), _carpetx_visitor()):
            for power in (sy.Rational(-1, 2), sy.Rational(3, 2), sy.Rational(2, 3)):
                with self.subTest(visitor=type(v).__name__, power=power):
                    self.assertEqual(v.complexity(x ** power), TRANSCENDENTAL_COST + 2)


class TestFrontendCostModel(unittest.TestCase):
    def _sqrt_complexity(self, frontend: DslFrontend[Any, Any, Any], fn: DslFunctionFrontend[Any]) -> int:
        u = frontend.decl("u", [])
        v = frontend.decl("v", [])
        fn.add_eqn(v, sqrt(u))
        frontend.bake()
        return fn.eqn_complex.eqn_lists[0].complexity[mk_symbol("v")]

    def test_thorn_uses_the_carpetx_cost_model(self) -> None:
        thorn = ThornDef("ARR", "TST")
        fn = thorn.create_function("f", ScheduleBin.Analysis)
        # A grid variable costs 10, and the exponent is not charged.
        self.assertEqual(self._sqrt_complexity(thorn, fn), TRANSCENDENTAL_COST + 10)

    def test_f90_module_uses_the_generic_cost_model(self) -> None:
        module = VanillaF90Module("m")
        fn = module.create_function("f")
        self.assertEqual(self._sqrt_complexity(module, fn), TRANSCENDENTAL_COST + 10 + 1)


if __name__ == "__main__":
    unittest.main()
