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
    mk_symbol, sec, sech, sin, sinh, sqrt, sympify, tan, tanh
from EinsteinEngine.emit.code.common.code_tree import SympyExpr
from EinsteinEngine.emit.code.cpp_carpetx.cpp_carpetx_complexity import CppCarpetXComplexityVisitor
from EinsteinEngine.emit.code.cpp_carpetx.cpp_carpetx_named_roots import named_root
from EinsteinEngine.emit.code.cpp_carpetx.cpp_carpetx_visitor import CppVisitor
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef
from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend
from EinsteinEngine.frontend.dsl.f90.vanilla_f90_frontend import VanillaF90Module
from EinsteinEngine.generators.cpp_carpetx_generator import CppCarpetXGenerator
from EinsteinEngine.generators.sympy_complexity import SympyComplexityVisitor


def _visitor() -> SympyComplexityVisitor:
    return SympyComplexityVisitor(lambda _s: False)


def _carpetx_visitor() -> CppCarpetXComplexityVisitor:
    return CppCarpetXComplexityVisitor(lambda _s: False)


def _surcharge(v: SympyComplexityVisitor, name: str) -> int:
    return v._transcendental_surcharge(name)


def _general_power(v: SympyComplexityVisitor) -> int:
    w = v.weights
    return w.pow_default + w.symbol_local + w.atom


class TestTranscendentalSurcharge(unittest.TestCase):
    def test_transcendental_functions_cost_the_surcharge_plus_arguments(self) -> None:
        v = _visitor()
        x = mk_symbol("x")
        # A local symbol costs symbol_local. An unlisted function costs the same as sin.
        costs: list[int] = []
        for fn in (sin, cos, tan, cot, sec, csc, atan,
                   sinh, cosh, tanh, coth, sech, csch,
                   exp, log, erf):
            with self.subTest(fn=fn):
                cost = v.complexity(fn(x))
                self.assertEqual(cost, _surcharge(v, fn.__name__) + v.weights.symbol_local)
                costs.append(cost)
        # guestimates.json lists sin/cos/exp/log and omits the rest. Those
        # omitted calls still share sin's cost. A measured profile lists tan
        # and the hyperbolics at their own weights, so this equality stays
        # with the profile that actually omits them.
        if v.weights.transcendental.get("tan") is None:
            self.assertEqual(len(set(costs)), 1)

    def test_other_function_has_no_surcharge(self) -> None:
        v = _visitor()
        self.assertEqual(v.complexity(sy.Abs(mk_symbol("x"))), v.weights.symbol_local)

    def test_generic_model_charges_sqrt_and_cbrt_as_powers(self) -> None:
        v = _visitor()
        x = mk_symbol("x")
        # Backends such as F90 emit these as general powers, so the exponent is charged.
        for fn in (sqrt, cbrt):
            with self.subTest(fn=fn):
                self.assertEqual(v.complexity(fn(x)), _general_power(v))

    def test_carpetx_charges_sqrt_and_cbrt_the_surcharge_plus_base(self) -> None:
        v = _carpetx_visitor()
        x = mk_symbol("x")
        # These are x**(1/2) and x**(1/3). The exponent is not charged.
        for fn in (sqrt, cbrt):
            with self.subTest(fn=fn):
                self.assertEqual(v.complexity(fn(x)), _surcharge(v, fn.__name__) + v.weights.symbol_local)

    def test_other_fractional_powers_keep_the_general_cost(self) -> None:
        x = mk_symbol("x")
        # These are emitted as pow(), so the exponent is still charged.
        for v in (_visitor(), _carpetx_visitor()):
            for power in (sy.Rational(-1, 2), sy.Rational(3, 2), sy.Rational(2, 3)):
                with self.subTest(visitor=type(v).__name__, power=power):
                    self.assertEqual(v.complexity(x ** power), _general_power(v))

    def test_float_exponent_keeps_the_general_cost(self) -> None:
        x = mk_symbol("x")
        # x**0.5 is emitted as pow(), not sqrt(), so it gets no discount. A Float costs atom.
        for v in (_visitor(), _carpetx_visitor()):
            with self.subTest(visitor=type(v).__name__):
                self.assertEqual(v.complexity(x ** sympify(0.5)), _general_power(v))

    def test_carpetx_sqrt_and_cbrt_still_charge_a_grid_variable_base(self) -> None:
        v = CppCarpetXComplexityVisitor(lambda _s: True)
        u = mk_symbol("u")
        for fn in (sqrt, cbrt):
            with self.subTest(fn=fn):
                self.assertEqual(v.complexity(fn(u)), _surcharge(v, fn.__name__) + v.weights.symbol_grid)


class TestCarpetXDiscountMatchesEmittedCode(unittest.TestCase):
    def test_only_powers_emitted_as_named_calls_are_discounted(self) -> None:
        powers = (sy.Rational(1, 2), sy.Rational(1, 3), sy.Rational(-1, 2), sy.Rational(2, 3), sympify(0.5))
        thorn = ThornDef("ARR", "TST")
        u = thorn.decl("u", [])
        fn = thorn.create_function("f", ScheduleBin.Analysis)
        for i, power in enumerate(powers):
            fn.add_eqn(thorn.decl(f"o{i}", []), u ** power)
        thorn.bake()
        emitter = CppVisitor(CppCarpetXGenerator(thorn))
        cost_model = _carpetx_visitor()

        u_sym = mk_symbol("u")
        w = cost_model.weights
        for power in powers:
            with self.subTest(power=power):
                emitted: str = emitter.visit(SympyExpr(u_sym ** power))
                root = named_root(power.p, power.q) if isinstance(power, sy.Rational) else None
                cost = cost_model.complexity(u_sym ** power)
                if root is not None:
                    self.assertFalse(emitted.startswith("pow("), emitted)
                    self.assertEqual(cost, _surcharge(cost_model, root) + w.symbol_local)
                else:
                    self.assertTrue(emitted.startswith("pow("), emitted)
                    self.assertEqual(cost, w.pow_default + w.symbol_local + w.atom)


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
        # The frontend builds its own visitor from the active profile. A grid
        # variable costs symbol_grid, and CarpetX does not charge the exponent.
        active = _carpetx_visitor()
        self.assertEqual(
            self._sqrt_complexity(thorn, fn),
            _surcharge(active, "sqrt") + active.weights.symbol_grid,
        )

    def test_f90_module_uses_the_generic_cost_model(self) -> None:
        module = VanillaF90Module("m")
        fn = module.create_function("f")
        active = _visitor()
        self.assertEqual(
            self._sqrt_complexity(module, fn),
            active.weights.pow_default + active.weights.symbol_grid + active.weights.atom,
        )


if __name__ == "__main__":
    unittest.main()
