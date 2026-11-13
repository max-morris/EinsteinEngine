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

"""Tests for the cost model's weights: the default model, profiles, and use_weights (sympy_complexity.py)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import sympy as sy

from EinsteinEngine.emit.code.cpp_carpetx.cpp_carpetx_complexity import CppCarpetXComplexityVisitor
from EinsteinEngine.generators import sympy_complexity as sc
from EinsteinEngine.generators.sympy_complexity import (
    ComplexityWeights,
    SympyComplexityVisitor,
)


def _visitor() -> SympyComplexityVisitor:
    return SympyComplexityVisitor(lambda s: s == sy.Symbol("g"))


def _weights(
    *,
    atom: int = 100,
    symbol_grid: int = 1000,
    symbol_local: int = 100,
    pow_default: int = 1500,
    pow_integer_floor: int = 200,
    stencil_center: int = 1000,
    stencil_x: int = 4000,
    stencil_yz: int = 10000,
    transcendental: dict[str, int] | None = None,
    transcendental_default: int = 0,
) -> ComplexityWeights:
    return ComplexityWeights(
        atom=atom,
        symbol_grid=symbol_grid,
        symbol_local=symbol_local,
        pow_default=pow_default,
        pow_integer_floor=pow_integer_floor,
        stencil_center=stencil_center,
        stencil_x=stencil_x,
        stencil_yz=stencil_yz,
        transcendental={} if transcendental is None else transcendental,
        transcendental_default=transcendental_default,
    )


class TestDefaultWeights(unittest.TestCase):
    def test_matches_the_model_on_master(self) -> None:
        # Complexities computed with master's hard-coded model, before weights were configurable.
        x, g = sy.Symbol("x"), sy.Symbol("g")
        stencil = sy.Function("stencil")
        exprs = [x + 2 * g, x ** 2, x ** 5, x ** 6, x ** 17, x ** -2, x ** sy.Rational(5, 2), sy.sqrt(x),
                 sy.cbrt(g), x ** sy.Rational(-1, 3), sy.sin(x) * sy.erf(g), sy.exp(x) + sy.log(x),
                 stencil(g, 0, 0, 0) + stencil(g, 1, 0, 0) + stencil(g, 0, -1, 2),
                 sy.Piecewise((x, x > 0), (g, True))]
        expected = {SympyComplexityVisitor: [12, 4, 4, 4, 6, 4, 17, 17, 26, 17, 41, 32, 150, 14],
                    CppCarpetXComplexityVisitor: [12, 4, 4, 4, 6, 4, 17, 16, 25, 17, 41, 32, 150, 14]}
        for visitor_type, costs in expected.items():
            visitor = visitor_type(lambda s: s == g)
            for expr, cost in zip(exprs, costs):
                with self.subTest(visitor=visitor_type.__name__, expr=expr):
                    self.assertEqual(visitor.complexity(expr), cost)

    def test_every_transcendental_costs_the_same(self) -> None:
        v = _visitor()
        x = sy.Symbol("x")
        for fn in (sy.sin, sy.exp, sy.tan, sy.cot, sy.erf):
            with self.subTest(fn=fn):
                self.assertEqual(v.complexity(fn(x)), 16)

    def test_pow_formula(self) -> None:
        v = SympyComplexityVisitor(lambda s: False)
        x = sy.Symbol("x")
        # Integer powers cost max(2, int(log2|p|)) plus the two atoms.
        self.assertEqual(v.complexity(x**2), 4)
        self.assertEqual(v.complexity(x**5), 4)  # int(log2(5)) = 2
        self.assertEqual(v.complexity(x**6), 4)
        self.assertEqual(v.complexity(x**8), 5)
        self.assertEqual(v.complexity(x ** sy.Symbol("y")), 17)  # pow_default 15 plus the two symbols
        self.assertEqual(v.complexity(sy.Pow(x, 1, evaluate=False)), 4)  # |p| <= 1 uses the floor
        self.assertEqual(v.complexity(sy.Pow(x, -1, evaluate=False)), 4)

    def test_integer_power_uses_atom_units(self) -> None:
        weights = _weights(atom=100, symbol_local=100, pow_integer_floor=100, pow_default=2100)
        v = SympyComplexityVisitor(lambda s: False, weights=weights)
        x = sy.Symbol("x")
        self.assertEqual(v.complexity(x**2), 300)  # max(100, 100) + 2 atoms
        self.assertEqual(v.complexity(x**5), 400)  # atom * int(log2(5)) = 200
        self.assertEqual(v.complexity(x**8), 500)  # max(100, 300) + 2 atoms

    def test_symbols_and_stencil(self) -> None:
        v = _visitor()
        g = sy.Symbol("g")
        stencil = sy.Function("stencil")
        self.assertEqual(v.complexity(g), 10)
        self.assertEqual(v.complexity(sy.Symbol("x")), 1)
        self.assertEqual(v.complexity(stencil(g, 0, 0, 0)), 10)
        self.assertEqual(v.complexity(stencil(g, 1, 0, 0)), 40)
        self.assertEqual(v.complexity(stencil(g, 0, 1, 0)), 100)
        self.assertEqual(v.complexity(stencil(g, 0, 0, 2)), 100)

    def test_unweighted_function_passthrough(self) -> None:
        v = SympyComplexityVisitor(lambda s: False)
        self.assertEqual(v.complexity(sy.Abs(sy.Symbol("x"))), 1)

    def test_generic_model_charges_half_and_third_as_powers(self) -> None:
        weights = _weights(
            atom=1,
            symbol_grid=10,
            symbol_local=1,
            pow_default=21,
            pow_integer_floor=2,
            stencil_center=10,
            stencil_x=40,
            stencil_yz=100,
            transcendental={"sqrt": 3, "cbrt": 4},
            transcendental_default=0,
        )
        v = SympyComplexityVisitor(lambda s: False, weights=weights)
        x = sy.Symbol("x")
        # F90 emits these as pow(), so the exponent is charged: 21 + 1 + 1.
        self.assertEqual(v.complexity(sy.sqrt(x)), 23)
        self.assertEqual(v.complexity(x ** sy.Rational(1, 2)), 23)
        self.assertEqual(v.complexity(sy.cbrt(x)), 23)
        self.assertEqual(v.complexity(x ** sy.Rational(1, 3)), 23)
        self.assertEqual(v.complexity(x ** sy.Rational(-1, 2)), 23)
        self.assertEqual(v.complexity(x ** sy.Rational(-1, 3)), 23)
        carpetx = CppCarpetXComplexityVisitor(lambda s: False, weights=weights)
        # CarpetX emits +1/2 and +1/3 as calls and does not charge the exponent.
        self.assertEqual(carpetx.complexity(sy.sqrt(x)), 4)  # sqrt 3 + base
        self.assertEqual(carpetx.complexity(x ** sy.Rational(1, 2)), 4)
        self.assertEqual(carpetx.complexity(sy.cbrt(x)), 5)  # cbrt 4 + base
        self.assertEqual(carpetx.complexity(x ** sy.Rational(1, 3)), 5)
        self.assertEqual(carpetx.complexity(x ** sy.Rational(-1, 2)), 23)
        self.assertEqual(carpetx.complexity(x ** sy.Rational(-1, 3)), 23)

    def test_fitted_profile_applies_sqrt_and_cbrt(self) -> None:
        weights = _weights(pow_default=2100, transcendental={"sin": 900, "tan": 1300, "sqrt": 300, "cbrt": 700})
        v = SympyComplexityVisitor(lambda s: False, weights=weights)
        x = sy.Symbol("x")
        sqrt_w = weights.transcendental["sqrt"]
        cbrt_w = weights.transcendental["cbrt"]
        self.assertEqual(weights.symbol_grid, 1000)
        self.assertEqual(weights.pow_integer_floor, 200)
        self.assertEqual(
            (weights.stencil_center, weights.stencil_x, weights.stencil_yz),
            (1000, 4000, 10000),
        )
        self.assertGreater(weights.pow_default, sqrt_w)
        general = weights.pow_default + weights.symbol_local + weights.atom
        self.assertEqual(v.complexity(sy.sqrt(x)), general)
        self.assertEqual(v.complexity(x ** sy.Rational(1, 2)), general)
        self.assertEqual(v.complexity(sy.cbrt(x)), general)
        self.assertEqual(v.complexity(x ** sy.Rational(-1, 3)), general)
        carpetx = CppCarpetXComplexityVisitor(lambda s: False, weights=weights)
        self.assertEqual(carpetx.complexity(sy.sqrt(x)), sqrt_w + weights.symbol_local)
        self.assertEqual(carpetx.complexity(x ** sy.Rational(1, 2)), sqrt_w + weights.symbol_local)
        self.assertEqual(carpetx.complexity(sy.cbrt(x)), cbrt_w + weights.symbol_local)
        self.assertEqual(carpetx.complexity(x ** sy.Rational(-1, 3)), general)
        # tan is measured. cot is not, so cot copies sin and does not rank below it.
        self.assertNotEqual(weights.transcendental["tan"], weights.transcendental["sin"])
        self.assertEqual(v.complexity(sy.tan(x)), weights.transcendental["tan"] + weights.symbol_local)
        self.assertNotIn("cot", weights.transcendental)
        self.assertEqual(v.complexity(sy.cot(x)), weights.transcendental["sin"] + weights.symbol_local)


class TestProfileLoading(unittest.TestCase):
    def test_fixture_profile_loads_and_applies(self) -> None:
        doc = {
            "schema_version": 1,
            "weights": {
                "atom": 100,
                "symbol_grid": 1000,
                "symbol_local": 100,
                "pow_default": 800,
                "pow_integer_floor": 200,
                "stencil_center": 1000,
                "stencil_x": 4000,
                "stencil_yz": 10000,
                "transcendental": {"sin": 700, "erf": 2300},
                "transcendental_default": 0,
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(doc, f)
            name = f.name
        try:
            weights = sc.load_weights(name)
        finally:
            Path(name).unlink()
        self.assertEqual(weights.pow_default, 800)
        self.assertEqual(weights.transcendental["erf"], 2300)
        v = SympyComplexityVisitor(lambda s: False, weights=weights)
        self.assertEqual(v.complexity(sy.sin(sy.Symbol("x"))), 800)
        self.assertEqual(v.complexity(sy.erf(sy.Symbol("x"))), 2400)
        # tan is not in this profile, so it copies sin rather than staying free.
        self.assertEqual(v.complexity(sy.tan(sy.Symbol("x"))), 800)

    def test_use_weights_restores(self) -> None:
        original = sc.get_weights()
        replacement = _weights(atom=250)
        with self.assertRaises(KeyError):
            with sc.use_weights(replacement):
                self.assertIs(sc.get_weights(), replacement)
                self.assertEqual(_visitor().complexity(sy.Integer(3)), 250)
                raise KeyError
        self.assertIs(sc.get_weights(), original)

    def test_missing_file_is_fatal(self) -> None:
        with self.assertRaises(FileNotFoundError):
            sc.load_weights(Path(tempfile.gettempdir()) / "does-not-exist.json")

    def test_bad_schema_is_fatal(self) -> None:
        bad_docs = [
            {"schema_version": 999, "weights": {}},
            {"schema_version": 1},
            {"schema_version": 1, "weights": {"atom": "one"}},
            {"schema_version": 1, "weights": {"atom": -1}},
            {"schema_version": 1, "weights": {"atom": 1}},
        ]
        for i, doc in enumerate(bad_docs):
            with self.subTest(i=i):
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".json", delete=False
                ) as f:
                    json.dump(doc, f)
                    name = f.name
                try:
                    with self.assertRaises((ValueError, KeyError)):
                        sc.load_weights(name)
                finally:
                    Path(name).unlink()

    def test_env_override_in_subprocess(self) -> None:
        doc = {
            "schema_version": 1,
            "weights": {
                "atom": 1,
                "symbol_grid": 10,
                "symbol_local": 1,
                "pow_default": 8,
                "pow_integer_floor": 2,
                "stencil_center": 10,
                "stencil_x": 40,
                "stencil_yz": 100,
                "transcendental": {"sin": 7},
                "transcendental_default": 0,
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(doc, f)
            name = f.name
        try:
            env = dict(os.environ)
            env["EE_COMPLEXITY_WEIGHTS"] = name
            env["PYTHONPATH"] = str(Path(sc.__file__).resolve().parents[2])
            out = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from EinsteinEngine.generators import sympy_complexity as sc; "
                    "import sympy as sy; "
                    "print(sc.SympyComplexityVisitor(lambda s: False).complexity(sy.sin(sy.Symbol('x'))))",
                ],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
        finally:
            Path(name).unlink()
        self.assertEqual(out.stdout.strip(), "8")

    def test_bad_env_override_fails_on_first_use(self) -> None:
        env = dict(os.environ)
        env["EE_COMPLEXITY_WEIGHTS"] = str(Path(tempfile.gettempdir()) / "does-not-exist.json")
        env["PYTHONPATH"] = str(Path(sc.__file__).resolve().parents[2])
        imported = subprocess.run(
            [sys.executable, "-c", "import EinsteinEngine.generators.sympy_complexity"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)
        used = subprocess.run(
            [
                sys.executable,
                "-c",
                "from EinsteinEngine.generators import sympy_complexity as sc; sc.get_weights()",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertNotEqual(used.returncode, 0)
        self.assertIn("does-not-exist.json", used.stderr)


if __name__ == "__main__":
    unittest.main()
