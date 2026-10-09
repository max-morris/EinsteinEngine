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

"""Tests for JSON-driven complexity weights (sympy_complexity.py)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import sympy as sy

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


PROFILES_DIR = Path(__file__).resolve().parents[1] / "microbenchmarks" / "profiles"


class TestGuestimateProfile(unittest.TestCase):
    def test_transcendentals_take_effect(self) -> None:
        v = _visitor()
        self.assertEqual(v.complexity(sy.sin(sy.Symbol("x"))), 1600)
        self.assertEqual(v.complexity(sy.exp(sy.Symbol("x"))), 1600)

    def test_transcendental_surcharge_changes_rank_versus_legacy_zero(self) -> None:
        # The pre-branch predicate compared a call instance with the function
        # classes and added 0. guestimates.json adds 1500. A full CarpetX Z4c
        # regeneration was not run; this pins the cost delta that regeneration
        # would see.
        v = SympyComplexityVisitor(lambda s: False)
        x = sy.Symbol("x")
        adds = x + x + x + x + x + x
        sine = sy.sin(x)
        self.assertEqual(v.complexity(sine) - v.complexity(x), 1500)
        self.assertLess(v.complexity(adds), v.complexity(sine))
        legacy_sin = v.complexity(x)
        self.assertLess(legacy_sin, v.complexity(adds))

    def test_pow_formula(self) -> None:
        v = SympyComplexityVisitor(lambda s: False)
        x = sy.Symbol("x")
        # Guestimate scale: floor 200, atom 100. Integer powers truncate log2.
        self.assertEqual(v.complexity(x**2), 400)  # max(200, 100) + 2 atoms
        self.assertEqual(v.complexity(x**5), 400)  # int(log2(5))=2, not round(232)
        self.assertEqual(v.complexity(x**6), 400)  # int(log2(6))=2, not round(258)
        self.assertEqual(v.complexity(x**8), 500)  # max(200, 300) + 2 atoms
        self.assertEqual(v.complexity(x ** sy.Symbol("y")), 1700)  # 1500 + 2 atoms
        one = sy.Pow(x, 1, evaluate=False)
        self.assertEqual(v.complexity(one), 400)  # |p|<=1 uses the floor
        minus_one = sy.Pow(x, -1, evaluate=False)
        self.assertEqual(v.complexity(minus_one), 400)

    def test_integer_power_uses_atom_units(self) -> None:
        weights = _weights(atom=100, symbol_local=100, pow_integer_floor=100, pow_default=2100)
        v = SympyComplexityVisitor(lambda s: False, weights=weights)
        x = sy.Symbol("x")
        self.assertEqual(v.complexity(x**2), 300)  # max(100, 100) + 2 atoms
        self.assertEqual(v.complexity(x**5), 400)  # atom * int(log2(5)) = 200
        self.assertEqual(v.complexity(x**8), 500)  # max(100, 300) + 2 atoms

    def test_symbols_and_stencil(self) -> None:
        v = _visitor()
        self.assertEqual(v.complexity(sy.Symbol("g")), 1000)
        self.assertEqual(v.complexity(sy.Symbol("x")), 100)
        stencil = sy.Function("stencil")
        g = sy.Symbol("g")
        self.assertEqual(v.complexity(stencil(g, 0, 0, 0)), 1000)
        self.assertEqual(v.complexity(stencil(g, 1, 0, 0)), 4000)
        self.assertEqual(v.complexity(stencil(g, 0, 1, 0)), 10000)
        self.assertEqual(v.complexity(stencil(g, 0, 0, 2)), 10000)

    def test_unweighted_function_passthrough(self) -> None:
        v = SympyComplexityVisitor(lambda s: False)
        self.assertEqual(v.complexity(sy.tan(sy.Symbol("x"))), 100)

    def test_half_and_third_powers_use_sqrt_cbrt_weights(self) -> None:
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
        # Operation weight plus the base and the exponent.
        self.assertEqual(v.complexity(sy.sqrt(x)), 5)
        self.assertEqual(v.complexity(x ** sy.Rational(1, 2)), 5)
        self.assertEqual(v.complexity(sy.cbrt(x)), 6)
        self.assertEqual(v.complexity(x ** sy.Rational(1, 3)), 6)
        self.assertEqual(v.complexity(x ** sy.Rational(-1, 2)), 23)
        self.assertEqual(v.complexity(x ** sy.Rational(-1, 3)), 23)

    def test_a100_profile_applies_sqrt_and_cbrt(self) -> None:
        weights = sc.load_weights(PROFILES_DIR / "nvidia-a100-80gb-pcie.json")
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
        self.assertEqual(v.complexity(sy.sqrt(x)), sqrt_w + weights.symbol_local + weights.atom)
        self.assertEqual(v.complexity(x ** sy.Rational(1, 2)), sqrt_w + weights.symbol_local + weights.atom)
        self.assertEqual(v.complexity(sy.cbrt(x)), cbrt_w + weights.symbol_local + weights.atom)
        self.assertEqual(
            v.complexity(x ** sy.Rational(-1, 3)),
            weights.pow_default + weights.symbol_local + weights.atom,
        )

    def test_package_ships_only_the_default_profile(self) -> None:
        names = sorted(p.name for p in sc.WEIGHTS_DIR.glob("*.json"))
        self.assertEqual(names, ["guestimates.json"])
        self.assertTrue((PROFILES_DIR / "nvidia-a100-80gb-pcie.json").is_file())
        self.assertFalse((sc.WEIGHTS_DIR / "nvidia-a100-80gb-pcie.json").exists())

    def test_every_profile_smallest_positive_weight_is_100(self) -> None:
        paths = sorted(sc.WEIGHTS_DIR.glob("*.json")) + sorted(PROFILES_DIR.glob("*.json"))
        self.assertGreaterEqual(len(paths), 5)
        for path in paths:
            with self.subTest(path.name):
                doc = json.loads(path.read_text(encoding="utf-8"))
                weights = doc["weights"]
                positives: list[int] = []

                def walk(node: object) -> None:
                    if isinstance(node, dict):
                        for value in node.values():
                            walk(value)
                    elif isinstance(node, int) and not isinstance(node, bool) and node > 0:
                        positives.append(node)

                walk(weights)
                self.assertEqual(min(positives), 100)
                self.assertEqual(weights["transcendental_default"], 0)


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

    def test_set_and_restore_weights(self) -> None:
        original = sc.get_weights()
        replacement = _weights(atom=250)
        try:
            sc.set_weights(replacement)
            self.assertIs(sc.get_weights(), replacement)
            self.assertEqual(sc.get_weights().atom, 250)
        finally:
            sc.set_weights(original)
        self.assertIs(sc.get_weights(), original)

    def test_missing_file_is_fatal(self) -> None:
        with self.assertRaises(FileNotFoundError):
            sc.load_weights(sc.WEIGHTS_DIR / "does-not-exist.json")

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
                "atom": 100,
                "symbol_grid": 1000,
                "symbol_local": 100,
                "pow_default": 800,
                "pow_integer_floor": 200,
                "stencil_center": 1000,
                "stencil_x": 4000,
                "stencil_yz": 10000,
                "transcendental": {"sin": 700},
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
        self.assertEqual(out.stdout.strip(), "800")

    def test_bad_env_override_fails_on_first_use(self) -> None:
        env = dict(os.environ)
        env["EE_COMPLEXITY_WEIGHTS"] = str(sc.WEIGHTS_DIR / "does-not-exist.json")
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
