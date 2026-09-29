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


class TestGuestimateProfile(unittest.TestCase):
    def test_transcendentals_take_effect(self) -> None:
        v = _visitor()
        self.assertEqual(v.complexity(sy.sin(sy.Symbol("x"))), 16)
        self.assertEqual(v.complexity(sy.exp(sy.Symbol("x"))), 16)

    def test_pow_formula(self) -> None:
        v = SympyComplexityVisitor(lambda s: False)
        x = sy.Symbol("x")
        self.assertEqual(v.complexity(x**2), 4)  # max(2, 1) + 2 args
        self.assertEqual(v.complexity(x**8), 5)  # max(2, 3) + 2 args
        self.assertEqual(v.complexity(x ** sy.Symbol("y")), 17)  # 15 + 2 args

    def test_symbols_and_stencil(self) -> None:
        v = _visitor()
        self.assertEqual(v.complexity(sy.Symbol("g")), 10)
        self.assertEqual(v.complexity(sy.Symbol("x")), 1)
        stencil = sy.Function("stencil")
        g = sy.Symbol("g")
        self.assertEqual(v.complexity(stencil(g, 0, 0, 0)), 10)
        self.assertEqual(v.complexity(stencil(g, 1, 0, 0)), 40)
        self.assertEqual(v.complexity(stencil(g, 0, 1, 0)), 100)
        self.assertEqual(v.complexity(stencil(g, 0, 0, 2)), 100)

    def test_unweighted_function_passthrough(self) -> None:
        v = SympyComplexityVisitor(lambda s: False)
        self.assertEqual(v.complexity(sy.tan(sy.Symbol("x"))), 1)


class TestProfileLoading(unittest.TestCase):
    def test_machine_profile_loads_and_applies(self) -> None:
        path = sc.WEIGHTS_DIR / "amd-ryzen-ai-9-hx-pro-370.json"
        weights = sc.load_weights(path)
        self.assertEqual(weights.pow_default, 8)
        self.assertEqual(weights.transcendental["erf"], 23)
        v = SympyComplexityVisitor(lambda s: False, weights=weights)
        self.assertEqual(v.complexity(sy.sin(sy.Symbol("x"))), 8)
        self.assertEqual(v.complexity(sy.erf(sy.Symbol("x"))), 24)

    def test_available_profiles(self) -> None:
        self.assertIn("guestimates", sc.available_profiles())

    def test_set_and_restore_weights(self) -> None:
        original = sc.get_weights()
        try:
            sc.set_weights(ComplexityWeights())
            self.assertEqual(sc.get_weights().atom, 1)
            self.assertEqual(sc.get_weights().transcendental, {})
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
        machine = sc.WEIGHTS_DIR / "amd-ryzen-ai-9-hx-pro-370.json"
        env = dict(os.environ)
        env["EE_COMPLEXITY_WEIGHTS"] = str(machine)
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
        self.assertEqual(out.stdout.strip(), "8")

    def test_bad_env_override_is_fatal(self) -> None:
        env = dict(os.environ)
        env["EE_COMPLEXITY_WEIGHTS"] = str(sc.WEIGHTS_DIR / "does-not-exist.json")
        env["PYTHONPATH"] = str(Path(sc.__file__).resolve().parents[2])
        proc = subprocess.run(
            [sys.executable, "-c", "import EinsteinEngine.generators.sympy_complexity"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
