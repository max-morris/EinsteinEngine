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

"""Tests for fit_weights: synthetic ee_microbench output in, a schema_version 1 profile out."""

import unittest
from typing import Any

from EinsteinEngine.generators.sympy_complexity import weights_from_profile
from EinsteinEngine.tuning.fit_weights import build_profile, load_results

TRANSCENDENTAL = ["sin", "cos", "tan", "sinh", "cosh", "tanh", "exp", "log", "erf", "sqrt", "cbrt"]


def row(group: str, name: str, ns: float, median: float | None = None) -> dict[str, Any]:
    return {"group": group, "name": name, "complexity_node": name, "ns_per_elem": ns,
            "ns_median": ns if median is None else median, "gops": 1.0, "ops_per_elem": 1, "checksum": 1.0}


def base_rows(**transcendental_ns: float) -> list[dict[str, Any]]:
    """add is 2 ns. The pow group's sqrt and cbrt (4 ns) must lose to the transcendental group's."""
    rows = [row("arith", name, ns) for name, ns in
            [("add_stream", 2.0), ("sub_stream", 2.0), ("mul_stream", 2.0), ("div_stream", 3.2), ("neg_stream", 2.0)]]
    rows += [row("pow", name, ns) for name, ns in
             [("pow2_stream", 2.0), ("pown16_stream", 2.0), ("pown_neg2_stream", 2.0), ("sqrt_stream", 4.0),
              ("cbrt_stream", 4.0), ("powvar_stream", 16.0)]]
    rows += [row("memory", "stencil_x_stream", 80.0), row("branch", "if_else_stream", 2.0)]
    defaults = {"sin": 16.0, "erf": 50.0, "cbrt": 24.0, "sqrt": 10.0}
    rows += [row("transcendental", f"{fn}_stream", transcendental_ns.get(fn, defaults.get(fn, 2.0)))
             for fn in TRANSCENDENTAL]
    return rows


def fit(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    return build_profile(load_results({"results": rows}))


class TestFitWeights(unittest.TestCase):
    def test_fitted_slots(self) -> None:
        profile, warnings = fit(base_rows())
        self.assertEqual(warnings, [])
        weights = profile["weights"]
        self.assertEqual(weights["pow_default"], 8)
        self.assertEqual({k: weights["transcendental"][k] for k in ("sin", "erf", "cbrt", "sqrt", "cos")},
                         {"sin": 8, "erf": 25, "cbrt": 12, "sqrt": 5, "cos": 1})
        # Policy slots keep the default model's values, and unmapped ops are not weights.
        self.assertEqual((weights["atom"], weights["symbol_grid"], weights["pow_integer_floor"]), (1, 10, 2))
        self.assertEqual((weights["stencil_center"], weights["stencil_x"], weights["stencil_yz"]), (10, 40, 100))
        self.assertFalse({"div", "sub", "neg", "const", "cmp_lt", "if_else"} & set(weights))
        applied = {(r["group"], r["name"]): r["applied_to"] for r in profile["measurements"]["rows"]}
        self.assertEqual(applied[("transcendental", "sqrt_stream")], "transcendental.sqrt")
        self.assertIsNone(applied[("pow", "sqrt_stream")])
        self.assertEqual(weights_from_profile(profile, "fit").pow_default, 8)

    def test_quantizer_rounds_to_whole_adds(self) -> None:
        # 1.6x add is 2, 2.5x is 3 (half up, not round-half-even's 2), 0.5x is 1, and 0.4x clamps to 1.
        trans = fit(base_rows(tan=3.2, log=5.0, cos=1.0, exp=0.8))[0]["weights"]["transcendental"]
        self.assertEqual((trans["tan"], trans["log"], trans["cos"], trans["exp"]), (2, 3, 1, 1))

    def test_identity_is_subtracted(self) -> None:
        rows = base_rows(sin=9.0, cos=1.2) + [row("arith", "identity_stream", 1.0)]
        next(r for r in rows if r["name"] == "add_stream")["ns_per_elem"] = 3.0
        next(r for r in rows if r["name"] == "add_stream")["ns_median"] = 3.0
        profile = fit(rows)[0]
        # Adjusted add 2 and sin 8 give 4; cos floors at 0 and clamps to 1.
        self.assertEqual(profile["weights"]["transcendental"]["sin"], 4)
        self.assertEqual(profile["weights"]["transcendental"]["cos"], 1)
        self.assertTrue(profile["measurements"]["identity_subtracted"])

    def test_identity_not_below_add_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "identity"):
            fit(base_rows() + [row("arith", "identity_stream", 2.0)])

    def test_spread_warns(self) -> None:
        rows = base_rows()
        next(r for r in rows if r["name"] == "sin_stream")["ns_median"] = 32.0
        profile, warnings = fit(rows)
        self.assertEqual(len(warnings), 1)
        self.assertIn("transcendental/sin_stream", warnings[0])
        self.assertEqual(profile["measurements"]["warnings"], warnings)

    def test_bad_input(self) -> None:
        for doc, message in [({"results": [row("pow", "x", 1.0), row("pow", "x", 2.0)]}, "duplicate"),
                             ({"results": [{"name": "add_stream"}]}, "group"),
                             ({"results": [row("arith", "add_stream", 2.0, median=1.0)]}, "below"),
                             ("text", "JSON object or a list")]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                load_results(doc)
        with self.assertRaisesRegex(ValueError, "missing benchmark arith/add_stream"):
            build_profile(load_results([row("pow", "powvar_stream", 2.0)]))


if __name__ == "__main__":
    unittest.main()
