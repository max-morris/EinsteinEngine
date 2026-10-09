#!/usr/bin/env python3

# Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
#
# This file is part of the Einstein Engine (EinsteinEngine).
#
# EinsteinEngine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EinsteinEngine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Regression test for scripts/fit_weights.py.

Feeds synthetic ee_microbench JSON (no benchmark run) and checks the profile.
Run: python3 scripts/test_fit_weights.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).with_name("fit_weights.py")
REPO = Path(__file__).resolve().parents[2]

TRANSCENDENTAL = [
    "sin_stream", "cos_stream", "tan_stream", "sinh_stream", "cosh_stream",
    "tanh_stream", "exp_stream", "log_stream", "erf_stream", "sqrt_stream",
    "cbrt_stream",
]


def row(group: str, name: str, ns: float, median: float | None = None) -> dict[str, Any]:
    return {
        "group": group,
        "name": name,
        "complexity_node": name,
        "ns_per_elem": ns,
        "ns_median": ns if median is None else median,
        "gops": 1.0,
        "ops_per_elem": 1,
        "checksum": 1.0,
    }


def base_rows() -> list[dict[str, Any]]:
    # add = 2 ns. div = 3.2 (1.6x -> 160). half = 1.0 (0.5x -> 100).
    # two_and_half = 5.0 (2.5x -> 250, not the old banker's 200).
    # transcendental sqrt = 10 (5x -> 500). pow sqrt = 4 must not win.
    rows = [
        row("arith", "add_stream", 2.0),
        row("arith", "sub_stream", 2.0),
        row("arith", "mul_stream", 2.0),
        row("arith", "div_stream", 3.2),
        row("arith", "neg_stream", 2.0),
        row("arith", "half_stream", 1.0),
        row("pow", "pow2_stream", 2.0),
        row("pow", "pown3_stream", 2.0),
        row("pow", "pown4_stream", 2.0),
        row("pow", "pown8_stream", 2.0),
        row("pow", "pown16_stream", 2.0),
        row("pow", "pown_neg1_stream", 2.0),
        row("pow", "pown_neg2_stream", 2.0),
        row("pow", "sqrt_stream", 4.0),
        row("pow", "cbrt_stream", 4.0),
        row("pow", "powvar_stream", 16.0),
        row("pow", "pow2p5_stream", 16.0),
        row("pow", "two_and_half_stream", 5.0),
        row("memory", "const_stream", 2.0),
        row("memory", "center_stream", 2.0),
        row("memory", "grid_add_stream", 2.0),
        row("memory", "stencil_x_stream", 80.0),
        row("memory", "stencil_y_stream", 80.0),
        row("memory", "stencil_z_stream", 80.0),
        row("branch", "cmp_lt_stream", 2.0),
        row("branch", "if_else_stream", 2.0),
        row("branch", "if_else_branch_stream", 2.0),
    ]
    for name in TRANSCENDENTAL:
        ns = 2.0
        if name == "sin_stream":
            ns = 16.0
        elif name == "erf_stream":
            ns = 50.0
        elif name == "cbrt_stream":
            ns = 24.0
        elif name == "sqrt_stream":
            ns = 10.0
        rows.append(row("transcendental", name, ns))
    return rows


def run(payload: dict[str, Any] | list[dict[str, Any]], extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    try:
        return subprocess.run(
            [sys.executable, str(SCRIPT), path, *(extra_args or [])],
            capture_output=True,
            text=True,
        )
    finally:
        Path(path).unlink()


def quantize(ns: float, add: float = 2.0) -> int:
    return max(100, round(100 * (ns / add)))


def main() -> int:
    failures: list[str] = []

    out = run({"results": base_rows()})
    if out.returncode != 0:
        failures.append(f"base fit failed:\n{out.stderr}\n{out.stdout}")
    else:
        if "print('current guestimate" in out.stdout or "print('current guestimate" in out.stderr:
            failures.append("fitter still prints a Python source line")
        try:
            doc = json.loads(out.stdout)
        except json.JSONDecodeError as exc:
            failures.append(f"stdout is not JSON: {exc}\n{out.stdout}")
            doc = None
        if isinstance(doc, dict):
            if doc.get("schema_version") != 1:
                failures.append(f"schema_version {doc.get('schema_version')!r}")
            weights = doc.get("weights")
            if not isinstance(weights, dict):
                failures.append("missing weights object")
            else:
                if weights.get("pow_default") != quantize(16.0):
                    failures.append(f"pow_default {weights.get('pow_default')}")
                if weights.get("pow_integer_floor") != 200:
                    failures.append("pow_integer_floor must stay the policy value 200")
                if (weights.get("stencil_center"), weights.get("stencil_x"), weights.get("stencil_yz")) != (1000, 4000, 10000):
                    failures.append(f"stencil policy changed: {weights.get('stencil_center')}, {weights.get('stencil_x')}, {weights.get('stencil_yz')}")
                if weights.get("symbol_grid") != 1000:
                    failures.append("symbol_grid must stay 1000")
                trans = weights.get("transcendental")
                if not isinstance(trans, dict):
                    failures.append("transcendental weights missing")
                else:
                    if trans.get("sin") != quantize(16.0):
                        failures.append(f"sin {trans.get('sin')}")
                    if trans.get("erf") != quantize(50.0):
                        failures.append(f"erf {trans.get('erf')}")
                    if trans.get("cbrt") != quantize(24.0):
                        failures.append(f"cbrt {trans.get('cbrt')}")
                    if trans.get("sqrt") != quantize(10.0):
                        failures.append(
                            f"sqrt {trans.get('sqrt')} must come from transcendental (500), not pow"
                        )
                for absent in ("div", "sub", "neg", "const", "cmp_lt", "if_else"):
                    if absent in weights:
                        failures.append(f"{absent} was stored as a weight")
            measurements = doc.get("measurements")
            if isinstance(measurements, dict):
                recorded = measurements.get("rows")
                if isinstance(recorded, list):
                    sqrt_rows = [r for r in recorded if isinstance(r, dict) and r.get("name") == "sqrt_stream"]
                    if len(sqrt_rows) != 2:
                        failures.append(f"expected both sqrt_stream rows, got {len(sqrt_rows)}")
                    applied = [
                        r.get("applied_to")
                        for r in sqrt_rows
                        if isinstance(r, dict) and r.get("group") == "transcendental"
                    ]
                    if applied != ["transcendental.sqrt"]:
                        failures.append(f"transcendental sqrt applied_to {applied}")
            sys.path.insert(0, str(REPO))
            from EinsteinEngine.generators.sympy_complexity import load_weights

            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(doc, f)
                fitted_path = f.name
            try:
                loaded = load_weights(fitted_path)
            finally:
                Path(fitted_path).unlink()
            if loaded.pow_default != quantize(16.0):
                failures.append("load_weights did not accept the fitted profile")

    # 1.6x must keep the fraction. 2.5x must stay 250. 0.5x clamps to 100.
    if not failures:
        weights_obj = json.loads(out.stdout)["weights"]
        # div is unmapped, so check via a mapped stand-in already covered.
        # Re-read recorded ratios through a dedicated transcendental stand-in:
        # half_stream is unmapped. Use the decisions only as a smoke check that
        # the quantizer formula is the one in the description.
        if "max(100, round(100 * adjusted_ns / adjusted_add))" not in json.loads(out.stdout)["description"]:
            failures.append("profile description does not state the quantizer")
        del weights_obj

    ratio_rows = base_rows()
    # Put the 1.6x, 2.5x, and 0.5x cases on mapped transcendental names so the
    # weights object shows the quantizer directly.
    for item in ratio_rows:
        if item["group"] == "transcendental" and item["name"] == "tan_stream":
            item["ns_per_elem"] = 3.2
            item["ns_median"] = 3.2
        if item["group"] == "transcendental" and item["name"] == "log_stream":
            item["ns_per_elem"] = 5.0
            item["ns_median"] = 5.0
        if item["group"] == "transcendental" and item["name"] == "cos_stream":
            item["ns_per_elem"] = 1.0
            item["ns_median"] = 1.0
    ratio_out = run({"results": ratio_rows})
    if ratio_out.returncode != 0:
        failures.append(f"ratio fit failed:\n{ratio_out.stderr}")
    else:
        trans = json.loads(ratio_out.stdout)["weights"]["transcendental"]
        if trans["tan"] != 160:
            failures.append(f"1.6x add must be 160, got {trans['tan']}")
        if trans["log"] != 250:
            failures.append(f"2.5x add must be 250, got {trans['log']}")
        if trans["cos"] != 100:
            failures.append(f"0.5x add must clamp to 100, got {trans['cos']}")

    identity_rows = base_rows()
    identity_rows.append(row("arith", "identity_stream", 1.0))
    for item in identity_rows:
        if item["group"] == "arith" and item["name"] == "add_stream":
            item["ns_per_elem"] = 3.0
            item["ns_median"] = 3.0
        if item["group"] == "transcendental" and item["name"] == "sin_stream":
            item["ns_per_elem"] = 9.0
            item["ns_median"] = 9.0
        if item["group"] == "transcendental" and item["name"] == "cos_stream":
            item["ns_per_elem"] = 1.2
            item["ns_median"] = 1.2
    ident = run({"results": identity_rows})
    if ident.returncode != 0:
        failures.append(f"identity fit failed:\n{ident.stderr}\n{ident.stdout}")
    else:
        doc = json.loads(ident.stdout)
        # adjusted add = 2, adjusted sin = 8, ratio 4 -> 400.
        # adjusted cos = 0 -> 100.
        if doc["weights"]["transcendental"]["sin"] != 400:
            failures.append(f"identity-adjusted sin {doc['weights']['transcendental']['sin']}")
        if doc["weights"]["transcendental"]["cos"] != 100:
            failures.append(f"floored cos {doc['weights']['transcendental']['cos']}")
        if doc["measurements"]["identity_subtracted"] is not True:
            failures.append("identity subtraction was not recorded")

    spread_rows = base_rows()
    for item in spread_rows:
        if item["name"] == "sin_stream":
            item["ns_median"] = item["ns_per_elem"] * 2
    spread = run({"results": spread_rows})
    if spread.returncode != 0:
        failures.append(f"spread fit should warn, not fail:\n{spread.stderr}")
    elif "sin_stream" not in spread.stderr or "WARNING" not in spread.stderr:
        failures.append(f"missing spread warning:\n{spread.stderr}")
    else:
        warnings = json.loads(spread.stdout)["measurements"]["warnings"]
        if not any("sin_stream" in w for w in warnings):
            failures.append(f"spread warning missing from profile: {warnings}")

    dup = run({"results": [row("pow", "sqrt_stream", 1.0), row("pow", "sqrt_stream", 2.0), row("arith", "add_stream", 2.0)]})
    if dup.returncode == 0:
        failures.append("duplicate (group, name) did not fail")
    elif "duplicate" not in dup.stderr and "duplicate" not in dup.stdout:
        failures.append(f"duplicate failure did not say so:\n{dup.stderr}{dup.stdout}")

    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "/nonexistent.json"],
        capture_output=True,
        text=True,
    )
    if missing.returncode == 0:
        failures.append("missing file did not fail")

    bad = run({"results": [{"name": "add_stream"}]})
    if bad.returncode == 0:
        failures.append("malformed row did not fail")

    if failures:
        print("FATAL:\n" + "\n".join(failures))
        return 1
    print("test_fit_weights: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
