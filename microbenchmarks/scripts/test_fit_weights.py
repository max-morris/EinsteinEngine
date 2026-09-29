#!/usr/bin/env python3
# Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression test for scripts/fit_weights.py.

Feeds synthetic ee_microbench JSON (no benchmark run needed) and checks the
fitted weights follow round(ns_op / ns_add). Fatal on any mismatch.
Run: python3 scripts/test_fit_weights.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

NAMES = [
    "add_stream", "sub_stream", "mul_stream", "div_stream", "neg_stream",
    "pow2_stream", "pown3_stream", "pown4_stream", "pown8_stream", "pown16_stream",
    "sqrt_stream", "cbrt_stream", "powvar_stream", "pow2p5_stream",
    "sin_stream", "cos_stream", "tan_stream", "sinh_stream", "cosh_stream",
    "tanh_stream", "exp_stream", "log_stream", "erf_stream",
    "const_stream", "center_stream", "grid_add_stream",
    "stencil_x_stream", "stencil_y_stream", "stencil_z_stream",
    "cmp_lt_stream", "if_else_stream", "if_else_branch_stream",
]


def main() -> int:
    script = Path(__file__).with_name("fit_weights.py")
    # Synthetic timings: add=2.0ns, sin=16.0 (->8), erf=50.0 (->25), rest 2.0 (->1).
    timings = {n: 2.0 for n in NAMES}
    timings.update({"sin_stream": 16.0, "erf_stream": 50.0, "cbrt_stream": 24.0})
    payload = {
        "results": [
            {"group": "g", "name": n, "complexity_node": n, "ns_per_elem": v,
             "ns_median": v, "gops": 1.0, "ops_per_elem": 1, "checksum": 1.0}
            for n, v in timings.items()
        ]
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    try:
        out = subprocess.run(
            [sys.executable, str(script), path],
            capture_output=True, text=True, check=True,
        ).stdout
    finally:
        Path(path).unlink()

    def weight(name: str) -> int:
        m = re.search(rf"'{name}': (\d+)", out)
        if m is None:
            print(f"FATAL: no weight emitted for {name}\n{out}")
            return -1
        return int(m.group(1))

    failures = []
    for name, ns in timings.items():
        if (got := weight(name)) != (want := max(1, round(ns / 2.0))):
            failures.append(f"{name}: got {got}, want {want}")
    # Malformed input must fail loudly, never silently.
    bad = subprocess.run(
        [sys.executable, str(script), "/nonexistent.json"],
        capture_output=True, text=True,
    )
    if bad.returncode == 0:
        failures.append("missing file did not fail")
    if failures:
        print("FATAL:\n" + "\n".join(failures))
        return 1
    print("test_fit_weights: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
