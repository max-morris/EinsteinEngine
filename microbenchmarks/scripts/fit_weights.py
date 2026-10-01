#!/usr/bin/env python3
# Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit integer complexity weights from ee_microbench JSON output.

The additive model in EinsteinEngine/generators/sympy_complexity.py assigns
each node an integer cost; Add/Mul sum their children. This script converts
measured ns/element into suggested integers. Add is the reference, and the
smallest suggestion is 100 (profiles use that same scale; a stored 1 on the
old scale is 100 here):

    weight(op) = 100 * max(1, round(ns_op / ns_add))

A ratio below half an add still clamps to 100, as the old formula clamped it
to 1. It also reports ratios against the guestimate profile (integer powers
vs max(200, round(100*log2(|p|))), stencil center/x/y/z vs 1000/4000/10000,
transcendentals vs a flat 1500, grid Symbol vs local 100). Those lines are
comparisons, not instructions to copy the guestimates onto a new machine. A
device run whose x/y/z stencil timings are flat should not inherit 4000/10000.

Usage:
    ./build/ee_microbench --json results.json > table.txt  # (JSON goes to stdout)
    python3 scripts/fit_weights.py results.json
    ./ee_microbench --quick --json - | python3 scripts/fit_weights.py -

Reads JSON of the form {"results":[{...,"name":...,"ns_per_elem":...}]}.
Failures are fatal (non-zero exit, no silent defaults).
"""

from __future__ import annotations

import json
import math
import sys


def load_results(path: str) -> dict[str, float]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        with open(path) as f:
            raw = f.read()
    doc = json.loads(raw)
    # Accept either {"results":[...]} or a bare [...] list.
    rows = doc["results"] if isinstance(doc, dict) else doc
    out: dict[str, float] = {}
    for r in rows:
        try:
            out[r["name"]] = float(r["ns_per_elem"])
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit(f"FATAL: malformed result row {r!r}: {e}")
    return out


def need(ns: dict[str, float], key: str) -> float:
    try:
        return ns[key]
    except KeyError:
        raise SystemExit(
            f"FATAL: missing benchmark '{key}'. "
            f"Re-run ee_microbench without --group filtering."
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: fit_weights.py <results.json|->")
    ns = load_results(sys.argv[1])

    add = need(ns, "add_stream")
    if not math.isfinite(add) or add <= 0:
        raise SystemExit(f"FATAL: invalid add_stream timing {add}")

    def w(key: str) -> int:
        v = need(ns, key)
        if not math.isfinite(v) or v < 0:
            raise SystemExit(f"FATAL: invalid timing for {key}: {v}")
        return 100 * max(1, round(v / add))

    print("# Suggested weights (smallest positive weight is 100; add_stream rounds to 100)")
    print(f"# add_stream ns/elem = {add:.4f}\n")
    print("ADD_MUL = {")
    for key in ["add_stream", "sub_stream", "mul_stream", "div_stream", "neg_stream"]:
        print(f"    {key!r}: {w(key)},  # {need(ns, key) / add:.2f}x add")
    print("}")
    print("\nPOW = {")
    for key in [
        "pow2_stream",
        "pown3_stream",
        "pown4_stream",
        "pown8_stream",
        "pown16_stream",
        "sqrt_stream",
        "cbrt_stream",
        "powvar_stream",
        "pow2p5_stream",
    ]:
        print(f"    {key!r}: {w(key)},  # {need(ns, key) / add:.2f}x add")
    print("}")
    print("print('current guestimate: default 1500, int-pow max(200, round(100*log2(|p|)))')")
    print("\nTRANSCENDENTAL = {")
    for key in [
        "sin_stream",
        "cos_stream",
        "tan_stream",
        "sinh_stream",
        "cosh_stream",
        "tanh_stream",
        "exp_stream",
        "log_stream",
        "erf_stream",
    ]:
        print(f"    {key!r}: {w(key)},  # {need(ns, key) / add:.2f}x add")
    print("}")
    print("print('current guestimate: sin/cos/exp/log/sqrt/cbrt = 1500')")
    print("\nMEMORY = {")
    for key in [
        "const_stream",
        "center_stream",
        "grid_add_stream",
        "stencil_x_stream",
        "stencil_y_stream",
        "stencil_z_stream",
    ]:
        print(f"    {key!r}: {w(key)},  # {need(ns, key) / add:.2f}x add")
    print("}")
    print("print('current guestimate: center=1000, x=4000, y/z=10000, grid-symbol=1000, local=100')")
    print("\nBRANCH = {")
    for key in ["cmp_lt_stream", "if_else_stream", "if_else_branch_stream"]:
        print(f"    {key!r}: {w(key)},  # {need(ns, key) / add:.2f}x add")
    print("}")


if __name__ == "__main__":
    main()
