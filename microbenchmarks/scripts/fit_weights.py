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

"""Fit a schema_version 1 complexity profile from ee_microbench JSON.

Add is the reference. The stored weight is

    max(100, round(100 * adjusted_ns / adjusted_add))

which keeps the fractional part the x100 scale was for. ``100 * max(1, round(ratio))``
collapsed every ratio in ``[0.5, 1.5)`` to 100 and, because ``round`` ties to even,
turned 2.5 into 200.

When the results contain ``arith/identity_stream`` (the device harness with no
math op), each timing used for a ratio is ``max(0, ns - identity)``. A
non-positive adjusted add is fatal. Host runs omit that row and are not
adjusted.

Rows are keyed by ``(group, name)``. A duplicate key is fatal, so a
``sqrt_stream`` in ``pow`` cannot overwrite the one in ``transcendental``.
The transcendental row supplies the ``sqrt`` and ``cbrt`` weights.

Only cost-model slots become weights. ``sub``, ``div``, ``neg``, ``const``,
``cmp_lt``, and ``if_else*`` are recorded and are not weights. ``atom``,
``symbol_local``, ``symbol_grid``, ``pow_integer_floor``, the three stencil
weights, and ``transcendental_default`` stay the guestimate policy.
``bench_memory`` is informational: a single-node loop cannot see a ghost
exchange.

A min-to-median spread above 25% is reported on stderr and stored in the
profile. The script does not probe the machine, and it does not invent
measured numbers for slots it does not fit.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any, NamedTuple

SPREAD_WARN = 0.25

POLICY_WEIGHTS: dict[str, int] = {
    "atom": 100,
    "symbol_grid": 1000,
    "symbol_local": 100,
    "pow_integer_floor": 200,
    "stencil_center": 1000,
    "stencil_x": 4000,
    "stencil_yz": 10000,
    "transcendental_default": 0,
}

# Benchmark name -> transcendental weight key. The transcendental group wins
# when the same name also exists under pow.
TRANSCENDENTAL_STREAMS: dict[str, str] = {
    "sin_stream": "sin",
    "cos_stream": "cos",
    "tan_stream": "tan",
    "sinh_stream": "sinh",
    "cosh_stream": "cosh",
    "tanh_stream": "tanh",
    "exp_stream": "exp",
    "log_stream": "log",
    "erf_stream": "erf",
    "sqrt_stream": "sqrt",
    "cbrt_stream": "cbrt",
}


class Row(NamedTuple):
    group: str
    name: str
    ns_per_elem: float
    ns_median: float


def _fatal(message: str) -> SystemExit:
    return SystemExit(f"FATAL: {message}")


def _require_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise _fatal(f"result row {row!r} field {key!r} must be a non-empty string")
    return value


def _require_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fatal(f"result row {row!r} field {key!r} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise _fatal(f"result row {row!r} field {key!r} must be finite and >= 0, got {number}")
    return number


def load_results(path: str) -> dict[tuple[str, str], Row]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read()
        except OSError as exc:
            raise _fatal(f"cannot read {path}: {exc}") from exc
    try:
        doc: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _fatal(f"{path} is not JSON: {exc}") from exc

    rows_obj: Any
    if isinstance(doc, dict):
        rows_obj = doc.get("results")
    elif isinstance(doc, list):
        rows_obj = doc
    else:
        raise _fatal(f"{path} must be a JSON object or a list of result rows")
    if not isinstance(rows_obj, list):
        raise _fatal(f"{path} field 'results' must be a list")

    out: dict[tuple[str, str], Row] = {}
    for item in rows_obj:
        if not isinstance(item, dict):
            raise _fatal(f"malformed result row {item!r}")
        row_dict: dict[str, Any] = item
        group = _require_str(row_dict, "group")
        name = _require_str(row_dict, "name")
        key = (group, name)
        if key in out:
            raise _fatal(f"duplicate result for group {group!r} name {name!r}")
        ns = _require_float(row_dict, "ns_per_elem")
        median = _require_float(row_dict, "ns_median")
        if ns == 0:
            raise _fatal(f"{group}/{name} ns_per_elem is 0")
        if median + 1e-9 < ns:
            raise _fatal(f"{group}/{name} ns_median {median} is below ns_per_elem {ns}")
        out[key] = Row(group, name, ns, median)
    return out


def _need(rows: dict[tuple[str, str], Row], group: str, name: str) -> Row:
    try:
        return rows[(group, name)]
    except KeyError:
        raise _fatal(
            f"missing benchmark {group}/{name}. "
            "Re-run ee_microbench without --group filtering."
        ) from None


def _pick_named(rows: dict[tuple[str, str], Row], name: str, preferred_group: str) -> Row:
    matches = [row for row in rows.values() if row.name == name]
    if not matches:
        raise _fatal(
            f"missing benchmark {name!r}. Re-run ee_microbench without --group filtering."
        )
    preferred = [row for row in matches if row.group == preferred_group]
    if preferred:
        return preferred[0]
    if len(matches) == 1:
        return matches[0]
    groups = ", ".join(sorted(row.group for row in matches))
    raise _fatal(
        f"{name!r} appears in groups {groups} and none of them is {preferred_group!r}"
    )


def _adjusted(row: Row, identity: Row | None) -> float:
    if identity is None:
        return row.ns_per_elem
    return max(0.0, row.ns_per_elem - identity.ns_per_elem)


def _quantize(ratio: float) -> int:
    if not math.isfinite(ratio) or ratio < 0:
        raise _fatal(f"invalid ratio {ratio}")
    return max(100, round(100 * ratio))


def _spread_warning(row: Row) -> str | None:
    spread = (row.ns_median - row.ns_per_elem) / row.ns_per_elem
    if spread > SPREAD_WARN:
        return (
            f"{row.group}/{row.name} min/median spread is {spread:.0%} "
            f"({row.ns_per_elem:.6g} vs {row.ns_median:.6g})"
        )
    return None


def build_profile(rows: dict[tuple[str, str], Row]) -> tuple[dict[str, Any], list[str]]:
    identity = rows.get(("arith", "identity_stream"))
    add = _need(rows, "arith", "add_stream")
    add_ns = _adjusted(add, identity)
    if add_ns <= 0:
        raise _fatal(
            "adjusted add_stream is 0 after subtracting identity_stream. "
            "The identity baseline is not below add, so ratios are undefined."
        )

    warnings: list[str] = []
    for row in rows.values():
        warning = _spread_warning(row)
        if warning is not None:
            warnings.append(warning)

    def weight_for(row: Row) -> int:
        return _quantize(_adjusted(row, identity) / add_ns)

    pow_default_row = _need(rows, "pow", "powvar_stream")
    transcendental: dict[str, int] = {}
    transcendental_source: dict[str, str] = {}
    for stream_name, fn_name in TRANSCENDENTAL_STREAMS.items():
        picked = _pick_named(rows, stream_name, "transcendental")
        transcendental[fn_name] = weight_for(picked)
        transcendental_source[fn_name] = f"{picked.group}/{picked.name}"

    weights: dict[str, Any] = {
        "atom": POLICY_WEIGHTS["atom"],
        "symbol_grid": POLICY_WEIGHTS["symbol_grid"],
        "symbol_local": POLICY_WEIGHTS["symbol_local"],
        "pow_default": weight_for(pow_default_row),
        "pow_integer_floor": POLICY_WEIGHTS["pow_integer_floor"],
        "stencil_center": POLICY_WEIGHTS["stencil_center"],
        "stencil_x": POLICY_WEIGHTS["stencil_x"],
        "stencil_yz": POLICY_WEIGHTS["stencil_yz"],
        "transcendental": transcendental,
        "transcendental_default": POLICY_WEIGHTS["transcendental_default"],
    }

    applied: dict[tuple[str, str], str] = {("pow", "powvar_stream"): "pow_default"}
    for stream_name, fn_name in TRANSCENDENTAL_STREAMS.items():
        source = transcendental_source[fn_name]
        group_name, _, bench_name = source.partition("/")
        applied[(group_name, bench_name)] = f"transcendental.{fn_name}"

    recorded: list[dict[str, Any]] = []
    for row_key in sorted(rows):
        row = rows[row_key]
        entry: dict[str, Any] = {
            "group": row.group,
            "name": row.name,
            "ns_per_elem": row.ns_per_elem,
            "ns_median": row.ns_median,
            "applied_to": applied.get(row_key),
        }
        recorded.append(entry)

    scale = "max(100, round(100 * adjusted_ns / adjusted_add))"
    decisions = {
        "scale": (
            f"Fitted slots use {scale}. "
            "atom stays 100, so one adjusted add rounds to 100."
        ),
        "identity": (
            "arith/identity_stream was subtracted and the difference floored at 0."
            if identity is not None
            else "No arith/identity_stream row, so timings were not adjusted. "
            "Device runs emit that row; host runs do not."
        ),
        "pow_default": (
            f"pow/powvar_stream -> {weights['pow_default']}. "
            "Integer pown timings are recorded and are not a weight. "
            "pow_integer_floor stays 200 (2*atom)."
        ),
        "transcendental": (
            "sqrt and cbrt are taken from the transcendental group when that "
            "row exists, so a pow/sqrt_stream measurement cannot overwrite them. "
            + ", ".join(f"{key} from {transcendental_source[key]}" for key in sorted(transcendental_source))
        ),
        "stencil": (
            "stencil_center/stencil_x/stencil_yz stay 1000/4000/10000. "
            "bench_memory is informational and is not fitted."
        ),
        "symbol_grid": "symbol_grid stays 1000. grid_add_stream is recorded and is not fitted.",
        "unmapped": (
            "sub, div, neg, const, cmp_lt, and if_else have no slot in the cost "
            "model. Their timings are under measurements.rows and are not weights."
        ),
    }

    profile: dict[str, Any] = {
        "schema_version": 1,
        "profile": "fitted",
        "description": (
            "Draft profile written by fit_weights.py. Fitted slots use "
            f"{scale}. Policy slots (atom, symbol_grid, symbol_local, "
            "pow_integer_floor, stencil weights, transcendental_default) are "
            "the guestimate values, not a fit of bench_memory. Device ratios "
            "need arith/identity_stream before they are trusted. Host timings "
            "(one op per element, memory- and loop-bound for cheap ops) and "
            "device timings (256 reps per element, plus harness overhead) are "
            "not comparable. Fill architecture in by hand before shipping this file."
        ),
        "architecture": {
            "scalar_only": True,
            "note": (
                "ee_microbench times scalar vreal. Arith::simd<CCTK_REAL> has "
                "different relative costs and is not measured. fit_weights.py "
                "does not probe the CPU, the GPU, or the compiler."
            ),
        },
        "weights": weights,
        "measurements": {
            "unit": "ns_per_elem_min_over_repeats",
            "reference": {
                "add_stream": add.ns_per_elem,
                "identity_stream": None if identity is None else identity.ns_per_elem,
            },
            "identity_subtracted": identity is not None,
            "rows": recorded,
            "warnings": warnings,
            "decisions": decisions,
        },
    }
    return profile, warnings


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: fit_weights.py <results.json|->")
    rows = load_results(sys.argv[1])
    profile, warnings = build_profile(rows)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    json.dump(profile, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
