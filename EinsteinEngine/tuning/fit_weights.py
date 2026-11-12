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

The tuner fits the results it measured on its target (see microbench.py). To fit a
results file by hand: ``python -m EinsteinEngine.tuning.fit_weights results.json``.

Add is the reference, and the unit. ee_microbench runs the whole suite in
rounds and reports each case's time in every round (``ns_rounds``). A case's
ratio is the median, over rounds, of its time divided by add's time in the same
round, so a change in clock speed between rounds cancels. The stored weight is

    max(1, floor(median_ratio + 0.5))

that is, the ratio rounded to whole adds (half up, not to even), and at least
one add. A row without ``ns_rounds`` counts as one round of ``ns_per_elem``.

Device runs time the harness with no math op as ``identity_stream``: in
``arith`` for the two-input kernel, and in ``transcendental`` for the one-input
kernel. Each timing used for a ratio is ``max(0, ns - identity)``, with the
identity of the row's own group if it has one, and ``arith``'s otherwise. A
non-positive adjusted add is fatal. Host runs omit those rows and are not
adjusted.

Rows are keyed by ``(group, name)``. A duplicate key is fatal, so a
``sqrt_stream`` in ``pow`` cannot overwrite the one in ``transcendental``.
The transcendental row supplies the ``sqrt`` and ``cbrt`` weights.

Only cost-model slots become weights. ``sub``, ``div``, ``neg``, ``const``,
``cmp_lt``, and ``if_else*`` are recorded and are not weights. ``atom``,
``symbol_local``, ``symbol_grid``, ``pow_integer_floor``, the three stencil
weights, and ``transcendental_default`` keep the values of ``DEFAULT_WEIGHTS``.
``bench_memory`` is informational: a single-node loop cannot see a ghost
exchange.

A min-to-median spread above 25% is returned as a warning and stored in the
profile. The script does not probe the machine, and it does not invent
measured numbers for slots it does not fit.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import asdict
from typing import Any, NamedTuple

from EinsteinEngine.generators.sympy_complexity import DEFAULT_WEIGHTS

SPREAD_WARN = 0.25

# The slots that are not fitted keep the default model's values.
POLICY_WEIGHTS: dict[str, int] = {
    key: value for key, value in asdict(DEFAULT_WEIGHTS).items()
    if key not in ("pow_default", "transcendental")
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
    ns_rounds: tuple[float, ...]


def _fatal(message: str) -> ValueError:
    return ValueError(message)


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


def load_results(doc: Any) -> dict[tuple[str, str], Row]:
    """The rows of parsed ee_microbench output (an object with "results", or a bare list of rows)."""
    rows_obj: Any
    if isinstance(doc, dict):
        rows_obj = doc.get("results")
    elif isinstance(doc, list):
        rows_obj = doc
    else:
        raise _fatal("ee_microbench output must be a JSON object or a list of result rows")
    if not isinstance(rows_obj, list):
        raise _fatal("ee_microbench output field 'results' must be a list")

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
        raw_rounds = row_dict.get("ns_rounds", [ns])
        if not isinstance(raw_rounds, list) or not raw_rounds:
            raise _fatal(f"{group}/{name} ns_rounds must be a non-empty list")
        rounds = tuple(_require_float({"ns_rounds": value}, "ns_rounds") for value in raw_rounds)
        out[key] = Row(group, name, ns, median, rounds)
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


def _baseline(rows: dict[tuple[str, str], Row], row: Row) -> Row | None:
    """The identity timing to subtract from `row`: its own group's, or the two-input kernel's in arith."""
    return rows.get((row.group, "identity_stream"), rows.get(("arith", "identity_stream")))


def _round_ratio(rows: dict[tuple[str, str], Row], row: Row, add: Row) -> float:
    """The median over rounds of `row`'s adjusted time divided by add's adjusted time in the same round."""
    row_base, add_base = _baseline(rows, row), _baseline(rows, add)
    involved = [r for r in (row, add, row_base, add_base) if r is not None]
    if len({len(r.ns_rounds) for r in involved}) != 1:
        raise _fatal(f"{row.group}/{row.name} and its references were not timed in the same number of rounds")
    ratios: list[float] = []
    for k in range(len(row.ns_rounds)):
        add_ns = add.ns_rounds[k] - (add_base.ns_rounds[k] if add_base is not None else 0.0)
        if add_ns <= 0:
            raise _fatal(
                f"adjusted add_stream is not positive in round {k} after subtracting identity_stream. "
                "The identity baseline is not below add, so ratios are undefined."
            )
        row_ns = max(0.0, row.ns_rounds[k] - (row_base.ns_rounds[k] if row_base is not None else 0.0))
        ratios.append(row_ns / add_ns)
    return statistics.median(ratios)


def _quantize(ratio: float) -> int:
    if not math.isfinite(ratio) or ratio < 0:
        raise _fatal(f"invalid ratio {ratio}")
    return max(1, math.floor(ratio + 0.5))


def _spread_warning(row: Row) -> str | None:
    spread = (row.ns_median - row.ns_per_elem) / row.ns_per_elem
    if spread > SPREAD_WARN:
        return (
            f"{row.group}/{row.name} min/median spread is {spread:.0%} "
            f"({row.ns_per_elem:.6g} vs {row.ns_median:.6g})"
        )
    return None


def build_profile(rows: dict[tuple[str, str], Row]) -> tuple[dict[str, Any], list[str]]:
    add = _need(rows, "arith", "add_stream")
    identities = {group: row for (group, name), row in rows.items() if name == "identity_stream"}
    _round_ratio(rows, add, add)  # rejects an add that is not above its baseline

    warnings: list[str] = []
    for row in rows.values():
        warning = _spread_warning(row)
        if warning is not None:
            warnings.append(warning)

    def weight_for(row: Row) -> int:
        return _quantize(_round_ratio(rows, row, add))

    pow_default_row = _need(rows, "pow", "powvar_stream")
    transcendental: dict[str, int] = {}
    transcendental_source: dict[str, str] = {}
    for stream_name, fn_name in TRANSCENDENTAL_STREAMS.items():
        picked = _pick_named(rows, stream_name, "transcendental")
        transcendental[fn_name] = weight_for(picked)
        transcendental_source[fn_name] = f"{picked.group}/{picked.name}"

    weights: dict[str, Any] = {
        **POLICY_WEIGHTS,
        "pow_default": weight_for(pow_default_row),
        "transcendental": transcendental,
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
            "ns_rounds": list(row.ns_rounds),
            "applied_to": applied.get(row_key),
        }
        recorded.append(entry)

    scale = "max(1, floor(median over rounds of adjusted_ns / adjusted_add + 0.5))"
    decisions = {
        "scale": (
            f"Fitted slots use {scale}. "
            "One adjusted add is 1."
        ),
        "identity": (
            f"identity_stream ({', '.join(sorted(identities))}) was subtracted, the row's own group's if it has one "
            "and arith's otherwise, and the difference floored at 0."
            if identities
            else "No identity_stream rows, so timings were not adjusted. Device runs emit them; host runs do not."
        ),
        "pow_default": (
            f"pow/powvar_stream -> {weights['pow_default']}. "
            "Integer pown timings are recorded and are not a weight. "
            f"pow_integer_floor stays {POLICY_WEIGHTS['pow_integer_floor']}."
        ),
        "transcendental": (
            "sqrt and cbrt are taken from the transcendental group when that "
            "row exists, so a pow/sqrt_stream measurement cannot overwrite them. "
            + ", ".join(f"{key} from {transcendental_source[key]}" for key in sorted(transcendental_source))
        ),
        "stencil": (
            "stencil_center/stencil_x/stencil_yz keep the default model's values. "
            "bench_memory is informational and is not fitted."
        ),
        "symbol_grid": "symbol_grid keeps the default model's value. grid_add_stream is recorded and is not fitted.",
        "unmapped": (
            "sub, div, neg, const, cmp_lt, and if_else have no slot in the cost "
            "model. Their timings are under measurements.rows and are not weights."
        ),
    }

    profile: dict[str, Any] = {
        "schema_version": 1,
        "profile": "fitted",
        "description": (
            "Profile fitted by EinsteinEngine.tuning.fit_weights. Fitted slots use "
            f"{scale}. Policy slots (atom, symbol_grid, symbol_local, "
            "pow_integer_floor, stencil weights, transcendental_default) are "
            "the default model's values, not a fit of bench_memory. Device ratios "
            "need the identity_stream rows before they are trusted. Host timings "
            "(one op per element, memory- and loop-bound for cheap ops) and "
            "device timings (256 reps per element, plus harness overhead) are "
            "not comparable."
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
            "unit": "ns_per_elem; ns_rounds holds each round's min over repeats",
            "reference": {
                "add_stream": add.ns_per_elem,
                "identity_stream": {group: row.ns_per_elem for group, row in sorted(identities.items())},
            },
            "identity_subtracted": bool(identities),
            "rows": recorded,
            "warnings": warnings,
            "decisions": decisions,
        },
    }
    return profile, warnings


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m EinsteinEngine.tuning.fit_weights <results.json|->")
    path = sys.argv[1]
    try:
        if path == "-":
            doc = json.load(sys.stdin)
        else:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        profile, warnings = build_profile(load_results(doc))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    json.dump(profile, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
