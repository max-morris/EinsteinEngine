#  Copyright (C) 2025-2026 Max Morris and other Einstein Engine contributors.
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

"""Cost model for SymPy expressions, with weights loaded from JSON at startup.

The active weight profile is selected once at import time:
  * ``EE_COMPLEXITY_WEIGHTS`` env var set -> that JSON file is loaded
    (a missing/unreadable file is a fatal error, never a silent fallback);
  * otherwise the bundled ``guestimates.json`` profile is used.

Call :func:`set_weights` (or point the env var at another file and reimport)
to switch profiles. Measured profiles live next to ``guestimates.json``;
``microbenchmarks/README.md`` says which runs produced them.
``microbenchmarks/scripts/fit_weights.py`` only suggests integers from a
results file — it does not write a profile. :func:`available_profiles`
lists the JSON profiles shipped with the package.
"""

import json
import os
from dataclasses import dataclass, field
from math import log2
from pathlib import Path
from typing import Any, cast, Protocol, Union

import sympy as sy
from multimethod import multimethod
from sympy.core.function import UndefinedFunction

WEIGHTS_SCHEMA_VERSION = 1
WEIGHTS_ENV_VAR = "EE_COMPLEXITY_WEIGHTS"
WEIGHTS_DIR = Path(__file__).resolve().parent / "complexity_weights"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "guestimates.json"


class IsGridVariableFn(Protocol):
    def __call__(self, symbol: sy.Symbol, /) -> bool: ...


@dataclass(frozen=True)
class ComplexityWeights:
    """Integer cost-model weights; ``Add``/``Mul`` sum their children."""

    atom: int = 1
    symbol_grid: int = 10
    symbol_local: int = 1
    pow_default: int = 15
    pow_integer_floor: int = 2
    stencil_center: int = 10
    stencil_x: int = 40
    stencil_yz: int = 100
    transcendental: dict[str, int] = field(default_factory=dict)
    transcendental_default: int = 0


def _require_int(data: dict[str, Any], key: str) -> int:
    value: Any = data[key]  # KeyError propagates: missing key is fatal.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Complexity weight '{key}' must be an int, got {value!r}.")
    if value < 0:
        raise ValueError(f"Complexity weight '{key}' must be >= 0, got {value}.")
    return int(value)


def load_weights(path: Union[str, Path]) -> ComplexityWeights:
    """Load and validate a weight profile. Any problem raises (fatal)."""
    resolved = Path(path).expanduser()
    with open(resolved, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"Weight file '{resolved}' must contain a JSON object.")
    if doc.get("schema_version") != WEIGHTS_SCHEMA_VERSION:
        raise ValueError(
            f"Weight file '{resolved}' has schema_version {doc.get('schema_version')!r}, "
            f"expected {WEIGHTS_SCHEMA_VERSION}."
        )
    raw_weights = doc.get("weights")
    if not isinstance(raw_weights, dict):
        raise ValueError(f"Weight file '{resolved}' is missing the 'weights' object.")
    weights = cast(dict[str, Any], raw_weights)

    raw_trans = weights.get("transcendental", {})
    if not isinstance(raw_trans, dict):
        raise ValueError(f"Weight file '{resolved}': 'transcendental' must be an object.")
    transcendental: dict[str, int] = {}
    for name, value in raw_trans.items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"Weight file '{resolved}': transcendental entry {name!r} must map "
                f"a function name to an int, got {value!r}."
            )
        if value < 0:
            raise ValueError(
                f"Weight file '{resolved}': transcendental entry {name!r} must be >= 0."
            )
        transcendental[name] = value

    return ComplexityWeights(
        atom=_require_int(weights, "atom"),
        symbol_grid=_require_int(weights, "symbol_grid"),
        symbol_local=_require_int(weights, "symbol_local"),
        pow_default=_require_int(weights, "pow_default"),
        pow_integer_floor=_require_int(weights, "pow_integer_floor"),
        stencil_center=_require_int(weights, "stencil_center"),
        stencil_x=_require_int(weights, "stencil_x"),
        stencil_yz=_require_int(weights, "stencil_yz"),
        transcendental=transcendental,
        transcendental_default=_require_int(weights, "transcendental_default"),
    )


def available_profiles() -> list[str]:
    """Names (stems) of the JSON profiles shipped with the package."""
    if not WEIGHTS_DIR.is_dir():
        raise FileNotFoundError(f"Weight directory '{WEIGHTS_DIR}' does not exist.")
    return sorted(p.stem for p in WEIGHTS_DIR.glob("*.json"))


_ACTIVE_WEIGHTS: ComplexityWeights


def _load_startup_weights() -> ComplexityWeights:
    override = os.environ.get(WEIGHTS_ENV_VAR, "").strip()
    if override:
        return load_weights(override)
    return load_weights(DEFAULT_WEIGHTS_PATH)


def get_weights() -> ComplexityWeights:
    """The currently active weight profile."""
    return _ACTIVE_WEIGHTS


def set_weights(weights: ComplexityWeights) -> None:
    """Switch the active weight profile at runtime."""
    global _ACTIVE_WEIGHTS
    _ACTIVE_WEIGHTS = weights


_ACTIVE_WEIGHTS = _load_startup_weights()


def calculate_complexity(expr: sy.Basic, *, is_grid_variable: IsGridVariableFn) -> int:
    return cast(int, SympyComplexityVisitor(is_grid_variable).complexity(expr))

def calculate_complexities(eqns: dict[sy.Symbol, sy.Expr], *, is_grid_variable: IsGridVariableFn) -> dict[sy.Symbol, int]:
    visitor = SympyComplexityVisitor(is_grid_variable)
    return {lhs: visitor.complexity(rhs) for lhs, rhs in eqns.items()}


class SympyComplexityVisitor:
    is_grid_variable: IsGridVariableFn
    weights: ComplexityWeights

    def __init__(self, is_grid_variable: IsGridVariableFn, weights: ComplexityWeights | None = None):
        self.is_grid_variable = is_grid_variable
        self.weights = weights if weights is not None else get_weights()

    @multimethod
    def complexity(self, _n: sy.Atom) -> int:
        return self.weights.atom

    @complexity.register
    def _(self, n: sy.Add) -> int:
        return sum([self.complexity(arg) for arg in n.args])

    @complexity.register
    def _(self, n: sy.Mul) -> int:
        return sum([self.complexity(arg) for arg in n.args])

    @complexity.register
    def _(self, n: sy.Pow) -> int:
        c: int = self.weights.pow_default
        power = n.args[1]
        if power.is_Integer:
            # log2(|p|) is in units of one atom. atom is 1 in the dataclass
            # defaults and 100 in normalized JSON profiles.
            magnitude = abs(int(power))
            exponent_cost = 0 if magnitude <= 1 else round(self.weights.atom * log2(magnitude))
            c = max(self.weights.pow_integer_floor, exponent_cost)
        elif power == sy.Rational(1, 2):
            # Emitter lowers a positive half to sqrt(), not pow().
            c = self.weights.transcendental.get("sqrt", self.weights.transcendental_default)
        elif power == sy.Rational(1, 3):
            # Emitter lowers a positive third to cbrt(). Negative thirds stay pow().
            c = self.weights.transcendental.get("cbrt", self.weights.transcendental_default)
        return int(c + sum([self.complexity(arg) for arg in n.args]))

    @complexity.register
    def _(self, n: sy.Symbol) -> int:
        return self.weights.symbol_grid if self.is_grid_variable(n) else self.weights.symbol_local

    @complexity.register
    def _(self, _n: sy.IndexedBase) -> int:
        return self.weights.atom

    @complexity.register
    def _(self, _n: sy.Indexed) -> int:
        return self.weights.atom

    @complexity.register
    def _(self, _n: sy.Number) -> int:
        return self.weights.atom

    @complexity.register
    def _(self, n: sy.Piecewise) -> int:
        total = 0
        for pair in n.args:
            assert isinstance(pair, sy.core.containers.Tuple) and len(pair) == 2
            e, c = pair
            total += self.complexity(e) + self.complexity(c)
        return total

    @complexity.register
    def _(self, n: sy.core.relational.Relational) -> int:
        return sum([self.complexity(arg) for arg in n.args])

    @complexity.register
    def _(self, _n: sy.logic.boolalg.BooleanTrue) -> int:
        return self.weights.atom

    @complexity.register
    def _(self, _n: sy.logic.boolalg.BooleanFalse) -> int:
        return self.weights.atom

    def _complexity_undefined_fn(self, n: sy.Function) -> int:
        assert isinstance(n.func, UndefinedFunction)

        if n.name == 'stencil':  # type: ignore[attr-defined]
            sym, x, y, z = n.args
            assert isinstance(x, sy.Number)
            assert isinstance(y, sy.Number)
            assert isinstance(z, sy.Number)

            if y.evalf() != 0 or z.evalf() != 0:
                return self.weights.stencil_yz
            elif x.evalf() != 0:
                return self.weights.stencil_x
            else:
                return self.weights.stencil_center

        args_complexity: int = sum([self.complexity(arg) for arg in n.args])
        return args_complexity

    @complexity.register
    def _(self, n: sy.Function) -> int:
        if isinstance(n.func, UndefinedFunction):
            return self._complexity_undefined_fn(n)

        args_complexity: int = sum([self.complexity(arg) for arg in n.args])
        extra = self.weights.transcendental.get(type(n).__name__, self.weights.transcendental_default)
        return extra + args_complexity
