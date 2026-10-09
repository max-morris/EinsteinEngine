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

"""Cost model for SymPy expressions. Weights come from JSON.

The active profile is loaded on the first call to :func:`get_weights`, not at
import, so a bad ``EE_COMPLEXITY_WEIGHTS`` does not break importing this
module:

  * ``EE_COMPLEXITY_WEIGHTS`` set to a path -> that JSON file is loaded
    (a missing or unreadable file is a fatal error, never a silent fallback);
  * otherwise the bundled ``guestimates.json`` profile is used.

That file is the only profile shipped inside the package. Measured profiles
live in ``microbenchmarks/profiles/`` and are selected by passing their path.
``microbenchmarks/README.md`` records which run produced each file.

:class:`ComplexityWeights` has no numeric defaults. The JSON profile is the
source of truth; the old dataclass defaults (``atom=1``, empty transcendentals)
did not match the profile that actually runs.

Weights are about 100 times the pre-profile integer model (``atom`` was 1 and
is 100). :func:`promote_threshold` and :func:`retain_threshold` take absolute
complexity integers, so a threshold chosen on the old scale now keeps about
100 times fewer candidates. Percentile and rank strategies do not use that
absolute scale. Nothing in this repository calls the threshold strategies.

This visitor is the backend-independent model. Every non-integer power,
including ``sqrt`` and ``cbrt``, costs ``pow_default`` plus both arguments.
``CppCarpetXComplexityVisitor`` overrides :meth:`_complexity_pow` because
CarpetX emits ``x**(1/2)`` and ``x**(1/3)`` as ``sqrt`` and ``cbrt``: those
cost the function's profile weight plus the base, and the exponent is not
charged. Other backends emit them as ``pow`` and keep the general cost.

A listed transcendental costs its profile weight plus its arguments. The
functions ``sympywrap`` exports that a profile does not list (``tan``,
``cot``, ``sec``, ``csc``, ``atan``, the hyperbolics, ``erf``, and the rest
of that set) cost the same as ``sin``, so they do not rank cheaper than
``sin``. A profile that lists the function uses that listed weight, which is
how a measured profile can rank ``tan`` differently from ``sin``. Any other
call, such as ``Abs``, has no surcharge.

:func:`set_weights` replaces the process-wide profile. Callers that use it
temporarily have to restore the previous value; a later ``get_weights`` will
not reload the startup file while a profile is installed.
"""

import json
import os
from dataclasses import dataclass
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


# Every transcendental function sympywrap exports. Compare ``n.func`` with this
# set: ``n in [sy.sin, ...]`` compares the applied call to the classes and
# never matches. sqrt and cbrt are not here; SymPy builds them as powers, and
# only the CarpetX hook charges those exponents as calls. A name in this set
# that the active profile omits costs the same as sin.
_TRANSCENDENTAL_FUNCTIONS = frozenset({
    sy.sin, sy.cos, sy.tan, sy.cot, sy.sec, sy.csc, sy.atan,
    sy.sinh, sy.cosh, sy.tanh, sy.coth, sy.sech, sy.csch,
    sy.exp, sy.log, sy.erf,
})
_TRANSCENDENTAL_NAMES = frozenset(fn.__name__ for fn in _TRANSCENDENTAL_FUNCTIONS) | {"sqrt", "cbrt"}


class IsGridVariableFn(Protocol):
    def __call__(self, symbol: sy.Symbol, /) -> bool: ...


@dataclass(frozen=True)
class ComplexityWeights:
    """Integer cost-model weights; ``Add``/``Mul`` sum their children.

    Every field is required. Shipped numbers live in ``guestimates.json``.
    """

    atom: int
    symbol_grid: int
    symbol_local: int
    pow_default: int
    pow_integer_floor: int
    stencil_center: int
    stencil_x: int
    stencil_yz: int
    transcendental: dict[str, int]
    transcendental_default: int


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


_ACTIVE_WEIGHTS: ComplexityWeights | None = None


def _load_startup_weights() -> ComplexityWeights:
    override = os.environ.get(WEIGHTS_ENV_VAR, "").strip()
    if override:
        return load_weights(override)
    return load_weights(DEFAULT_WEIGHTS_PATH)


def get_weights() -> ComplexityWeights:
    """The currently active weight profile, loaded on first use."""
    global _ACTIVE_WEIGHTS
    if _ACTIVE_WEIGHTS is None:
        _ACTIVE_WEIGHTS = _load_startup_weights()
    return _ACTIVE_WEIGHTS


def set_weights(weights: ComplexityWeights) -> None:
    """Replace the process-wide weight profile.

    This does not reload ``EE_COMPLEXITY_WEIGHTS``. Restore the previous
    profile when the replacement was only meant for the current call.
    """
    global _ACTIVE_WEIGHTS
    _ACTIVE_WEIGHTS = weights


def calculate_complexity(expr: sy.Basic, *, is_grid_variable: IsGridVariableFn) -> int:
    return cast(int, SympyComplexityVisitor(is_grid_variable).complexity(expr))

def calculate_complexities(eqns: dict[sy.Symbol, sy.Expr], *, is_grid_variable: IsGridVariableFn) -> dict[sy.Symbol, int]:
    visitor = SympyComplexityVisitor(is_grid_variable)
    return {lhs: visitor.complexity(rhs) for lhs, rhs in eqns.items()}


class SympyComplexityVisitor:
    """
    The backend-independent cost model.

    A backend whose output makes some operations cheaper than this model assumes
    subclasses it and overrides the matching `_complexity_*` hook.
    """

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

    def _transcendental_surcharge(self, name: str) -> int:
        """Profile weight for ``name``.

        A listed name uses that weight. An unlisted name in
        :data:`_TRANSCENDENTAL_NAMES` uses ``sin`` when the profile lists
        ``sin``, so ``cot`` does not rank cheaper than ``sin`` on a profile
        that never measured ``cot``. Anything else uses
        ``transcendental_default`` (0 in every shipped profile).
        """
        listed = self.weights.transcendental.get(name)
        if listed is not None:
            return listed
        if name in _TRANSCENDENTAL_NAMES:
            sin_cost = self.weights.transcendental.get("sin")
            if sin_cost is not None:
                return sin_cost
        return self.weights.transcendental_default

    @complexity.register
    def _(self, n: sy.Pow) -> int:
        return self._complexity_pow(n)

    def _complexity_pow(self, n: sy.Pow) -> int:
        # Every non-integer power, sqrt and cbrt included, is a general power:
        # pow_default plus the base and the exponent. CarpetX overrides this
        # hook for the exponents it emits as sqrt() and cbrt().
        power = n.args[1]
        if power.is_Integer:
            # The pre-scale model was max(2, int(log2(|p|))). Scaling that by
            # atom (100 in every shipped profile) is
            # max(pow_integer_floor, atom * int(log2(|p|))), with the floor
            # equal to 2*atom. Truncating, rather than rounding atom*log2,
            # keeps x**5 at 200 and x**6 at 200 on that scale. |p| <= 1 uses
            # the floor (the old cost was 2, which scales to 200).
            magnitude = abs(int(power))
            if magnitude <= 1:
                exponent_cost = 0
            else:
                exponent_cost = self.weights.atom * int(log2(magnitude))
            operation_cost = max(self.weights.pow_integer_floor, exponent_cost)
        else:
            operation_cost = self.weights.pow_default
        args_complexity: int = sum([self.complexity(arg) for arg in n.args])
        return operation_cost + args_complexity

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
        # Compare the function class. `n in [sy.sin, ...]` compares the
        # applied call to those classes and never matches. sqrt and cbrt
        # never reach here, since SymPy builds them as powers. The weight
        # itself comes from the profile, keyed by the class name.
        name = type(n).__name__
        if n.func in _TRANSCENDENTAL_FUNCTIONS or name in self.weights.transcendental:
            return self._transcendental_surcharge(name) + args_complexity
        return args_complexity
