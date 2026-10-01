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

from math import log2
from typing import cast, Protocol

import sympy as sy
from multimethod import multimethod
from sympy.core.function import UndefinedFunction


# Every transcendental function sympywrap exports. They all carry the same surcharge, so
# none of them ranks cheaper than another. sqrt and cbrt are not here; SymPy builds them
# as powers.
_TRANSCENDENTAL_FUNCTIONS = frozenset({
    sy.sin, sy.cos, sy.tan, sy.cot, sy.sec, sy.csc, sy.atan,
    sy.sinh, sy.cosh, sy.tanh, sy.coth, sy.sech, sy.csch,
    sy.exp, sy.log, sy.erf,
})


class IsGridVariableFn(Protocol):
    def __call__(self, symbol: sy.Symbol, /) -> bool: ...


def calculate_complexity(expr: sy.Basic, *, is_grid_variable: IsGridVariableFn) -> int:
    return cast(int, SympyComplexityVisitor(is_grid_variable).complexity(expr))

def calculate_complexities(eqns: dict[sy.Symbol, sy.Expr], *, is_grid_variable: IsGridVariableFn) -> dict[sy.Symbol, int]:
    visitor = SympyComplexityVisitor(is_grid_variable)
    return {lhs: visitor.complexity(rhs) for lhs, rhs in eqns.items()}


class SympyComplexityVisitor:
    is_grid_variable: IsGridVariableFn

    def __init__(self, is_grid_variable: IsGridVariableFn):
        self.is_grid_variable = is_grid_variable

    @multimethod
    def complexity(self, _n: sy.Atom) -> int:
        return 1

    @complexity.register
    def _(self, n: sy.Add) -> int:
        return sum([self.complexity(arg) for arg in n.args])

    @complexity.register
    def _(self, n: sy.Mul) -> int:
        return sum([self.complexity(arg) for arg in n.args])

    @complexity.register
    def _(self, n: sy.Pow) -> int:
        base, power = n.args

        # SymPy builds sqrt(x) and cbrt(x) as x**(1/2) and x**(1/3), and the CarpetX
        # backend emits exactly those two powers as sqrt() and cbrt(). Charge them like
        # the listed functions below: the surcharge plus the base, not the exponent.
        if power in (sy.Rational(1, 2), sy.Rational(1, 3)):
            return int(15 + self.complexity(base))

        c: int = 15
        if power.is_Integer:
            c = max(2, int(log2(abs(power.evalf()))))
        return int(c + sum([self.complexity(arg) for arg in n.args]))

    @complexity.register
    def _(self, n: sy.Symbol) -> int:
        return 10 if self.is_grid_variable(n) else 1

    @complexity.register
    def _(self, _n: sy.IndexedBase) -> int:
        return 1

    @complexity.register
    def _(self, _n: sy.Indexed) -> int:
        return 1

    @complexity.register
    def _(self, _n: sy.Number) -> int:
        return 1

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
        return 1

    @complexity.register
    def _(self, _n: sy.logic.boolalg.BooleanFalse) -> int:
        return 1

    def _complexity_undefined_fn(self, n: sy.Function) -> int:
        assert isinstance(n.func, UndefinedFunction)

        if n.name == 'stencil':  # type: ignore[attr-defined]
            sym, x, y, z = n.args
            assert isinstance(x, sy.Number)
            assert isinstance(y, sy.Number)
            assert isinstance(z, sy.Number)

            if y.evalf() != 0 or z.evalf() != 0:
                return 100
            elif x.evalf() != 0:
                return 40
            else:
                return 10

        args_complexity: int = sum([self.complexity(arg) for arg in n.args])
        return args_complexity

    @complexity.register
    def _(self, n: sy.Function) -> int:
        if isinstance(n.func, UndefinedFunction):
            return self._complexity_undefined_fn(n)

        args_complexity: int = sum([self.complexity(arg) for arg in n.args])

        # Compare the function class. `n in [sy.sin, ...]` compares the
        # applied call to those classes and never matches, so the +15
        # surcharge was never applied. sqrt and cbrt are matched in the Pow
        # handler, since SymPy never builds them as Function calls.
        if n.func in _TRANSCENDENTAL_FUNCTIONS:
            return 15 + args_complexity
        else:
            return args_complexity
