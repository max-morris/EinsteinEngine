from math import log2
from typing import cast, Protocol

import sympy as sy
from multimethod import multimethod
from sympy.core.function import UndefinedFunction


class IsGridVariableFn(Protocol):
    def __call__(self, symbol: sy.Symbol, /) -> bool: ...


def calculate_complexity(expr: sy.Basic, *, is_grid_variable: IsGridVariableFn) -> int:
    return cast(int, SympyComplexityVisitor(is_grid_variable).complexity(expr))


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
        c: int = 15
        if (power := n.args[1]).is_Integer:
            c = max(2, int(log2(abs(power.evalf()))))  # type: ignore[no-untyped-call]
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

    def _complexity_undefined_fn(self, n: sy.Function) -> int:
        assert isinstance(n.func, UndefinedFunction)

        if n.name == 'stencil':  # type: ignore[attr-defined]
            sym, x, y, z = n.args
            assert isinstance(x, sy.Number)
            assert isinstance(y, sy.Number)
            assert isinstance(z, sy.Number)

            if y.evalf() != 0 or z.evalf() != 0:  # type: ignore[no-untyped-call]
                return 100
            elif x.evalf() != 0:  # type: ignore[no-untyped-call]
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

        if n in [sy.sin, sy.cos, sy.exp, sy.log, sy.sqrt, sy.cbrt]:
            return 15 + args_complexity
        else:
            return args_complexity
