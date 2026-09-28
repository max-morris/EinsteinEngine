#  Copyright (C) 2026 Steven R. Brandt, Max Morris, and other Einstein Engine contributors.
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

from typing import Set, Dict, Optional

import sympy as sy
from multimethod import multimethod
from sympy import Symbol

from EinsteinEngine.common.sympywrap import *

x = mk_symbol("x")
y = mk_symbol("y")
z = mk_symbol("z")
t = mk_symbol("t")


class AnalyticFunctionChecker:
    def __init__(self, params: Set[Symbol], eqns: Dict[Symbol, sy.Expr]) -> None:
        self.eqns: Dict[Symbol, sy.Expr] = eqns
        self.is_analytic: Dict[Symbol, bool] = dict()
        self.exc: Set[Symbol] = set()
        for k in [x, y, z, t]:
            self.is_analytic[k] = True
            self.exc.add(k)
        for k in params:
            self.is_analytic[k] = True
            self.exc.add(k)

    def analytic(self) -> Set[Symbol]:
        done = False
        while not done:
            done = True
            for lhs, rhs in self.eqns.items():
                if lhs in self.is_analytic:
                    continue
                a = self.visit(rhs)
                if a in [True, False]:
                    self.is_analytic[lhs] = a
                    done = False
        return set([k for k in self.is_analytic if self.is_analytic[k] == True and k not in self.exc])

    @multimethod
    def visit(self, a: sy.Symbol) -> Optional[bool]:
        return self.is_analytic.get(a, None)

    @visit.register
    def _(self, a: sy.Number) -> Optional[bool]:
        return True

    @visit.register
    def _(self, a: sy.NumberSymbol) -> Optional[bool]:
        return True

    @visit.register
    def _(self, a: sy.Add | sy.Mul | sy.Function | sy.Pow) -> Optional[bool]:
        for arg in a.args:
            b = self.visit(arg)
            assert b is None or type(b) == bool
            if b != True:
                return b
        return True

    @visit.register
    def _(self, a: sy.Piecewise) -> Optional[bool]:
        for pair in a.args:
            assert isinstance(pair, sy.core.containers.Tuple) and len(pair) == 2
            e, c = pair
            for part in (e, c):
                b = self.visit(part)
                assert b is None or type(b) == bool
                if b != True:
                    return b
        return True

    @visit.register
    def _(self, a: sy.core.relational.Relational) -> Optional[bool]:
        for arg in a.args:
            b = self.visit(arg)
            assert b is None or type(b) == bool
            if b != True:
                return b
        return True

    @visit.register
    def _(self, _a: sy.logic.boolalg.BooleanTrue | sy.logic.boolalg.BooleanFalse) -> Optional[bool]:
        return True


if __name__ == "__main__":
    u = mk_symbol("u")
    t1 = mk_symbol("t1")
    p = mk_symbol("p")

    b = AnalyticFunctionChecker({p}, {t1: 3 + p, u: t1 * 2 + 9})
    ans = b.analytic()
    assert ans == {t1, u}, f"ans = {ans}"

    a = AnalyticFunctionChecker({p}, {t: 3 + p, u: t * 2 + 9})
    assert not a.visit(u)
    assert a.visit(x)
    assert a.visit(sin(x))
    assert a.visit(sin(x) ** 2)
    assert a.visit(sin(x) ** 2 + y)
    assert not a.visit(sin(u) ** 2 + y)
