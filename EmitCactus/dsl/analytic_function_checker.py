from typing import Set, Tuple, List, Dict, Optional
from EmitCactus.dsl.sympywrap import *
import sympy as sy
from sympy import Symbol
from multimethod import multimethod

x = mkSymbol("x")
y = mkSymbol("y")
z = mkSymbol("z")
t = mkSymbol("t")

class AnalyticFunctionChecker:
    def __init__(self, params:Set[Symbol], eqns:Dict[Symbol, sy.Expr])->None:
        self.eqns:Dict[Symbol, sy.Expr] = eqns
        self.is_analytic:Dict[Symbol, bool] = dict()
        self.exc:Set[Symbol] = set()
        for k in [x, y, z, t]:
            self.is_analytic[k] = True
            self.exc.add(k)
        for k in params:
            self.is_analytic[k] = True
            self.exc.add(k)

    def analytic(self)->Set[Symbol]:
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
    def visit(self, a:sy.Symbol)->Optional[bool]:
        return self.is_analytic.get(a, None)

    @visit.register
    def _(self, a:sy.Number)->Optional[bool]:
        return True

    @visit.register
    def _(self, a:sy.NumberSymbol)->Optional[bool]:
        return True

    @visit.register
    def _(self, a:sy.Add|sy.Mul|sy.Function|sy.Pow)->Optional[bool]:
        for arg in a.args:
            b = self.visit(arg)
            assert b is None or type(b) == bool
            if b != True:
                return b
        return True

if __name__ == "__main__":
    u = mkSymbol("u")
    t1 = mkSymbol("t1")
    p = mkSymbol("p")

    b = AnalyticFunctionChecker({p}, {t1:3+p, u:t1*2+9})
    ans = b.analytic()
    assert ans == {t1,u}, f"ans = {ans}"

    a = AnalyticFunctionChecker({p}, {t:3+p, u:t*2+9})
    assert not a.visit(u)
    assert a.visit(x)
    assert a.visit(sin(x))
    assert a.visit(sin(x)**2)
    assert a.visit(sin(x)**2+y)
    assert not a.visit(sin(u)**2+y)
