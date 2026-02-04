from typing import Set, Any
from sympy import Symbol, Function, Indexed, IndexedBase, Idx, Basic, Expr
import sympy as sy
from EmitCactus.dsl.sympywrap import *
from multimethod import multimethod

stencil = mkFunction("stencil")
x = mkSymbol("x")
y = mkSymbol("y")
z = mkSymbol("z")
t = mkSymbol("t")

def is_stencil(x):
    if x.__class__.__name__ != "stencil":
        return False
    if len(x.args) != 4:
        return False
    if not isinstance(x.args[0], sy.Symbol) and not isinstance(x.args[0], sy.IndexedBase):
        return False
    if not isinstance(x.args[1], sy.Integer):
        return False
    if not isinstance(x.args[2], sy.Integer):
        return False
    if not isinstance(x.args[3], sy.Integer):
        return False
    return True

class FactorStencil:

    def __init__(self, inputs):
        self.inputs = inputs
        self.temps = dict()

    def mktemp(self):
        return mkSymbol(f"temp{len(self.temps)}")
    
    @multimethod
    def visit(self, a:Any)->None:
        raise Exception(f"{type(a)}: {a}")

    @visit.register
    def _(self, a:sy.NumberSymbol)->None:
        return a

    @visit.register
    def _(self, a:sy.Number)->None:
        return a

    @visit.register
    def _(self, a:IndexedBase)->None:
        if a in self.inputs and a not in [x, y, z, t]:
            return self.replace(stencil(a, 0, 0, 0))
        else:
            return a

    @visit.register
    def _(self, a:Symbol)->None:
        if a in self.inputs and a not in [x, y, z, t]:
            return self.replace(stencil(a, 0, 0, 0))
        else:
            return a

    @visit.register
    def _(self, a:stencil)->None:
        assert is_stencil(a), f"Expected stencil, got: {a}"
        return self.replace(a)

    @visit.register
    def _(self, a:sy.Function)->None:
        arglist = []
        for b in a.args:
            arglist.append(self.visit(b))
        return a.__class__(*arglist)

    @visit.register
    def _(self, a:sy.Mul)->None:
        r = sympify(1)
        for a in a.args:
            r *= self.visit(a)
        return r

    @visit.register
    def _(self, a:sy.Add)->None:
        r = sympify(0)
        for a in a.args:
            r += self.visit(a)
        return r

    @visit.register
    def _(self, a:sy.Pow)->None:
        return Pow(self.visit(a.args[0]), a.args[1])

    def replace(self, a:Any)->None:
        b = self.temps.get(a, None)
        if b is not None:
            return b
        tmp = self.mktemp()
        self.temps[tmp] = a
        return tmp

    def mktup(self, a):
        args = self.temps[a].args
        return (str(args[0]), *args[1:])

if __name__ == "__main__":
    stencil = mkFunction("stencil")
    u = mkSymbol("u")
    v = mkSymbol("v")

    msin = mkFunction("msin")
    x = mkSymbol("x")
    eqn = u #stencil(u, 1, 0, 0)*v*sin(x)+u**2
    fs = FactorStencil([u, v])
    result = fs.visit(eqn)
    print("FROM:", eqn)
    print("TO:")
    eqlist = sorted(list(fs.temps.keys()), key = lambda a: fs.mktup(a))
    for lhs in eqlist:
        rhs = fs.temps[lhs]
        print(lhs,"=",rhs)
    print(result)
