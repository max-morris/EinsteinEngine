from typing import Set, Any
from sympy import Symbol, Function, Indexed, IndexedBase, Idx, Basic, Expr
import sympy as sy
from EmitCactus.dsl.sympywrap import *
from multimethod import multimethod

class Symbify:

    @multimethod
    def visit(self, a:Any)->None:
        raise Exception(f"{type(a)}: {a}")

    @visit.register
    def _(self, a:sy.Number)->None:
        return a

    @visit.register
    def _(self, a:sy.core.numbers.Pi)->None:
        return a

    @visit.register
    def _(self, a:IndexedBase)->None:
        return a.args[0]

    @visit.register
    def _(self, a:Symbol)->None:
        return a

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
