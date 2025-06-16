from typing import List, Dict, cast

from multimethod import multimethod
from sympy import Expr, Mul, Add, Symbol
from sympy.core.numbers import NegativeOne, One, Integer

from EmitCactus.dsl.eqnlist import EqnList
from EmitCactus.dsl.sympywrap import *

madd = mkFunction("muladd")

class Maddifier:
    eqn_list: EqnList

    def __init__(self, eqn_list: EqnList) -> None:
        self.eqn_list = eqn_list

    def maddify_in_place(self) -> None:
        self.eqn_list.eqns = self.maddify()

    def maddify(self) -> Dict[Symbol, Expr]:
        return {sym: self._visit(rhs) for sym, rhs in self.eqn_list.eqns.items()}

    @staticmethod
    def _get_mul(expr: Expr) -> Expr:
        if len(expr.args) == 2:
            v = expr.args[1]
        else:
            v = Mul(*expr.args[1:])
        assert isinstance(v, Expr)
        return v

    @staticmethod
    def _avoid(arg: Expr) -> bool:
        if isinstance(arg, One) or isinstance(arg, NegativeOne):
            return True

        if isinstance(arg, Integer):
            return arg.p in [1, -1, 2, -2]

        return False

    @multimethod
    def _visit(self, add: Add) -> Expr:
        new_args: List[Expr] = list()
        new_args.append(add.args[0])
        for i in range(1, len(add.args)):
            prev = new_args[-1]
            curr = add.args[i]
            if isinstance(prev, Mul) and not self._avoid(arg2 := self._get_mul(prev)):
                arg1 = prev.args[0]
                new_args[-1] = madd(self._visit(arg1), self._visit(arg2), curr)
            elif isinstance(curr, Mul) and not self._avoid(arg2 := self._get_mul(curr)):
                arg1 = curr.args[0]
                new_args[-1] = madd(self._visit(arg1), self._visit(arg2), prev)
            else:
                new_args.append(curr)
        if len(new_args) == 1:
            return new_args[0]
        else:
            v: Expr = Add(*new_args)
            return v

    @_visit.register
    def _(self, expr: Mul) -> Expr:
        return cast(Expr, Mul(*[self._visit(arg) for arg in expr.args]))

    @_visit.register
    def _(self, expr: Expr) -> Expr:
        return expr

