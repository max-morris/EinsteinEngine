from typing import List, Dict, cast, Optional

from multimethod import multimethod
from sympy import Expr, Mul, Add, Symbol
from sympy.core.numbers import NegativeOne, One, Integer

from EmitCactus.dsl.eqnlist import EqnList
from EmitCactus.dsl.sympywrap import *

madd = mkFunction("muladd")


class Maddifier:
    eqn_list: EqnList
    recency_threshold: Optional[int]

    def __init__(self, eqn_list: EqnList, *, recency_threshold: Optional[int] = None) -> None:
        self.eqn_list = eqn_list
        self.recency_threshold = recency_threshold

    def maddify_in_place(self) -> None:
        self.eqn_list.eqns = self.maddify()

    def maddify(self) -> Dict[Symbol, Expr]:
        sorted_eqns = sorted(self.eqn_list.eqns.items(), key=lambda kv: self.eqn_list.order.index(kv[0]))
        return {sym: self._visit(rhs, idx) for idx, (sym, rhs) in enumerate(sorted_eqns)}

    @staticmethod
    def _get_mul(expr: Expr) -> Expr:
        if len(expr.args) == 2:
            v = expr.args[1]
        else:
            v = Mul(*expr.args[1:])
        assert isinstance(v, Expr)
        return v

    def _should_maddify(self, arg: Expr, idx: int) -> bool:
        if isinstance(arg, One) or isinstance(arg, NegativeOne):
            return False

        if isinstance(arg, Integer):
            return arg.p not in [1, -1, 2, -2]

        if not any(symbol_dependencies := free_symbols(arg)):
            return False

        if self.recency_threshold is None:
            return True

        if not any(temporary_dependencies := [s for s in symbol_dependencies if s in self.eqn_list.temporaries]):
            return True

        oldest_dependency = min(map(lambda s: self.eqn_list.order.index(s), temporary_dependencies))
        return idx - oldest_dependency <= self.recency_threshold

    @multimethod
    def _visit(self, expr: Add, idx: int) -> Expr:
        if not self._should_maddify(expr, idx):
            return expr

        new_args: List[Expr] = list()
        new_args.append(expr.args[0])
        for i in range(1, len(expr.args)):
            prev = new_args[-1]
            curr = expr.args[i]
            if isinstance(prev, Mul) and self._should_maddify(arg2 := self._get_mul(prev), idx):
                arg1 = prev.args[0]
                new_args[-1] = madd(self._visit(arg1, idx), self._visit(arg2, idx), curr)
            elif isinstance(curr, Mul) and self._should_maddify(arg2 := self._get_mul(curr), idx):
                arg1 = curr.args[0]
                new_args[-1] = madd(self._visit(arg1, idx), self._visit(arg2, idx), prev)
            else:
                new_args.append(curr)
        if len(new_args) == 1:
            return new_args[0]
        else:
            v: Expr = Add(*new_args)
            return v

    @_visit.register
    def _(self, expr: Mul, idx: int) -> Expr:
        if not self._should_maddify(expr, idx):
            return expr

        return cast(Expr, Mul(*[self._visit(arg, idx) for arg in expr.args]))

    @_visit.register
    def _(self, expr: Expr, _idx: int) -> Expr:
        return expr
