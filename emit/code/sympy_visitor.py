import typing
from typing import Any

# noinspection PyUnresolvedReferences
import sympy as sy
from multimethod import multimethod

from emit.code.code_tree import *
from emit.tree import *


class SympyExprVisitor:
    @multimethod
    def visit(self, expr: sy.Basic) -> Expr:
        raise NotImplementedError(f'visit({expr.func}) not implemented in SympyVisitor')

    @visit.register
    def _(self, expr: sy.Add) -> Expr:
        return NArityOpExpr(Operator.Add, [self.visit(a) for a in expr.args])

    @visit.register
    def _(self, expr: sy.Mul) -> Expr:
        return NArityOpExpr(Operator.Mul, [self.visit(a) for a in expr.args])

    @visit.register
    def _(self, expr: sy.Pow) -> Expr:
        lhs, rhs = expr.args
        return BinOpExpr(self.visit(lhs), Operator.Pow, self.visit(rhs))

    @visit.register
    def _(self, expr: sy.Symbol) -> Expr:
        assert len(expr.args) == 0
        return IdExpr(Identifier(expr.name))

    @visit.register
    def _(self, expr: sy.IndexedBase) -> Expr:
        base, tup = expr.args
        assert len(tup.args) == 0
        return typing.cast(Expr, self.visit(base))

    @visit.register
    def _(self, expr: sy.Function) -> Expr:
        return FunctionCall(Identifier(expr.func.name), [self.visit(a) for a in expr.args], [])
