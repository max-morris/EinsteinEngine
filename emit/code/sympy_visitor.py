import typing
from typing import Callable, List, Optional

# noinspection PyUnresolvedReferences
import sympy as sy
from multimethod import multimethod

from emit.code.code_tree import *
from emit.tree import *


class SympyExprVisitor:
    substitution_fn: Callable[[str, bool], str]
    visiting_stencil_fn_args: bool
    stencil_fns: List[str]

    def __init__(self, *, stencil_fns: Optional[List[str]] = None, substitution_fn: Optional[Callable[[str, bool], str]] = None):
        self.substitution_fn = substitution_fn if substitution_fn is not None else lambda s, _: s
        self.stencil_fns = stencil_fns if stencil_fns is not None else list()
        self.visiting_stencil_fn_args = False

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
        return IdExpr(Identifier(self.substitution_fn(expr.name, self.visiting_stencil_fn_args)))

    @visit.register
    def _(self, expr: sy.IndexedBase) -> Expr:
        base, tup = expr.args
        assert len(tup.args) == 0, f"This has args! {str(expr)} {tup.args} {len(tup.args)}"
        return typing.cast(Expr, self.visit(base))

    @visit.register
    def _(self, expr: sy.Function) -> Expr:
        arg_list: list[Expr]

        if isinstance(expr.func, sy.core.function.UndefinedFunction):  # Undefined function calls are preserved as-is
            if expr.func.name in self.stencil_fns:  # type: ignore[attr-defined]
                self.visiting_stencil_fn_args = True
            arg_list = [self.visit(a) for a in expr.args]
            self.visiting_stencil_fn_args = False
            return FunctionCall(Identifier(expr.func.name), arg_list, [])  # type: ignore[attr-defined]

        # If we're here, the function is some sort of standard mathematical function (e.g., sin, cos)
        fn_type: StandardizedFunctionCallType

        if isinstance(expr, sy.sin):
            fn_type = StandardizedFunctionCallType.Sin
        elif isinstance(expr, sy.cos):
            fn_type = StandardizedFunctionCallType.Cos
        else:
            raise NotImplementedError(f"visit({type(expr)}) not implemented in SympyExprVisitor")

        arg_list = [self.visit(a) for a in expr.args]
        return StandardizedFunctionCall(fn_type, arg_list)

    @visit.register
    def _(self, _: sy.core.numbers.Zero) -> Expr:
        return IntLiteralExpr(0)

    @visit.register
    def _(self, _: sy.core.numbers.One) -> Expr:
        return IntLiteralExpr(1)

    @visit.register
    def _(self, _: sy.core.numbers.NegativeOne) -> Expr:
        return IntLiteralExpr(-1)

    @visit.register
    def _(self, expr: sy.core.numbers.Integer) -> Expr:
        return IntLiteralExpr(expr.p)
