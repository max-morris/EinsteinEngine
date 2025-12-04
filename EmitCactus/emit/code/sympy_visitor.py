import typing
from typing import List, Optional

# noinspection PyUnresolvedReferences
import sympy as sy
from multimethod import multimethod
from sympy.logic.boolalg import Boolean

from EmitCactus.dsl.dsl_exception import DslException
from EmitCactus.emit.code.code_tree import NArityOpExpr, Expr, BinOp, UnOpExpr, UnOp, BinOpExpr, IdExpr, FunctionCall, \
    StandardizedFunctionCallType, StandardizedFunctionCall, IntLiteralExpr, FloatLiteralExpr, IfElseExpr
from EmitCactus.emit.tree import Identifier
from EmitCactus.generators.util import SympyNameSubstitutionFn

class SympyExprVisitor:
    substitution_fn: SympyNameSubstitutionFn
    visiting_stencil_fn_args: bool
    stencil_fns: List[str]

    standard_fns: dict[sy.Function, StandardizedFunctionCallType] = {
        sy.sin: StandardizedFunctionCallType.Sin,
        sy.cos: StandardizedFunctionCallType.Cos,
        sy.tan: StandardizedFunctionCallType.Tan,
        sy.cot: StandardizedFunctionCallType.Cot,
        sy.sec: StandardizedFunctionCallType.Sec,
        sy.csc: StandardizedFunctionCallType.Csc,
        sy.sinh: StandardizedFunctionCallType.Sinh,
        sy.cosh: StandardizedFunctionCallType.Cosh,
        sy.tanh: StandardizedFunctionCallType.Tanh,
        sy.coth: StandardizedFunctionCallType.Coth,
        sy.sech: StandardizedFunctionCallType.Sech,
        sy.csch: StandardizedFunctionCallType.Csch,
        sy.exp: StandardizedFunctionCallType.Exp,
        sy.erf: StandardizedFunctionCallType.Erf,
        sy.log: StandardizedFunctionCallType.Log
    }

    def __init__(self, *, stencil_fns: Optional[List[str]] = None, substitution_fn: Optional[SympyNameSubstitutionFn] = None):
        self.substitution_fn = substitution_fn if substitution_fn is not None else lambda s, _: s
        self.stencil_fns = stencil_fns if stencil_fns is not None else list()
        self.visiting_stencil_fn_args = False

    @multimethod
    def visit(self, expr: sy.Basic) -> Expr:
        raise NotImplementedError(f'visit({expr.func}) not implemented in SympyExprVisitor expr={expr}')

    @visit.register
    def _(self, expr: sy.Add) -> Expr:
        return NArityOpExpr(BinOp.Add, [self.visit(a) for a in expr.args])

    @visit.register
    def _(self, expr: sy.Mul) -> Expr:
        visited_args: List[Expr] = [self.visit(a) for a in expr.args]

        if len(visited_args) == 2:
            # noinspection PyUnresolvedReferences
            if isinstance(visited_args[0], IntLiteralExpr) and visited_args[0].integer == -1:
                return UnOpExpr(UnOp.Neg, visited_args[1])
            elif isinstance(visited_args[1], IntLiteralExpr) and visited_args[1].integer == -1:
                return UnOpExpr(UnOp.Neg, visited_args[0])

        return NArityOpExpr(BinOp.Mul, visited_args)

    @visit.register
    def _(self, expr: sy.Pow) -> Expr:
        lhs, rhs = expr.args
        return BinOpExpr(self.visit(lhs), BinOp.Pow, self.visit(rhs))

    @visit.register
    def _(self, expr: sy.LessThan) -> Expr:
        lhs, rhs = expr.args
        return BinOpExpr(self.visit(lhs), BinOp.Lte, self.visit(rhs))

    @visit.register
    def _(self, expr: sy.GreaterThan) -> Expr:
        lhs, rhs = expr.args
        return BinOpExpr(self.visit(lhs), BinOp.Gte, self.visit(rhs))

    @visit.register
    def _(self, expr: sy.StrictLessThan) -> Expr:
        lhs, rhs = expr.args
        return BinOpExpr(self.visit(lhs), BinOp.Lt, self.visit(rhs))

    @visit.register
    def _(self, expr: sy.StrictGreaterThan) -> Expr:
        lhs, rhs = expr.args
        return BinOpExpr(self.visit(lhs), BinOp.Gt, self.visit(rhs))

    @visit.register
    def _(self, expr: sy.Equality) -> Expr:
        lhs, rhs = expr.args
        return BinOpExpr(self.visit(lhs), BinOp.Eq, self.visit(rhs))

    @visit.register
    def _(self, expr: sy.Unequality) -> Expr:
        lhs, rhs = expr.args
        return BinOpExpr(self.visit(lhs), BinOp.Neq, self.visit(rhs))

    @visit.register
    def _(self, expr: sy.Symbol) -> Expr:
        assert len(expr.args) == 0
        return IdExpr(Identifier(self.substitution_fn(expr.name, self.visiting_stencil_fn_args)))

    @visit.register
    def _(self, expr: sy.IndexedBase) -> Expr:
        base, tup = expr.args
        assert len(tup.args) == 0, f"Missing arguments on symbol: {str(expr)} {tup.args} {len(tup.args)}"
        return typing.cast(Expr, self.visit(base))

    @visit.register
    def _(self, expr: sy.Function) -> Expr:
        arg_list: list[Expr]

        if isinstance(expr.func, sy.core.function.UndefinedFunction):  # Undefined function calls are preserved as-is
            assert hasattr(expr.func, 'name')
            if expr.func.name in self.stencil_fns:
                self.visiting_stencil_fn_args = True
            arg_list = [self.visit(a) for a in expr.args]
            self.visiting_stencil_fn_args = False
            return FunctionCall(Identifier(expr.func.name), arg_list, [])

        # If we're here, the function is some sort of standard mathematical function (e.g., sin, cos)
        fn_type: StandardizedFunctionCallType

        if expr.func in self.standard_fns:
            fn_type = self.standard_fns[expr.func]
        else:
            raise NotImplementedError(f"visit({expr.func}) not implemented in SympyExprVisitor")

        arg_list = [self.visit(a) for a in expr.args]
        return StandardizedFunctionCall(fn_type, arg_list)

    def _visit_piecewise(self, expr: sy.Piecewise, i: int = 0) -> Expr:
        piecewise_args = typing.cast(tuple[tuple[sy.Expr, Boolean]], expr.args)
        i_expr: sy.Expr
        i_cond: Boolean
        i_expr, i_cond = piecewise_args[i]

        if i_cond == sy.S.true:
            return typing.cast(Expr, self.visit(i_expr))

        return IfElseExpr(
            self.visit(i_cond),
            self.visit(i_expr),
            self._visit_piecewise(expr, i + 1)
        )

    @visit.register
    def _(self, expr: sy.Piecewise) -> Expr:
        piecewise_args = typing.cast(tuple[tuple[sy.Expr, Boolean]], expr.args)

        if len(piecewise_args) == 0:
            raise DslException(f"Piecewise function has no arguments: {expr}")

        if piecewise_args[-1][1] != sy.S.true:
            raise DslException(f"Piecewise function is incomplete (does not end with a True condition): {expr}")

        return self._visit_piecewise(expr)

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

    @visit.register
    def _(self, expr: sy.core.numbers.Float) -> Expr:
        return FloatLiteralExpr(expr.n()) # type: ignore[no-untyped-call]

    @visit.register
    def _(self, expr: sy.core.numbers.Pi) -> Expr:
        return FloatLiteralExpr(expr.n())  # type: ignore[no-untyped-call]

    @visit.register
    def _(self, expr: sy.core.numbers.Rational) -> Expr:
        # Cast to floats to avoid floor division in e.g. C++
        return BinOpExpr(FloatLiteralExpr(float(expr.p)), BinOp.Div, FloatLiteralExpr(float(expr.q)))
