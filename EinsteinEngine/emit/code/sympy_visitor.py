#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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

import typing
from typing import List, Optional, cast

# noinspection PyUnresolvedReferences
import sympy as sy
from multimethod import multimethod
from sympy.logic.boolalg import Boolean

from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.common.stencil_idx import StencilIdx
from EinsteinEngine.emit.code.common.code_tree import NArityOpExpr, Expr, BinOp, UnOpExpr, UnOp, BinOpExpr, IdExpr, FunctionCall, \
    StandardizedFunctionCallType, StandardizedFunctionCall, IntLiteralExpr, FloatLiteralExpr, IfElseExpr
from EinsteinEngine.emit.tree import Identifier


class BaseSympyExprVisitor[ExprT: Expr]:
    visiting_stencil_fn_args: bool
    stencil_fns: set[str]

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

    def __init__(
            self,
            *,
            stencil_fns: Optional[set[str]] = None
    ):
        self.stencil_fns = stencil_fns if stencil_fns is not None else set()
        self.visiting_stencil_fn_args = False

    @multimethod
    def visit(self, expr: sy.Basic) -> Expr:
        raise NotImplementedError(f'visit({expr.func}) not implemented in BaseSympyExprVisitor expr={expr}')

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
        return self._visit_symbol(self._sanitize_symbol_name(expr.name))

    @visit.register
    def _(self, expr: sy.IndexedBase) -> Expr:
        base, tup = expr.args
        assert len(tup.args) == 0, f"Missing arguments on symbol: {str(expr)} {tup.args} {len(tup.args)}"
        return typing.cast(Expr, self.visit(base))

    def _sanitize_symbol_name(self, name: str) -> str:
        return name.replace("'", "")

    def _visit_symbol(self, symbol_name: str) -> Expr:
        return IdExpr(Identifier(symbol_name))

    def _visit_stencil_access(
            self,
            *,
            expr: sy.Function,
            stencil_idx: StencilIdx
    ) -> Expr:
        gf_arg = self.visit(expr.args[0])
        return FunctionCall(
            Identifier(expr.func.name),
            [gf_arg, IntLiteralExpr(stencil_idx.x), IntLiteralExpr(stencil_idx.y), IntLiteralExpr(stencil_idx.z)],
            []
        )

    @staticmethod
    def _to_stencil_offset(v: sy.Basic) -> int:
        if isinstance(v, sy.Integer):
            return int(v)

        # noinspection PyTypeChecker
        return int(cast(sy.Expr, v).evalf())  # type: ignore[no-untyped-call]

    def _visit_stencil_call(self, expr: sy.Function) -> Expr:
        if len(expr.args) != 4:
            raise DslException(f"Stencil call `{expr}` should have 4 args (gf, i, j, k), got {len(expr.args)}.")

        x_arg, y_arg, z_arg = expr.args[1:]
        stencil_idx = StencilIdx(
            self._to_stencil_offset(x_arg),
            self._to_stencil_offset(y_arg),
            self._to_stencil_offset(z_arg)
        )

        self.visiting_stencil_fn_args = True
        try:
            return self._visit_stencil_access(
                expr=expr,
                stencil_idx=stencil_idx
            )
        finally:
            self.visiting_stencil_fn_args = False

    @visit.register
    def _(self, expr: sy.Function) -> Expr:
        arg_list: list[Expr]

        if isinstance(expr.func, sy.core.function.UndefinedFunction):  # Undefined function calls are preserved as-is
            assert hasattr(expr.func, 'name')
            if expr.func.name in self.stencil_fns:
                return self._visit_stencil_call(expr)
            else:
                arg_list = [self.visit(a) for a in expr.args]
                return FunctionCall(Identifier(expr.func.name), arg_list, [])

        # If we're here, the function is some sort of standard mathematical function (e.g., sin, cos)
        fn_type: StandardizedFunctionCallType

        if expr.func in self.standard_fns:
            fn_type = self.standard_fns[expr.func]
        else:
            raise NotImplementedError(f"visit({expr.func}) not implemented in BaseSympyExprVisitor")

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
