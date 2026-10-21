#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
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

import math
from fractions import Fraction
from typing import Callable, Optional

import sympy as sy

from EinsteinEngine.common.stencil_idx import StencilIdx
from EinsteinEngine.emit.code.common.code_tree import Expr, IdExpr, FunctionCall, BinOpExpr, BinOp, FloatLiteralExpr
from EinsteinEngine.emit.code.amrex.amrex_tree import FieldLoad
from EinsteinEngine.emit.code.sympy_visitor import BaseSympyExprVisitor
from EinsteinEngine.emit.tree import Identifier
from EinsteinEngine.generators.generator_exception import GeneratorException


class AMReXSympyVisitor(BaseSympyExprVisitor[Expr]):
    """
    Lowers field reads to FieldLoads at their stencil offset (0 for a plain symbol), checking the offset against the
    module's reach; every other symbol becomes an identifier named by `name_of` (point mode prefixes locals and spells
    parameters as struct members). |x| becomes amrex::Math::abs; Max and Min (not sympy Functions, so not reachable
    through the base visitor's rewrite table) become amrex::max/amrex::min with Real operands; And, Or and Not become
    &&, || and !.

    Numbers follow section 6 of the design: a Float of less than 53 bits of precision is refused (its value is a
    low-precision approximation); a Rational p/q prints as (p.0 / q.0) when |p| and |q| are at most 2^53, else as the
    correctly rounded double of p/q; pi and E print as their correctly rounded doubles, and any other constant is
    refused; so is a call of a function that is neither a stencil read nor one the backend itself writes.
    """
    MAX_EXACT = 2 ** 53
    backend_functions = {'amrex::Math::abs', 'noop'}
    is_field: Callable[[str], bool]
    name_of: Callable[[str], str]
    max_offset: int

    standard_fns_rewrite = {
        **BaseSympyExprVisitor.standard_fns_rewrite,
        sy.Abs: lambda a, *_: sy.Function('amrex::Math::abs')(a),
    }

    def __init__(self, *, stencil_fns: Optional[set[str]] = None, is_field: Callable[[str], bool],
                 name_of: Callable[[str], str] = lambda n: n, max_offset: int = 3):
        super().__init__(stencil_fns=stencil_fns)
        self.is_field = is_field
        self.name_of = name_of
        self.max_offset = max_offset

    def visit(self, expr: sy.Basic) -> Expr:
        if isinstance(expr, sy.Float):
            if expr._prec < 53:
                raise GeneratorException(f"AMReX backend: the Float {expr} has {expr._prec} bits of precision (< 53); "
                                         f"use a full-precision Float or a Rational.")
        elif isinstance(expr, sy.Integer):
            if abs(expr.p) > self.MAX_EXACT:  # an integer literal past 2^63 would wrap; past 2^53 it rounds anyway
                return FloatLiteralExpr(self._rounded(int(expr.p), 1, expr))
        elif isinstance(expr, sy.Rational):
            if abs(expr.p) > self.MAX_EXACT or abs(expr.q) > self.MAX_EXACT:
                return FloatLiteralExpr(self._rounded(int(expr.p), int(expr.q), expr))
        elif expr is sy.E:
            return FloatLiteralExpr(math.e)
        elif isinstance(expr, (sy.NumberSymbol, sy.core.numbers.NaN, sy.core.numbers.Infinity,
                               sy.core.numbers.NegativeInfinity, sy.core.numbers.ComplexInfinity,
                               sy.core.numbers.ImaginaryUnit)) and expr is not sy.pi:
            raise GeneratorException(f"AMReX backend: the constant {expr} is not supported (only pi and E).")
        elif isinstance(expr, sy.Function) and isinstance(expr.func, sy.core.function.UndefinedFunction):
            name = getattr(expr.func, 'name', '')
            if name not in self.stencil_fns and name not in self.backend_functions:
                raise GeneratorException(f"AMReX backend: call of the undefined function {name}; only the standard "
                                         f"functions of section 6 and stencil reads print.")
        if isinstance(expr, (sy.Max, sy.Min)):
            name = 'amrex::max' if isinstance(expr, sy.Max) else 'amrex::min'
            args = [self.visit(sy.Float(a) if a.is_Integer else a) for a in expr.args]  # type: ignore[no-untyped-call]
            out = args[0]
            for a in args[1:]:
                out = FunctionCall(Identifier(name), [out, a], [])
            return out
        if isinstance(expr, (sy.And, sy.Or)):
            op = BinOp.And if isinstance(expr, sy.And) else BinOp.Or
            terms = [self.visit(a) for a in expr.args]
            out = terms[0]
            for t in terms[1:]:
                out = BinOpExpr(out, op, t)
            return out
        if isinstance(expr, sy.Not):
            return FunctionCall(Identifier('!'), [self.visit(expr.args[0])], [])
        return super().visit(expr)  # type: ignore[no-any-return]

    @staticmethod
    def _rounded(p: int, q: int, expr: sy.Basic) -> float:
        """The correctly rounded double of p/q; refuses overflow, and underflow to zero of a nonzero value."""
        try:
            value = float(Fraction(p, q))
        except OverflowError:
            raise GeneratorException(f"AMReX backend: the number {expr} is not a finite double.")
        if value == 0.0 and p != 0:
            raise GeneratorException(f"AMReX backend: the number {expr} underflows to zero as a double.")
        return value

    def _visit_symbol(self, symbol_name: str) -> Expr:
        if self.is_field(symbol_name) and not self.visiting_stencil_fn_args:
            return FieldLoad(symbol_name, StencilIdx(0, 0, 0))
        return IdExpr(Identifier(self.name_of(symbol_name)))

    def _visit_stencil_access(self, *, expr: sy.Function, stencil_idx: StencilIdx) -> Expr:
        var = self._sanitize_symbol_name(str(expr.args[0]))
        if not self.is_field(var):
            raise GeneratorException(f"AMReX backend: stencil read of {var}, which is not a field.")
        if any(abs(k) > self.max_offset for k in stencil_idx):
            raise GeneratorException(f"AMReX backend: stencil read of {var} at offset {tuple(stencil_idx)} exceeds the "
                                     f"module's max_stencil_offset {self.max_offset}.")
        return FieldLoad(var, stencil_idx)
