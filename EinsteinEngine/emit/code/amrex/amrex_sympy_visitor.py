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

from typing import Callable, Optional

import sympy as sy

from EinsteinEngine.common.stencil_idx import StencilIdx
from EinsteinEngine.emit.code.common.code_tree import Expr, IdExpr, FunctionCall, BinOpExpr, BinOp
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
    """
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
