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

from typing import Optional

import sympy as sy

from EinsteinEngine import Identifier
from EinsteinEngine.common.stencil_idx import StencilIdx
from EinsteinEngine.emit.code.common.code_tree import Expr, IdExpr, BinOpExpr, BinOp, IntLiteralExpr
from EinsteinEngine.emit.code.f90.f90_tree import ArrayAccess
from EinsteinEngine.emit.code.sympy_visitor import BaseSympyExprVisitor
from EinsteinEngine.generators.util import SymbolInStencilArgsPredicate
from EinsteinEngine.emit.code.f90.f90_tree import F90ExprNode


class F90SympyVisitor(BaseSympyExprVisitor[F90ExprNode]):
    should_inject_array_access: SymbolInStencilArgsPredicate
    i_name: str
    j_name: str
    k_name: str

    def __init__(
            self,
            *,
            stencil_fns: Optional[set[str]] = None,
            should_inject_array_access: Optional[SymbolInStencilArgsPredicate] = None,
            i_name: str = "i",
            j_name: str = "j",
            k_name: str = "k"
    ):
        super().__init__(stencil_fns=stencil_fns)
        self.should_inject_array_access = should_inject_array_access if should_inject_array_access is not None else lambda _0, _1: False
        self.i_name = i_name
        self.j_name = j_name
        self.k_name = k_name

    @staticmethod
    def _with_offset(base_name: str, offset: int) -> F90ExprNode:
        base = IdExpr(Identifier(base_name))
        if offset == 0:
            return base
        elif offset < 0:
            return BinOpExpr(base, BinOp.Sub, IntLiteralExpr(-offset))
        return BinOpExpr(base, BinOp.Add, IntLiteralExpr(offset))

    def _visit_symbol(self, symbol_name: str) -> F90ExprNode:
        if not self.should_inject_array_access(symbol_name, self.visiting_stencil_fn_args):
            return IdExpr(Identifier(symbol_name))

        return ArrayAccess(
            Identifier(symbol_name),
            [
                IdExpr(Identifier(self.i_name)),
                IdExpr(Identifier(self.j_name)),
                IdExpr(Identifier(self.k_name))
            ]
        )

    def _visit_stencil_access(
            self,
            *,
            expr: sy.Function,
            stencil_idx: StencilIdx
    ) -> F90ExprNode:
        var_name = self._sanitize_symbol_name(str(expr.args[0]))
        return ArrayAccess(
            Identifier(var_name),
            [
                self._with_offset(self.i_name, stencil_idx.x),
                self._with_offset(self.j_name, stencil_idx.y),
                self._with_offset(self.k_name, stencil_idx.z)
            ]
        )
