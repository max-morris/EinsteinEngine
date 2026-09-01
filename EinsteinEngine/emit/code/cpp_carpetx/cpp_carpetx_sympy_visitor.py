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

from typing import Optional, Callable

import sympy as sy
from sympy.core.function import UndefinedFunction as UFunc

from EinsteinEngine import Identifier, as_f64, as_f32, as_f16
from EinsteinEngine.common.stencil_idx import StencilIdxWithCentering, StencilIdx
from EinsteinEngine.emit.code.common.code_tree import Expr, IdExpr, FunctionCall
from EinsteinEngine.emit.code.sympy_visitor import BaseSympyExprVisitor
from EinsteinEngine.emit.util import encode_stencil_idx
from EinsteinEngine.frontend.util import require
from EinsteinEngine.generators.util import SymbolInStencilArgsPredicate, VarCenteringFn
from EinsteinEngine.emit.code.cpp_carpetx.cpp_carpetx_tree import CppCarpetXExprNode, StaticCast


class CppCarpetXSympyVisitor(BaseSympyExprVisitor[CppCarpetXExprNode]):
    should_wrap_with_access_fn: SymbolInStencilArgsPredicate
    centering_fn: VarCenteringFn

    numeric_conversion_rewrite: dict[UFunc, Callable[[CppCarpetXExprNode], CppCarpetXExprNode]] = {
        as_f64: lambda expr: StaticCast(Identifier("CCTK_REAL8"), expr),
        as_f32: lambda expr: StaticCast(Identifier("CCTK_REAL4"), expr),
        as_f16: lambda expr: StaticCast(Identifier("CCTK_REAL2"), expr)
    }

    def __init__(
            self,
            *,
            stencil_fns: Optional[set[str]] = None,
            should_wrap_with_access_fn: Optional[SymbolInStencilArgsPredicate] = None,
            centering_fn: Optional[VarCenteringFn] = None
    ):
        super().__init__(stencil_fns=stencil_fns)
        self.should_wrap_with_access_fn = should_wrap_with_access_fn if should_wrap_with_access_fn is not None else lambda _0, _1: False
        self.centering_fn = centering_fn if centering_fn is not None else lambda _: None

    def _visit_symbol(self, symbol_name: str) -> Expr:
        if not self.should_wrap_with_access_fn(symbol_name, self.visiting_stencil_fn_args):
            return IdExpr(Identifier(symbol_name))

        centering = require(self.centering_fn(symbol_name), lambda: f'Unknown centering for variable {symbol_name}')
        encoded_idx = encode_stencil_idx(StencilIdxWithCentering(StencilIdx(0, 0, 0), centering))
        return FunctionCall(
            Identifier("access"),
            [IdExpr(Identifier(symbol_name)), IdExpr(Identifier(encoded_idx))],
            []
        )

    def _visit_stencil_access(
            self,
            *,
            expr: sy.Function,
            stencil_idx: StencilIdx
    ) -> Expr:
        gf_arg = self.visit(expr.args[0])
        var_name = self._sanitize_symbol_name(str(expr.args[0]))
        centering = require(
            self.centering_fn(var_name),
            lambda: f"Stencil call references variable `{var_name}` without a defined centering."
        )
        args_encoded = encode_stencil_idx(StencilIdxWithCentering(stencil_idx, centering))
        return FunctionCall(Identifier(expr.func.name), [gf_arg, IdExpr(Identifier(args_encoded))], [])
