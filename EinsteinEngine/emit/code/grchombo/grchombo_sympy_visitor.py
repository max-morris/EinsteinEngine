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
from EinsteinEngine.emit.code.common.code_tree import Expr, IdExpr
from EinsteinEngine.emit.code.grchombo.grchombo_tree import GridLoad
from EinsteinEngine.emit.code.sympy_visitor import BaseSympyExprVisitor
from EinsteinEngine.emit.tree import Identifier


class GRChomboSympyVisitor(BaseSympyExprVisitor[Expr]):
    """
    Lowers grid-variable reads to GridLoads at the stencil offset (0 for a plain symbol), and |x| to the header's
    ee_abs helper (std::abs for double, GRChombo's simd abs overload for simd<double>).
    """
    is_grid_var: Callable[[str], bool]

    standard_fns_rewrite = {
        **BaseSympyExprVisitor.standard_fns_rewrite,
        sy.Abs: lambda a, *_: sy.Function('ee_abs')(a),  # type: ignore[no-untyped-call]
    }

    def __init__(self, *, stencil_fns: Optional[set[str]] = None, is_grid_var: Callable[[str], bool]):
        super().__init__(stencil_fns=stencil_fns)
        self.is_grid_var = is_grid_var

    def _visit_symbol(self, symbol_name: str) -> Expr:
        if self.is_grid_var(symbol_name) and not self.visiting_stencil_fn_args:
            return GridLoad(symbol_name, StencilIdx(0, 0, 0))
        return IdExpr(Identifier(symbol_name))

    def _visit_stencil_access(self, *, expr: sy.Function, stencil_idx: StencilIdx) -> Expr:
        return GridLoad(self._sanitize_symbol_name(str(expr.args[0])), stencil_idx)
