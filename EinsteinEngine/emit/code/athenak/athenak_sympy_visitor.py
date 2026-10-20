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
from EinsteinEngine.emit.code.athenak.athenak_tree import GridLoad
from EinsteinEngine.emit.code.sympy_visitor import BaseSympyExprVisitor
from EinsteinEngine.emit.tree import Identifier


class AthenaKSympyVisitor(BaseSympyExprVisitor[Expr]):
    """
    Lowers grid-variable reads to GridLoads at the stencil offset (0 for a plain symbol), and |x| to Kokkos::fabs.
    """
    locate: Callable[[str], Optional[tuple[str, str]]]

    standard_fns_rewrite = {
        **BaseSympyExprVisitor.standard_fns_rewrite,
        sy.Abs: lambda a, *_: sy.Function('Kokkos::fabs')(a),  # type: ignore[no-untyped-call]
    }

    def __init__(self, *, stencil_fns: Optional[set[str]] = None,
                 locate: Callable[[str], Optional[tuple[str, str]]]):
        super().__init__(stencil_fns=stencil_fns)
        self.locate = locate

    def _visit_symbol(self, symbol_name: str) -> Expr:
        if not self.visiting_stencil_fn_args and (loc := self.locate(symbol_name)) is not None:
            return GridLoad(loc[0], loc[1], StencilIdx(0, 0, 0))
        return IdExpr(Identifier(symbol_name))

    def _visit_stencil_access(self, *, expr: sy.Function, stencil_idx: StencilIdx) -> Expr:
        name = self._sanitize_symbol_name(str(expr.args[0]))
        loc = self.locate(name)
        if loc is None:
            raise ValueError(f"Stencil access to {name}, which is not a grid variable")
        return GridLoad(loc[0], loc[1], stencil_idx)
