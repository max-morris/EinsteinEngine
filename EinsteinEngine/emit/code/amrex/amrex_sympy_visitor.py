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
from EinsteinEngine.emit.code.amrex.amrex_tree import StateLoad
from EinsteinEngine.emit.code.sympy_visitor import BaseSympyExprVisitor
from EinsteinEngine.emit.tree import Identifier
from EinsteinEngine.generators.generator_exception import GeneratorException


class AMReXSympyVisitor(BaseSympyExprVisitor[Expr]):
    """
    Lowers state reads to StateLoads at the stencil offset (0 for a plain symbol); every other symbol (temporaries,
    parameters, coordinates, rhs components loaded by the function) becomes an identifier. |x| becomes AMReX's
    device-safe amrex::Math::abs.
    """
    is_state: Callable[[str], bool]

    standard_fns_rewrite = {
        **BaseSympyExprVisitor.standard_fns_rewrite,
        sy.Abs: lambda a, *_: sy.Function('amrex::Math::abs')(a),  # type: ignore[no-untyped-call]
    }

    def __init__(self, *, stencil_fns: Optional[set[str]] = None, is_state: Callable[[str], bool]):
        super().__init__(stencil_fns=stencil_fns)
        self.is_state = is_state

    def _visit_symbol(self, symbol_name: str) -> Expr:
        if self.is_state(symbol_name) and not self.visiting_stencil_fn_args:
            return StateLoad(symbol_name, StencilIdx(0, 0, 0))
        return IdExpr(Identifier(symbol_name))

    def _visit_stencil_access(self, *, expr: sy.Function, stencil_idx: StencilIdx) -> Expr:
        var = self._sanitize_symbol_name(str(expr.args[0]))
        if not self.is_state(var):
            raise GeneratorException(f"AMReX backend: stencil read of {var}, which is not a state component.")
        return StateLoad(var, stencil_idx)
