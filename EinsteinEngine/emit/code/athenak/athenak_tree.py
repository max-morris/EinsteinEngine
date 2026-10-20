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

from dataclasses import dataclass
from typing import Optional, Sequence

from EinsteinEngine.common.stencil_idx import StencilIdx
from EinsteinEngine.emit.code.common.code_tree import Expr, Stmt, CodeNode


@dataclass
class GridLoad(Expr):
    """A read of a grid variable at a stencil offset from the current cell: view(m, enum, k+z, j+y, i+x)."""
    view: str
    enum: str
    offset: StencilIdx


@dataclass
class TempDecl(Stmt):
    """A local temporary. `reassign`: it reuses (recycles) an earlier temporary's variable."""
    name: str
    expr: Expr
    reassign: bool
    mutable: bool
    comment: Optional[str] = None


@dataclass
class GridStore(Stmt):
    """view(m, enum, k, j, i) = expr;"""
    view: str
    enum: str
    expr: Expr
    comment: Optional[str] = None


@dataclass
class KernelLoop(CodeNode):
    """One par_for over (m, k, j, i)."""
    label: str
    loop_bounds: str
    inv_spacing: tuple[str, str, str]
    body: Sequence[Stmt]


@dataclass
class KernelParam(CodeNode):
    name: str
    cpp_type: str
    init: str


@dataclass
class KernelFunctionDecl(CodeNode):
    name: str
    namespace: str
    includes: Sequence[str]
    args: Sequence[tuple[str, str]]
    preamble: Sequence[str]
    checks: Sequence[tuple[str, str]]  # (failure condition, message)
    params: Sequence[KernelParam]
    loops: Sequence[KernelLoop]
    banner: str
