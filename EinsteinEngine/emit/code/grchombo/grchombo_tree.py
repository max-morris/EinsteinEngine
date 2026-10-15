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
    """A read of a state variable at a stencil offset from the current cell: in_<var>[idx + offset]."""
    var: str
    offset: StencilIdx


@dataclass
class PointerDecl(Stmt):
    """const auto in_<var> = SIMDIFY<data_t>(box_pointers.m_in_ptr[<enum>]);"""
    var: str
    enum: str


@dataclass
class TempDecl(Stmt):
    """A local temporary. `reassign`: it reuses (recycles) an earlier temporary's variable."""
    name: str
    expr: Expr
    reassign: bool
    mutable: bool
    comment: Optional[str] = None


@dataclass
class StoreVar(Stmt):
    """current_cell.store_vars(<expr>, <enum>);"""
    enum: str
    expr: Expr
    comment: Optional[str] = None


@dataclass
class ComputeClassParam(CodeNode):
    name: str
    cpp_type: str
    init: str


@dataclass
class ComputeClassDecl(CodeNode):
    name: str
    namespace: str
    includes: Sequence[str]
    ctor_args: Sequence[tuple[str, str, Optional[str]]]  # (type, name, default)
    dx: str
    ctor_checks: Sequence[tuple[str, str]]  # (failure condition, message)
    params: Sequence[ComputeClassParam]
    body: Sequence[Stmt]
    banner: str
