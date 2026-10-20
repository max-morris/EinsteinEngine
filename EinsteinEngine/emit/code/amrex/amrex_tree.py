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
class StateLoad(Expr):
    """A read of a state component at a stencil offset from the current cell: in_<var>[offset]."""
    var: str
    offset: StencilIdx


@dataclass
class StatePointerDecl(Stmt):
    """const amrex::Real *const in_<var> = state_ptr + <enum> * sn;"""
    var: str
    enum: str


@dataclass
class RhsLoad(Stmt):
    """const amrex::Real <name> = rhs_ptr[<enum> * rn]; an rhs component written by an earlier function. Loaded before
    any store, so that an update of the same component in this function reads the earlier value."""
    name: str
    enum: str


@dataclass
class TileLoad(Stmt):
    """const amrex::Real <name> = tile_ptr[<index> * tn]; a tile temporary an earlier loop of this function stored."""
    name: str
    index: int


@dataclass
class TileStore(Stmt):
    """tile_ptr[<index> * tn] = <name>; a tile temporary this loop computed, for later loops of this function."""
    name: str
    index: int


@dataclass
class CoordDecl(Stmt):
    """const amrex::Real <name> = (<index> + 0.5) * DX; the cell-center coordinate."""
    name: str
    index: str


@dataclass
class TempDecl(Stmt):
    """A local temporary. `reassign`: it reuses (recycles) an earlier temporary's variable."""
    name: str
    expr: Expr
    reassign: bool
    mutable: bool
    comment: Optional[str] = None


@dataclass
class RhsStore(Stmt):
    """rhs_ptr[<enum> * rn] = <expr>;"""
    enum: str
    expr: Expr
    comment: Optional[str] = None


@dataclass
class KernelParam(CodeNode):
    name: str
    cpp_type: str
    init: str


@dataclass
class DeviceFunctionDecl(CodeNode):
    """An AMREX_GPU_DEVICE member function computing one cell of one loop. `uses_tile`: it takes the function's tile
    scratch array as a further argument."""
    name: str
    comment: str
    body: Sequence[Stmt]
    uses_tile: bool = False


@dataclass
class LaunchGroup(CodeNode):
    """The loops of one DSL function, launched in order, sharing a scratch MultiFab with one component per tile
    temporary (none if `tiles` is empty)."""
    name: str
    tiles: Sequence[str]
    functions: Sequence[DeviceFunctionDecl]


@dataclass
class KernelClassDecl(CodeNode):
    name: str
    namespace: str
    includes: Sequence[str]
    ctor_args: Sequence[tuple[str, str, Optional[str]]]  # (type, name, default)
    ctor_prelude: Sequence[str]
    dx: str
    ctor_checks: Sequence[tuple[str, str]]  # (failure condition, message)
    params: Sequence[KernelParam]
    launches: Sequence[LaunchGroup]  # in launch order
    banner: str
