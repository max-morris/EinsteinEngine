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

"""Code tree of the shared AMReX backend (kernel-launch and point-function modes)."""

from dataclasses import dataclass, field
from typing import Optional, Sequence

from EinsteinEngine.common.stencil_idx import StencilIdx
from EinsteinEngine.emit.code.common.code_tree import Expr, Stmt, CodeNode


@dataclass
class FieldLoad(Expr):
    """A read of a field component at a stencil offset from (i, j, k); printed by its array's AccessScheme."""
    var: str
    offset: StencilIdx


@dataclass
class Line(Stmt):
    """A verbatim statement (preamble lines, per-component declarations)."""
    text: str


@dataclass
class RhsLoad(Stmt):
    """Kernel mode: `const amrex::Real <name> = rhs_ptr[<enum> * rn];`, an rhs component an earlier loop wrote, loaded
    before any store, so that an update of the same component in this loop reads the earlier value."""
    name: str
    enum: str


@dataclass
class TileLoad(Stmt):
    """Kernel mode: `const amrex::Real <name> = tile_ptr[<index> * tn];`."""
    name: str
    index: int


@dataclass
class TileStore(Stmt):
    """Kernel mode: `tile_ptr[<index> * tn] = <name>;`."""
    name: str
    index: int


@dataclass
class CoordDecl(Stmt):
    """`const amrex::Real <name> = <expr>;`, a cell coordinate."""
    name: str
    expr: str


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
    """Kernel mode: `rhs_ptr[<enum> * rn] = <expr>;`."""
    enum: str
    expr: Expr
    comment: Optional[str] = None


@dataclass
class OutputAssign(Stmt):
    """Point mode: `<target> = <expr>;`, an output out-parameter or struct member."""
    target: str
    expr: Expr
    comment: Optional[str] = None


@dataclass
class KernelParam(CodeNode):
    name: str
    cpp_type: str
    init: str


@dataclass
class DeviceFunctionDecl(CodeNode):
    """Kernel mode: an AMREX_GPU_DEVICE member function computing one cell of one loop. `preamble`: the state access's
    preamble; `unused`: names to void; `uses_tile`: it takes the function's tile scratch array."""
    name: str
    comment: str
    body: Sequence[Stmt]
    state_array: str
    rhs_array: str
    preamble: Sequence[str]
    unused: Sequence[str]
    uses_tile: bool = False


@dataclass
class LaunchGroup(CodeNode):
    """Kernel mode: the loops of one DSL function, launched in order, sharing a scratch MultiFab with one component per
    tile temporary (none if `tiles` is empty)."""
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


@dataclass
class PointArg(CodeNode):
    cpp_type: str
    name: str
    used: bool


@dataclass
class PointFunctionDecl(CodeNode):
    """Point mode: an inline AMREX_GPU_HOST_DEVICE function. `outputs`: (name, member) in order; with a struct, the
    function returns `struct_name`."""
    name: str
    comment: str
    args: Sequence[PointArg]
    body: Sequence[Stmt]
    outputs: Sequence[str]
    struct_name: Optional[str] = None


@dataclass
class PointHeaderDecl(CodeNode):
    namespace: str
    includes: Sequence[str]
    functions: Sequence[PointFunctionDecl]
    banner: str
    guard: str
