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

"""Coordinates and parameters: the CoordinateScheme and ParamScheme strategies of an AMReX module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence


class CoordinateScheme(ABC):
    """The C++ expression of the coordinates x, y, z at cell (i, j, k)."""

    @abstractmethod
    def expr(self, axis: int, index: str) -> str:
        ...


@dataclass(frozen=True)
class CellCentered(CoordinateScheme):
    """Cell centers from the index origin, `(i + 0.5) * DX` (GRTeclyn's Coordinates before its center); DX is the
    ParamScheme's spacing."""
    spacing: str = 'DX'

    def expr(self, axis: int, index: str) -> str:
        return f'({index} + 0.5) * {self.spacing}'


@dataclass(frozen=True)
class NodeCentered(CoordinateScheme):
    """Grid nodes, `prob_lo[a] + i * dx[a]` (MHDuet's xcoord(i)); prob_lo and dx are host GpuArray<Real, 3>
    arguments."""
    prob_lo: str = 'prob_lo'
    dx: str = 'dx'

    def expr(self, axis: int, index: str) -> str:
        return f'{self.prob_lo}[{axis}] + {index} * {self.dx}[{axis}]'


@dataclass(frozen=True)
class CtorArg:
    cpp_type: str
    name: str
    default: str | None = None


@dataclass(frozen=True)
class CtorCheck:
    failure_condition: str  # C++ expression, valid after the constructor prelude
    message: str


class ParamScheme(ABC):
    """How parameters and the grid spacing are spelled in the generated code."""


@dataclass(frozen=True)
class MemberParams(ParamScheme):
    """
    Kernel mode (GRTeclyn): parameters are class members, assigned in the constructor body after a C++ prelude; the
    spacing is the members DX, DXI, DYI, DZI, from the expression `dx` (in the constructor arguments).
    """
    ctor_args: Sequence[CtorArg]
    dx: str
    ctor_prelude: Sequence[str] = ()
    ctor_checks: Sequence[CtorCheck] = ()


@dataclass(frozen=True)
class StructParams(ParamScheme):
    """
    Point mode (MHDuet): parameters are members of one struct argument, `<arg>.<member(name)>` (the name unchanged
    by default); the spacing comes from a host GpuArray<Real, 3> argument `dx`, inverted once into locals.
    """
    arg: str = 'parameters'
    cpp_type: str = 'const Simflowny_par &'
    member: Callable[[str], str] = field(default=lambda p: p)
    dx: str = 'dx'


class EmissionMode(ABC):
    """The overall shape of the generated code; the mode also fixes the output style."""


@dataclass(frozen=True)
class KernelLaunchMode(EmissionMode):
    """
    A header-only class: one AMREX_GPU_DEVICE member function per loop and a host `compute(rhs, state)` that launches
    one ParallelFor per loop over the valid cells, in creation order; outputs are stores to the rhs array (overwrites
    reload their earlier value before any store); hard splits keep their tile temporaries in a scratch MultiFab.
    `rhs_array` and `state_array` name the compute() arguments' arrays.
    """
    rhs_array: str = 'rhs'


@dataclass(frozen=True)
class PointFunctionMode(EmissionMode):
    """
    Header-only AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE free functions, one per DSL function, in a namespace, to be
    called inside a host-written kernel. Outputs stay in registers: `Real&` out-parameters (outputs='refs') or a
    returned struct (outputs='struct'), each assigned exactly once. Locals and temporaries get `prefix`.
    """
    outputs: str = 'refs'
    prefix: str = 'ee_'

    def __post_init__(self) -> None:
        if self.outputs not in ('refs', 'struct'):
            raise ValueError(f"PointFunctionMode outputs must be 'refs' or 'struct', not {self.outputs!r}")
