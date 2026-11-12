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

"""
Where fields live and how a stencil read prints: array arguments (ArrayParam), each with its AccessScheme, and the
binding of each field component to an array and a component expression (FieldBinder, FieldBinding).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from EinsteinEngine.common.stencil_idx import StencilIdx


@dataclass(frozen=True)
class FieldBinding:
    """A field component's array (an ArrayParam name) and its component expression in that array (C++ text)."""
    array: str
    component: str


@dataclass(frozen=True)
class FieldBinder:
    """Binds every component of one declared field: `component(name)` gives the C++ component expression of the
    component named `name` (as the module's naming rule names it, canonical for symmetric pairs)."""
    array: str
    component: Callable[[str], str]

    def bind(self, component_name: str) -> FieldBinding:
        return FieldBinding(self.array, self.component(component_name))


class AccessScheme(ABC):
    """How an array's stencil reads print, and the per-function preamble they need."""

    @abstractmethod
    def preamble(self, array: str) -> list[str]:
        """Statements at the top of a function that reads the array (before any per-component declaration)."""

    @abstractmethod
    def unused_markers(self, array: str) -> list[str]:
        """Names declared by the preamble that may stay unused, to be voided."""

    @abstractmethod
    def component_decl(self, array: str, var: str, component: str) -> str | None:
        """A per-component declaration (e.g. a pointer), or None."""

    @abstractmethod
    def load(self, array: str, var: str, component: str, offset: StencilIdx) -> str:
        """The read of component `var` (component expression `component`) at `offset` from (i, j, k)."""


@dataclass(frozen=True)
class StridedPointer(AccessScheme):
    """
    GRTeclyn's access: one `const Real*` per component from the cell's pointer, and int strides, as GRTeclyn's
    derivative classes do: `in_X[a + b*sy + c*sz]`. Supports one strided array per function (its names are fixed).
    """
    ptr: str = 'state_ptr'
    sy: str = 'sy'
    sz: str = 'sz'
    sn: str = 'sn'
    var_prefix: str = 'in_'

    def preamble(self, array: str) -> list[str]:
        return [f'const amrex::Real *const {self.ptr} = {array}.ptr(i, j, k);',
                f'const int {self.sy} = static_cast<int>({array}.stride.a[0]);',
                f'const int {self.sz} = static_cast<int>({array}.stride.a[1]);',
                f'const std::ptrdiff_t {self.sn} = {array}.stride.a[2];']

    def unused_markers(self, array: str) -> list[str]:
        return [self.sy, self.sz, self.sn]

    def component_decl(self, array: str, var: str, component: str) -> str | None:
        return (f'const amrex::Real *const {self.var_prefix}{var} = {self.ptr} + '
                f'static_cast<std::ptrdiff_t>({component}) * {self.sn};')

    def load(self, array: str, var: str, component: str, offset: StencilIdx) -> str:
        text = ''
        for k, stride in zip(offset, (None, self.sy, self.sz)):
            if k == 0:
                continue
            magnitude = abs(k)
            term = str(magnitude) if stride is None else stride if magnitude == 1 else f'{magnitude} * {stride}'
            text += f' {"+" if k > 0 else "-"} {term}'
        return f'{self.var_prefix}{var}[{text.lstrip(" +") or "0"}]'


@dataclass(frozen=True)
class Array4Call(AccessScheme):
    """Array4's own indexing: `arr(i + a, j + b, k + c, comp)`."""

    def preamble(self, array: str) -> list[str]:
        return []

    def unused_markers(self, array: str) -> list[str]:
        return []

    def component_decl(self, array: str, var: str, component: str) -> str | None:
        return None

    def load(self, array: str, var: str, component: str, offset: StencilIdx) -> str:
        args = []
        for k, index in zip(offset, ('i', 'j', 'k')):
            args.append(index if k == 0 else f'{index} {"+" if k > 0 else "-"} {abs(k)}')
        return f'{array}({", ".join(args)}, {component})'


@dataclass(frozen=True)
class ArrayParam:
    """An array argument of the generated functions: its C++ name and type, and how it is read."""
    name: str
    cpp_type: str
    access: AccessScheme
