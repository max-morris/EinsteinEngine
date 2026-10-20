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

"""GRTeclyn's kernel class on the shared AMReX backend (docs/AMREX-BACKEND-DESIGN.md, section 10, step 3)."""

from typing import Sequence

from EinsteinEngine.frontend.dsl.amrex.amrex_frontend import AMReXModule
from EinsteinEngine.frontend.dsl.amrex.layout import ArrayParam, FieldBinder, StridedPointer
from EinsteinEngine.frontend.dsl.amrex.schemes import CellCentered, CtorArg, KernelLaunchMode, MemberParams


class GRTeclynKernelClass(AMReXModule):
    """
    A class `namespace::name` for GRTeclyn's BinaryBH: a constructor from `ctor_args` (with `ctor_prelude`, and the
    grid spacing `dx`), one device function per DSL function reading the CCZ4 state through `StridedPointer` (int
    strides, as GRTeclyn's derivative classes), cell-centered coordinates, and a host `compute(rhs, state)` that
    launches one `ParallelFor` per loop (`KernelLaunchMode`). State fields bind by declaration; the rhs components are
    `c_<name>` of GRTeclyn's StateVariables.hpp, which every header includes first.
    """

    def __init__(self, name: str, *, ctor_args: Sequence[CtorArg], dx: str, ctor_prelude: Sequence[str] = (),
                 includes: Sequence[str] = (), namespace: str = "EE", derivative_stencil_width: int = 5) -> None:
        super().__init__(name, namespace=namespace,
                         arrays=[ArrayParam("state", "amrex::Array4<amrex::Real const> const &", StridedPointer())],
                         binder=lambda base: None, rhs_binder=FieldBinder("state", lambda c: f"c_{c}"),
                         coordinates=CellCentered(), mode=KernelLaunchMode(),
                         params=MemberParams(ctor_args=list(ctor_args), dx=dx, ctor_prelude=list(ctor_prelude)),
                         includes=["StateVariables.hpp", *includes], derivative_stencil_width=derivative_stencil_width)
