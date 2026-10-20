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
How MHDuet 2.0's generated code names what the CCZ4 recipe (ccz4_mhduet.py) reads and writes: the declarations the
MHDuet subclass of the shared AMReX backend (docs/AMREX-BACKEND-DESIGN.md, sections 3-5 and 10; point-function mode)
installs. Pure data and small functions, independent of the backend classes.

In MHDuet's fused RHS kernel (AdvanceLevel.cpp), at a point (i, j, k):
- the evolved fields are `un(i, j, k, simflowny_vars::<name>)`, where `un` is the stage's input Array4 (un, rk1,
  rk2 or rk3), so it is a function argument;
- the primitives are `unSyncAuxFieldGroup(i, j, k, simflowny_vars::<name>)` (a second enum in the same namespace);
- the parameters are members of the captured `Simflowny_par parameters`, under the recipe's names;
- the coordinates are grid nodes, `prob_lo[d] + i * dx[d]` (`xcoord(i)` in Commons.h), with `prob_lo` and `dx` the
  captured `GpuArray<Real, 3>`s;
- the RHS goes to the lambda's locals `RHS_<component>`, then through MHDuet's boundary override, KO and RK update.
"""

from typing import Optional

NAMESPACE = "ee_mhduet"
HEADER = "CCZ4_EE.H"
FUNCTIONS = {"rhs": "ccz4_rhs", "ko": "ccz4_ko"}  # DSL function name -> C++ function name

# Point-function arguments, in signature order (design 5.4): arrays, coordinates and spacing, parameter struct.
ARRAY_TYPE = "amrex::Array4<amrex::Real const> const&"
ARRAYS = [("un", ARRAY_TYPE), ("aux", ARRAY_TYPE)]  # host passes the stage array and unSyncAuxFieldGroup
COORD_ARGS = [("prob_lo", "amrex::GpuArray<amrex::Real, 3> const&"), ("dx", "amrex::GpuArray<amrex::Real, 3> const&")]
PARAM_STRUCT = ("parameters", "Simflowny_par const&")  # StructParams: member name = recipe parameter name

# Node-centered coordinates (CoordinateScheme NodeCentered).
COORDINATES = ("prob_lo[0] + i * dx[0]", "prob_lo[1] + j * dx[1]", "prob_lo[2] + k * dx[2]")

ENUM_NAMESPACE = "simflowny_vars"
# Declared base name -> (array, kind). kind: scalar, vector (x, y, z), symmetric (xx, xy, xz, yy, yz, zz).
FIELDS = {
    # evolved spacetime (state) fields
    "chi": ("un", "scalar"), "gtd": ("un", "symmetric"), "Atd": ("un", "symmetric"), "trK": ("un", "scalar"),
    "theta": ("un", "scalar"), "Gamh": ("un", "vector"), "Alpha": ("un", "scalar"), "Betau": ("un", "vector"),
    # evolved matter fields the CCZ4 RHS reads
    "Df": ("un", "scalar"), "tauf": ("un", "scalar"), "Sfd": ("un", "vector"), "Bfu": ("un", "vector"),
    # primitives
    "vfd": ("aux", "vector"), "pf": ("aux", "scalar"),
}
SUFFIXES = {"scalar": [""], "vector": ["_x", "_y", "_z"], "symmetric": ["_xx", "_xy", "_xz", "_yy", "_yz", "_zz"]}


def component_names(base: str) -> list[str]:
    """MHDuet's component names of a declared field, in the backend's canonical order (one name per symmetric pair,
    lower index first): gtd -> gtd_xx, gtd_xy, ..., gtd_zz."""
    return [base + s for s in SUFFIXES[FIELDS[base][1]]]


def binding(base: str) -> Optional[tuple[str, dict[str, str]]]:
    """The field binder's data for a declared base name: (array, {component name: component expression}), or None
    for anything that is not a bound field (temporaries, outputs)."""
    if base not in FIELDS:
        return None
    return FIELDS[base][0], {c: f"{ENUM_NAMESPACE}::{c}" for c in component_names(base)}


STATE = ["chi", "gtd", "Atd", "trK", "theta", "Gamh", "Alpha", "Betau"]  # recipe declaration order


def outputs(function: str) -> list[str]:
    """The register outputs of a function, in the backend's output-declaration order (the recipe declares the state,
    and so the RHS, in STATE order): RHS_<component> for rhs, ko_<component> for ko."""
    prefix = {"rhs": "RHS_", "ko": "ko_"}[function]
    return [prefix + c for base in STATE for c in component_names(base)]
