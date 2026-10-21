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

"""MHDuet's point functions on the shared AMReX backend (docs/AMREX-BACKEND-DESIGN.md, section 10, step 4)."""

from typing import Mapping, Sequence

from sympy import Idx, Indexed, Symbol

from EinsteinEngine.common.sympywrap import mk_symbol
from EinsteinEngine.frontend.dsl.amrex.amrex_frontend import AMReXModule
from EinsteinEngine.frontend.dsl.amrex.layout import Array4Call, ArrayParam, FieldBinder
from EinsteinEngine.frontend.dsl.amrex.schemes import NodeCentered, PointFunctionMode, StructParams
from EinsteinEngine.frontend.dsl.use_indices import idx_to_int
from EinsteinEngine.frontend.dsl.dsl_exception import DslException

MHDUET_ARRAY_TYPE = "amrex::Array4<amrex::Real const> const &"
MHDUET_ENUM_NAMESPACE = "simflowny_vars"


def mhduet_component_names(sym: Indexed, *_idxs: int) -> Symbol:
    """
    Simflowny's component names: the base name, an underscore, then one of x, y, z per index, without distinguishing
    up and down indices: gtd_xy, Betau_z, RHS_gtd_xx. Symmetric pairs get the canonical (lower index first) name.
    """
    name = str(sym.base) + "_"
    for ind in sym.args[1:]:
        assert isinstance(ind, Idx)
        name += "xyz"[idx_to_int(ind)]
    return mk_symbol(name)


class MHDuetPointFunction(AMReXModule):
    """
    A header of inline device functions `namespace::<function>(i, j, k, un, aux, prob_lo, dx, parameters, outputs...)`
    for MHDuet's Simflowny-generated fused kernels (`PointFunctionMode(outputs='refs')`). Fields are read with
    Array4's own indexing (`Array4Call`) from two arrays: `un`, the stage's evolved state (un, rk1, rk2 or rk3), and
    `aux`, MHDuet's unSyncAuxFieldGroup. A field's component `c` is `simflowny_vars::c` in its array. Coordinates are
    grid nodes, `prob_lo[a] + i * dx[a]` (MHDuet's xcoord). Parameters are members of `Simflowny_par const &
    parameters` under the recipe's names. Outputs are `amrex::Real &` out-parameters named as the host's locals.

    `fields` maps each declared field base name to its array ("un" or "aux"); the bases declared with
    `decl(..., rhs=R)` live in `un` and need no entry. Any other declaration without `rhs=` is a temporary.
    """

    def __init__(self, name: str, *, fields: Mapping[str, str], namespace: str, includes: Sequence[str] = (),
                 derivative_stencil_width: int = 5) -> None:
        arrays = [ArrayParam("un", MHDUET_ARRAY_TYPE, Array4Call()), ArrayParam("aux", MHDUET_ARRAY_TYPE, Array4Call())]
        names = {a.name for a in arrays}
        if bad := sorted(b for b, a in fields.items() if a not in names):
            raise DslException(f"Fields {bad} name an unknown array (arrays: {sorted(names)}).")
        binders = {a: FieldBinder(a, lambda c: f"{MHDUET_ENUM_NAMESPACE}::{c}") for a in names}
        table = dict(fields)
        super().__init__(name, namespace=namespace, arrays=arrays,
                         binder=lambda base: binders[table[base]] if base in table else None,
                         rhs_binder=binders["un"], coordinates=NodeCentered(), mode=PointFunctionMode(outputs='refs'),
                         params=StructParams(arg="parameters", cpp_type="Simflowny_par const &"),
                         naming=mhduet_component_names, includes=includes,
                         derivative_stencil_width=derivative_stencil_width, max_stencil_offset=3)
