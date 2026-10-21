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

"""Component naming rules for AMReX modules (a declaration's `substitution_rule`)."""

from sympy import Idx, Indexed, Symbol

from EinsteinEngine.common.sympywrap import mk_symbol
from EinsteinEngine.frontend.dsl.use_indices import idx_to_int


def digit_component_names(sym: Indexed, *_idxs: int) -> Symbol:
    """
    The base name followed by the 1-based component indices, without distinguishing up and down indices: h11, h12,
    Gamma1, shift3. GRChombo's and GRTeclyn's naming (the same rule as the GRChombo backend's subst_tensor_grchombo).
    """
    name = str(sym.base)
    for ind in sym.args[1:]:
        assert isinstance(ind, Idx)
        name += str(idx_to_int(ind) + 1)
    return mk_symbol(name)
