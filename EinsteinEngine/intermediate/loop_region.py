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
The region a loop runs over (Interior, Boundary, or Everywhere), inferred from the grid variables it writes or reads.

The generators use this to pick the loop region of each EqnList, and `EqnComplex` uses it to validate refinements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from sympy import Symbol

from EinsteinEngine.emit.ccl.schedule.schedule_tree import IntentRegion
from EinsteinEngine.frontend.definitions import stencil

if TYPE_CHECKING:
    from EinsteinEngine.intermediate.eqnlist import EqnList

# Names that are never grid variables: coordinates, time, and the grid spacings (and their inverses).
NON_GRID_VARIABLE_NAMES: frozenset[str] = frozenset({'t', 'x', 'y', 'z', 'DXI', 'DYI', 'DZI', 'DX', 'DY', 'DZ', 'DT'})


def may_be_grid_variable(sym: Symbol) -> bool:
    """False for symbols that are never grid variables by name: coordinates, time, and grid spacings."""
    return str(sym).replace("'", "") not in NON_GRID_VARIABLE_NAMES


@dataclass(frozen=True)
class LoopRegion:
    """The result of `infer_loop_region`."""
    region: Optional[IntentRegion]  # None if it cannot be inferred because the candidate regions conflict
    writes_grid_vars: bool  # True iff the region was inferred from written grid variables
    from_mixed_reads: bool = False  # True iff Interior was inferred from reads that mix Everywhere and Interior


def infer_loop_region(eqn_list: EqnList, is_grid_var: Callable[[Symbol], bool]) -> LoopRegion:
    """
    The region the loop for `eqn_list` runs over. All grid variables it writes must share one write region, which is
    the loop region. A loop that writes no grid variables (e.g., one that only computes tile temps) runs over the
    Interior if it applies a stencil with a nonzero offset, and otherwise over the read region of the grid variables it
    reads; a mix of Everywhere and Interior reads gives Interior. A loop that neither writes nor reads grid variables
    is analytic and runs Everywhere.
    """
    writes = frozenset(region for sym, region in eqn_list.write_decls.items() if is_grid_var(sym))
    reads = frozenset(region for sym, region in eqn_list.read_decls.items() if is_grid_var(sym))

    if len(writes) > 0:
        return LoopRegion(next(iter(writes)) if len(writes) == 1 else None, True)

    if len(reads) == 0:
        return LoopRegion(IntentRegion.Everywhere, False)

    for rhs in eqn_list.eqns.values():
        for sten in rhs.find(stencil):  # type: ignore[no-untyped-call]
            if sten.args[1] != 0 or sten.args[2] != 0 or sten.args[3] != 0:
                return LoopRegion(IntentRegion.Interior, False)

    if len(reads) == 1:
        return LoopRegion(next(iter(reads)), False)
    if reads == {IntentRegion.Everywhere, IntentRegion.Interior}:
        return LoopRegion(IntentRegion.Interior, False, from_mixed_reads=True)
    return LoopRegion(None, False)
