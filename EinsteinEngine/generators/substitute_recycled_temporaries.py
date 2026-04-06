#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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
from typing import Optional, cast

import sympy as sy

from EinsteinEngine.dsl.eqnlist import TemporaryReplacement, EqnList


@dataclass(frozen=True)
class SubstituteRecycledTemporariesResult:
    eqns: list[tuple[sy.Symbol, sy.Expr]]
    substituted_lhs_idxes: set[int]


def substitute_recycled_temporaries(eqn_list: EqnList) -> SubstituteRecycledTemporariesResult:
    eqns: list[tuple[sy.Symbol, sy.Expr]] = list()
    substituted_lhs_idxes: set[int] = set()

    for eqn_idx, (lhs, rhs) in enumerate(eqn_list.sorted_eqns):
        active_replacements = list(filter(
            lambda r: r.begin_eqn <= eqn_idx <= r.end_eqn,
            eqn_list.temporary_replacements
        ))

        current_line_replacement = cast(Optional[TemporaryReplacement],
                                        next(filter(lambda r: r.begin_eqn == eqn_idx, active_replacements), None))

        for replacement in active_replacements:
            rhs = rhs.replace(replacement.old, replacement.new)  # type: ignore[no-untyped-call]

        if current_line_replacement:
            assert lhs == current_line_replacement.old, "Current line replacement target doesn't match LHS"
            lhs = current_line_replacement.new
            substituted_lhs_idxes.add(eqn_idx)

        eqns.append((lhs, rhs))

    return SubstituteRecycledTemporariesResult(eqns, substituted_lhs_idxes)
