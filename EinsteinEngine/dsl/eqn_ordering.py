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

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Callable, TYPE_CHECKING

from sympy import Symbol, Expr

from EinsteinEngine.dsl.sympywrap import free_symbols

if TYPE_CHECKING:
    from EinsteinEngine.dsl.eqnlist import EqnList

EqnOrderingFn = Callable[[dict[Symbol, Expr], 'EqnList'], Iterator[Symbol]]

def maximize_symbol_reuse(eqns: dict[Symbol, Expr], eqn_list: EqnList) -> Iterator[Symbol]:
    """
    Orders equations based on symbol reuse, prioritizing equations that use symbols already present in previous equations.
    Equations with higher complexity are given higher priority. The first equation is always the most complex.
    """

    if len(eqns) == 0:
        return

    eqns_remaining = eqns.copy()
    in_memory: set[Symbol] = set()

    disambiguation = sorted(eqns_remaining.keys(), key=str, reverse=True)

    lhs, rhs = max(eqns_remaining.items(), key=lambda kv: (eqn_list.complexity[kv[0]], disambiguation.index(kv[0])))
    del eqns_remaining[lhs]
    in_memory.update(free_symbols(rhs))
    yield lhs

    while len(eqns_remaining) > 0:
        lhs, rhs = max(eqns_remaining.items(),
                       key=lambda kv: (len(free_symbols(kv[1]).intersection(in_memory)), eqn_list.complexity[kv[0]],
                                       disambiguation.index(kv[0])))
        del eqns_remaining[lhs]
        in_memory.update(free_symbols(rhs))
        yield lhs

def prioritize_rare_symbols(eqns: dict[Symbol, Expr],
                            eqn_list: EqnList,
                            consider_frequency: bool = True,
                            complexity_factor: float = 0.0) -> Iterator[Symbol]:
    """
    Orders equations based on symbol rarity.
    Equations which use symbols that are less common in other equations are given higher priority.

    To determine the rarity of a symbol, multiple occurrences of the same symbol in an equation are treated as one.
    If `consider_frequency` is true, when evaluating the overall priority of an equation, the rarity of each symbol is weighted positively by the frequency of that symbol in the equation.

    The complexity score of an equation, scaled by `complexity_factor`, is added to the priority.
    """

    if len(eqns) == 0:
        return

    reciprocal_rarity: dict[Symbol, float] = defaultdict(int)
    frequency_by_eqn: dict[Symbol, dict[Symbol, float]] = defaultdict(dict)  # {lhs: {sym: freq}}
    for lhs, rhs in eqns.items():
        for sym in free_symbols(rhs):
            reciprocal_rarity[sym] += 1
            frequency_by_eqn[lhs][sym] = rhs.count(sym)  # type: ignore[no-untyped-call]

    def symbol_rarity(sym: Symbol) -> float:
        return 1 / reciprocal_rarity[sym]

    def symbol_score(sym: Symbol, lhs: Symbol) -> float:
        return frequency_by_eqn[lhs][sym] * symbol_rarity(sym) if consider_frequency else symbol_rarity(sym)

    def eqn_score(lhs: Symbol) -> float:
        return (complexity_factor * eqn_list.complexity[lhs]) + sum(symbol_score(sym, lhs) for sym in free_symbols(eqns[lhs]))

    disambiguation = sorted(eqns.keys(), key=str, reverse=True)
    ordered = sorted(eqns.keys(), key=lambda lhs: (eqn_score(lhs), eqn_list.complexity[lhs], disambiguation.index(lhs)), reverse=True)

    yield from ordered.__iter__()
