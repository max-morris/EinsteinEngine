#  Copyright (C) 2026 Max Morris, Steven R. Brandt and other Einstein Engine contributors.
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

from EinsteinEngine.common.sympywrap import mk_idx
from sympy import Idx

__all__ = [
    "ui", "li", "uj", "lj", "uk", "lk",
    "ua", "la", "ub", "lb", "uc", "lc", "ud", "ld",
    "u0", "l0", "u1", "l1", "u2", "l2", "u3", "l3", "u4", "l4", "u5", "l5",
]


def _mk_index_pair(stem: str) -> tuple[Idx, Idx]:
    return mk_idx(f"u{stem}"), mk_idx(f"l{stem}")


ui, li = _mk_index_pair("i")
uj, lj = _mk_index_pair("j")
uk, lk = _mk_index_pair("k")
ua, la = _mk_index_pair("a")
ub, lb = _mk_index_pair("b")
uc, lc = _mk_index_pair("c")
ud, ld = _mk_index_pair("d")
u0, l0 = _mk_index_pair("0")
u1, l1 = _mk_index_pair("1")
u2, l2 = _mk_index_pair("2")
u3, l3 = _mk_index_pair("3")
u4, l4 = _mk_index_pair("4")
u5, l5 = _mk_index_pair("5")
