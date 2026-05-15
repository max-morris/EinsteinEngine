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

from typing import Optional, Set

from sympy import Indexed, IndexedBase, Expr, Symbol

from EinsteinEngine import mk_symbol
from EinsteinEngine.frontend.frontend import Frontend


class DslFrontend[ParamData](Frontend):
    params: dict[str, ParamData]

    def __init__(self, dimensionality: int = 3):
        super().__init__(dimensionality=dimensionality)
        self.params = dict()

    def _mk_param_set(self) -> Set[Symbol]:
        ret: Set[Symbol] = set()
        for k in self.params:
            ret.add(mk_symbol(k))
        return ret
