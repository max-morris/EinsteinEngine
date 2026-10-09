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

import sympy as sy

from EinsteinEngine.emit.code.cpp_carpetx.cpp_carpetx_named_roots import named_root
from EinsteinEngine.generators.sympy_complexity import SympyComplexityVisitor, TRANSCENDENTAL_COST


class CppCarpetXComplexityVisitor(SympyComplexityVisitor):
    """
    The cost model for code emitted by the CarpetX backend.

    SymPy builds sqrt(x) and cbrt(x) as x**(1/2) and x**(1/3), and this backend emits
    those powers, and any other in `named_root`, as a named call rather than pow().
    Charge them like the transcendental functions: the surcharge plus the base, not the
    exponent.
    """

    def _complexity_pow(self, n: sy.Pow) -> int:
        base, power = n.args
        if isinstance(power, sy.Rational) and named_root(power.p, power.q) is not None:
            base_complexity: int = self.complexity(base)
            return TRANSCENDENTAL_COST + base_complexity
        return super()._complexity_pow(n)
