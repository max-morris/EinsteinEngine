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

from EinsteinEngine.common.sympywrap import mk_function, sympify, mk_idx, mk_symbol
from sympy import Idx, Expr, Rational  # type: ignore[import-untyped]
from math import comb

one = sympify(1)
zero = sympify(0)

x = mk_symbol("x")
y = mk_symbol("y")
z = mk_symbol("z")
t = mk_symbol("t")
no_idx = mk_idx("no_idx")
dummy = mk_symbol("_dummy_")

stencil = mk_function("stencil")
DD = mk_function("DD")
DDI = mk_function("DDI")
noop = mk_function("noop")
div = mk_function("div")
D = mk_function("D")
muladd = mk_function("muladd")
pull_out = mk_function("pull_out")

# These symbols represent the inverse of the spatial discretization.
DXI = mk_symbol("DXI")
DYI = mk_symbol("DYI")
DZI = mk_symbol("DZI")
DX = mk_symbol("DX")
DY = mk_symbol("DY")
DZ = mk_symbol("DZ")


for func in [stencil, DD, DDI, noop, div, D, muladd]:
    if func.__module__ is None:
        func.__module__ = "functions"

def kreiss_oliger_stencil(stencil_width: int, la: Idx) -> Expr:
    n = (stencil_width + 1) // 2
    sign = (-1) ** (n + 1)
    k = n
    coef = comb(2 * n, n)
    formula = (-1) ** k * coef * stencil(0)
    for k in range(n):
        k1 = k
        k2 = 2 * n - k
        offset1 = n - k1
        offset2 = n - k2
        coef = comb(2 * n, k1)
        assert coef == comb(2 * n, k2)
        s1 = (-1) ** k1
        s2 = (-1) ** k2
        if s1 > 0:
            formula += coef * noop(stencil(la * offset1) + s2 * stencil(la * offset2))
        else:
            formula += s1 * coef * noop(stencil(la * offset1) + s1 * s2 * stencil(la * offset2))
    prefactor = sign * 2 ** (2 * n)
    result: Expr = formula * sign * Rational(1, prefactor) * DDI(la)
    return result
