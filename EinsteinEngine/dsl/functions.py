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

from EinsteinEngine.dsl.dimension import get_dimension
from EinsteinEngine.dsl.sympywrap import mkFunction
from sympy import Idx, Expr, Rational
from math import comb

stencil = mkFunction("stencil")
DD = mkFunction("DD")
DDI = mkFunction("DDI")
noop = mkFunction("noop")
div = mkFunction("div")
D = mkFunction("D")
muladd = mkFunction("muladd")

# First derivatives
for i in range(get_dimension()):
    div_nm = "div" + "xyz"[i]
    func = mkFunction(div_nm)
    func.__module__ = "functions"
    globals()[div_nm] = func

# Second derivatives
for i in range(get_dimension()):
    for j in range(i, get_dimension()):
        div_nm = "div" + "xyz"[i] + "xyz"[j]
        func = mkFunction(div_nm)
        func.__module__ = "functions"
        globals()[div_nm] = func

for func in [stencil, DD, DDI, noop, div, D, muladd]:
    if func.__module__ is None:
        func.__module__ = "functions"

def kreiss_oliger_stencil(M:int, la:Idx)->Expr:
    assert M % 2 == 1
    N = (M+1)//2
    sign = (-1) ** (N + 1)
    formula = [0] * (N + 1)
    for k in range(2 * N + 1):
        offset = N - k
        coeff = sign * (-1) ** k * comb(2 * N, k)
        formula[abs(offset)] += coeff*stencil(offset*la)
    prefactor = 2 ** (2 * N)
    result : Expr = sum([noop(f) for f in formula]) * Rational(1, prefactor) * DDI(la)
    return result
