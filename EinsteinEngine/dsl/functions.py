#  Copyright (C) 2024-2026 Max Morris and other Einstein Engine contributors.
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
