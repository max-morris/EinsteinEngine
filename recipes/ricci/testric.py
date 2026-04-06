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

from EinsteinEngine import *

# Create a set of grid functions
gf = ThornDef("TestEinsteinEngine", "Ricci")
gf.set_derivative_stencil(5)

a = gf.add_param("a", default=10.0, desc="Just a constant")
b = gf.add_param("b", default=0.2, desc="Just a constant")
c = gf.add_param("c", default=0.1, desc="Just a constant")

# Declare gfs
g = gf.decl("g", [li, lj], symmetries=[(li, lj)], from_thorn="ADMBaseX")
x, y, z = gf.mk_coords()

Ric = gf.decl("Ric", [la, lb], symmetries=[(la, lb)])
ZeroVal = gf.decl("ZeroVal", [], from_thorn="ZeroTest")
G = gf.decl("Affine", [ua, lb, lc], symmetries=[
            (lb, lc)], substitution_rule=None)

gmat = gf.get_matrix(g[la, lb])
print(gmat)
imat = simplify(inv(gmat)*det(gmat))
gf.add_substitution_rule(g[ua, ub], imat)

# Metric
grr = sqrt(1+c**2)*(a+b*x**2)
gqq = sqrt(1+c**2)/(a+b*x**2)
gpp = sympify(1)
Z = sympify(0)
gmat = mkMatrix([
    [grr,   c,   Z],
    [c, gqq,   Z],
    [Z,   Z, gpp]])
assert det(gmat) == 1

# Define the affine connections
gf.add_substitution_rule(
    G[la, lb, lc], (D(g[la, lb], lc) + D(g[la, lc], lb) - D(g[lb, lc], la)) / 2)
gf.add_substitution_rule(G[ud, lb, lc], g[ud, ua] * G[la, lb, lc])

fun = gf.create_function("setGL", ScheduleBin.Analysis)

fun.add_eqn(Ric[li, lj],
            D(G[ua, li, lj], la) - D(G[ua, la, li], lj) +
            G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])

fun = gf.create_function(
    "MetricSet", ScheduleBin.Analysis, schedule_before=["setGL"])
fun.add_eqn(g[li, lj], gmat)

fun = gf.create_function(
    "RicZero", ScheduleBin.Analysis, schedule_after=["setGL"])
fun.add_eqn(ZeroVal, Ric[l0, l0]-b*(a*c**2 + a - 3*b*c **
            2*x**2 - 3*b*x**2)/(a**2 + 2*a*b*x**2 + b**2*x**4))

gf.bake(
    do_cse=True,
    do_madd=False,
    do_recycle_temporaries=True,
    do_split_output_eqns=False
)

check_zero = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier('CheckZeroGroup'),
    at_or_in=AtOrIn.At,
    schedule_bin=Identifier('analysis'),
    description=String('Do the check'),
    after=[Identifier('RicZero')]
)

CppCarpetXWizard(
    gf,
    CppCarpetXGenerator(
        gf,
        sync_mode=SyncMode.EmulatePresync,
        extra_schedule_blocks=[check_zero]
    )
).generate_thorn()
