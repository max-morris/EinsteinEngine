#!/usr/bin/env python3

#  Copyright (C) 2026 Steven R. Brandt and other Einstein Engine contributors.
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

"""
Tests the upwind/downwind custom stencil ("osdiv") from unit_tests/test_use_indices.py
through an actual CarpetX run.

o0 is built out of h_step() so that, in each of x, y, and z, it is a piecewise-linear
function of slope 1 with a jump discontinuity of size 2 at coordinate == 1:

    coord < 1:  contributes (coord + 1)
    coord > 1:  contributes (coord - 1)

osdiv(o0, la) is a custom finite-difference stencil that picks forward- or
backward-differencing based on the sign of a "direction" field o1[la]:

    o1[la] > 0  ->  forward difference:  (f(x+dx) - f(x)) / dx
    o1[la] < 0  ->  backward difference: (f(x) - f(x-dx)) / dx

o1[la] is set to (coordinate_la - 1), i.e. it is negative on the side of the jump
where o0's slope is (coord + 1) and positive on the side where it's (coord - 1).
That guarantees the one-sided stencil always reads two points from the same smooth
linear piece and never straddles the jump at coordinate == 1 -- so, as long as no
grid point lands exactly on x == 1, y == 1, or z == 1 (see osdiv.par), osdiv(o0, la)
should come out to exactly 1 everywhere in the interior, despite o0's discontinuities.
"""

# Create a set of grid functions
gf = ThornDef("TestEinsteinEngine", "OsDivTest", derivative_stencil_width=5)

# Declare gfs
o0 = gf.decl("o0", [])
o1 = gf.decl("o1", [ua])
o2 = gf.decl("o2", [la])
o3 = gf.decl("o3", [la])
ZeroVal = gf.decl("ZeroVal", [], from_thorn="ZeroTest")

x, y, z = gf.mk_coords()

kdelta = gf.mk_kdelta()

# Custom upwind/downwind derivative operator (same recipe as in test_use_indices.py).
osdiv = gf.mk_stencil("osdiv", la, h_step( o1[ub]*kdelta[la,lb]) * finite_difference_stencil(2,1, 1,la)+
                                   h_step(-o1[ub]*kdelta[la,lb]) * finite_difference_stencil(2,1,-1,la))

set_o1 = gf.create_function("set_o1", ScheduleBin.Analysis)
set_o1.add_eqn(o1[u0], x - 1)
set_o1.add_eqn(o1[u1], y - 1)
set_o1.add_eqn(o1[u2], z - 1)

set_o0 = gf.create_function("set_o0", ScheduleBin.Analysis)
set_o0.add_eqn(o0,
               h_step(x - 1) * (x - 1) + h_step(1 - x) * (x + 1) +
               h_step(y - 1) * (y - 1) + h_step(1 - y) * (y + 1) +
               h_step(z - 1) * (z - 1) + h_step(1 - z) * (z + 1))

apply_osdiv = gf.create_function(
    "apply_osdiv", ScheduleBin.Analysis, schedule_after=["set_o1", "set_o0"])
apply_osdiv.add_eqn(o2[la], osdiv(o0, la))

osdiv_zero = gf.create_function(
    "OsdivZero", ScheduleBin.Analysis, schedule_after=["apply_osdiv"])
osdiv_zero.add_eqn(ZeroVal, (o2[l0] - 1) ** 2 + (o2[l1] - 1) ** 2 + (o2[l2] - 1) ** 2)

gf.bake(
    do_cse=True,
    do_madd=False,
    do_recycle_temporaries=True
)

check_zero = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier('CheckZeroGroup'),
    at_or_in=AtOrIn.At,
    schedule_bin=Identifier('analysis'),
    description=String('Check that osdiv(o0, la) is 1 everywhere'),
    after=[Identifier('OsdivZero')]
)

CppCarpetXWizard(
    gf,
    CppCarpetXGenerator(
        gf,
        sync_mode=SyncMode.EmulatePresync,
        extra_schedule_blocks=[check_zero]
    )
).generate_thorn()
