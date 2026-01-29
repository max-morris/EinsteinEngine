#!/usr/bin/env python3
from EmitCactus import *

"""
The waveequation! It can't be solved too many times.
"""

import nrpy.helpers.conditional_file_updater as cfu
from math import pi


# If we change our configuration, this will show us diffs of the
# new output and the old.
cfu.verbose = True


flat_metric = mkMatrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

# Create a set of grid functions
gf = ThornDef("TestEmitCactus", "WaveEqn")

# Use a NRPy calculated stencil instead
gf.set_derivative_stencil(5)

# Declare gfs
v_t = gf.decl("v_t", [], centering=Centering.VVV)
v = gf.decl("v", [], centering=Centering.VVV, rhs=v_t)
u_t = gf.decl("u_t", [], centering=Centering.VVV)
u = gf.decl("u", [], centering=Centering.VVV, rhs=u_t)
ZeroVal = gf.decl("ZeroVal", [], from_thorn="ZeroTest")

# Declare the metric
g = gf.decl("g", [li, lj], symmetries=[(li, lj)], substitution_rule=flat_metric)

# Declare params
spd = gf.add_param("spd", default=1.0, desc="The wave speed")
kx = gf.add_param("kx", default=pi / 20, desc="The wave number in the x-direction")
ky = gf.add_param("ky", default=pi / 20, desc="The wave number in the y-direction")
amp = gf.add_param("amp", default=10, desc="The amplitude")

# c = w/k
w = spd*sqrt(kx**2 + ky**2)

# Fill in values
gf.add_substitution_rule(g[ui, uj], flat_metric)

t, x, y, z = gf.mk_coords(with_time=True)

# Add the equations we want to evolve.
evo = gf.create_function("newwave_evo", ScheduleBin.Evolve)
evo.add_eqn(v_t, u)
evo.add_eqn(u_t, spd ** 2 * g[ui, uj] * D(v, li, lj))  # ==> u_t = spd^2 * (d^2_x + d^2_y) v
print('*** ThornFunction wave_evo:')

# Dump
#evo.dump()

# Show tensortypes
#evo.show_tensortypes()

# Again for wave_init
# du/dt = spd**2 * ((d/dx)**2 u + (d/dy)**2 u)
# dv/dt = u
vfun = amp*sin(kx * x) * sin(ky * y) * sin(w * t)
ufun = vfun.diff(t)

init = gf.create_function("newwave_init", ScheduleBin.Init)
init.add_eqn(u,  ufun)
init.add_eqn(v,  vfun)

refine = gf.create_function("refine", ScheduleBin.EstimateError)
regrid_error = gf.decl("regrid_error", [], centering=Centering.CCC, from_thorn='CarpetXRegrid')
refine.add_eqn(regrid_error, 9/((x-20)**2 + (y-20)**2))

wave_zero = gf.create_function("WaveZero", ScheduleBin.Analysis)
wave_zero.add_eqn(ZeroVal, u - ufun)


check_zero = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier('CheckZeroGroup'),
    at_or_in=AtOrIn.At,
    schedule_bin=Identifier('analysis'),
    description=String('Do the check'),
    after=[Identifier('RicZero')]
)

gf.bake(
    do_recycle_temporaries=False,
    do_cse=True,
    temporary_promotion_strategy=promote_none(),
    cse_optimization_level=CseOptimizationLevel.Fast
)

CppCarpetXWizard(
    gf,
    CppCarpetXGenerator(
        gf,
        interior_sync_mode=InteriorSyncMode.MixedRhs,
        extra_schedule_blocks=[check_zero]
    )
).generate_thorn()
