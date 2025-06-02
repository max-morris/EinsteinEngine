#!/usr/bin/env python3
from EmitCactus import *

"""
The waveequation! It can't be solved too many times.
"""

from sympy import Expr, Idx, Matrix
import nrpy.helpers.conditional_file_updater as cfu
from math import pi


# If we change our configuration, this will show us diffs of the
# new output and the old.
cfu.verbose = True


flat_metric = mkMatrix([
    [1,0,0],
    [0,1,0],
    [0,0,0]])

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
g = gf.decl("g", [li, lj], symmetries=[(li, lj)])

# Declare params
spd = gf.add_param("spd", default=1.0, desc="The wave speed")
kx = gf.add_param("kx", default=pi / 20, desc="The wave number in the x-direction")
ky = gf.add_param("ky", default=pi / 20, desc="The wave number in the y-direction")
amp = gf.add_param("amp", default=10, desc="The amplitude")
# c = w/k
w = spd*sqrt(kx**2 + ky**2)

# Fill in values
gf.mk_subst(g[li, lj], flat_metric)
gf.mk_subst(g[ui, uj], flat_metric)

# stencil(la) -> [stencil(f,1,0,0), stencil(f,0,1,0), stencil(f,0,0,1)]

mdiv = gf.mk_stencil("mdiv",
    la,la,(-2*stencil(0)+stencil(la)+stencil(-la))*DDI(la)**2,
    la,lb,(stencil(la+lb)-stencil(la-lb)-stencil(lb-la)+stencil(-la-lb))*DDI(la)*DDI(lb))
#max = gf.declfun("max", args=2, is_stencil=False)

## gf.mk_stencil(mydiv,la,la,(stencil(la)-2*stencil(0)+stencil(-la))/(DD[la]**2))
## gf.mk_stencil(mydiv,la,lb,(stencil(la+lb)-stencil(la-lb)+stencil(-la-lb)-stencil(-la+lb))/(2*DD[la]*DD[lb]))
## # la + lb, if la==l0 and lb==l1, [1,0,0] + [0,1,0] => [1,1,0]
## # la + l0, if la==l0, [1,0,0] + [1,0,0] => [2,0,0]
## # la + l0, if la==l1, [0,1,0] + [1,0,0] => [1,1,0]
# gf.mk_stencil(mydiv,la,c*stencil(2*la)+b*stencil(la)+a*stencil(0))/(2*DD[la]))
## # if la==l0, (c*stencil(f,2,0,0) + b*stencil(f,1,0,0) + a*stencil(f,0,0,0))/(2*DX)

t, x, y, z = gf.mk_coords(with_time=True)

# Add the equations we want to evolve.
fun = gf.create_function("newwave_evo", ScheduleBin.Evolve)
fun.add_eqn(v_t, u)
fun.add_eqn(u_t, spd ** 2 * g[ui, uj] * D(v, li, lj))
print('*** ThornFunction wave_evo:')
fun.bake()

# Dump
fun.dump()

# Show tensortypes
fun.show_tensortypes()

# Again for wave_init
# du/dt = spd**2 * ((d/dx)**2 u + (d/dy)**2 u)
# dv/dt = u
vfun = amp*sin(kx * x) * sin(ky * y) * sin(w * t)
ufun = vfun.diff(t) #max(vfun.diff(t), -2*amp*w)
fun = gf.create_function("newwave_init", ScheduleBin.Init)
fun.add_eqn(u,  ufun)
fun.add_eqn(v,  vfun)
fun.bake()
#fun.dump()
#fun.show_tensortypes()

fun = gf.create_function("refine", ScheduleBin.EstimateError)
regrid_error = gf.decl("regrid_error", [], centering=Centering.CCC, from_thorn='CarpetXRegrid')
#fun.add_eqn(regrid_error, 2*v*v)
fun.add_eqn(regrid_error, 9/((x-20)**2 + (y-20)**2))
fun.bake()

fun = gf.create_function("WaveZero", ScheduleBin.Analysis)
fun.add_eqn(ZeroVal, u - ufun)
fun.bake()

fun.dump()

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
        interior_sync_mode=InteriorSyncMode.IgnoreRhs,
        extra_schedule_blocks=[ check_zero ]
    )
).generate_thorn()
