"""
The waveequation! It can't be solved too many times.
"""

from dsl.use_indices import *
from typing import cast, Any
from sympy import Expr, Idx, cos, sin
from emit.code.code_tree import Centering
from nrpy.helpers.conditional_file_updater import ConditionalFileUpdater
import nrpy.helpers.conditional_file_updater as cfu
from math import pi
import os

from generators.wizards import CppCarpetXWizard

cfu.verbose = True


def flat_metric(out: Expr, ni: Idx, nj: Idx) -> Expr:
    i = to_num(ni)
    j = to_num(nj)
    if i == 2 or j == 2:
        return sympify(0)
    elif i == j:
        return sympify(1)
    else:
        return sympify(0)


# Create a set of grid functions
gf = ThornDef("TestWave", "WaveEqn")

# Use a NRPy calculated stencil instead
# of simply calling functions such as divx()
gf.set_div_stencil(5)

# Declare gfs
v_t = gf.decl("v_t", [], Centering.VVC)
v = gf.decl("v", [], Centering.VVC, rhs=v_t)
u_t = gf.decl("u_t", [], Centering.VVC)
u = gf.decl("u", [], Centering.VVC, rhs=u_t)

# Declare the metric
g = gf.decl("g", [li, lj])
gf.add_sym(g[li, lj], li, lj)

# Declare params
spd = gf.add_param("spd", default=1.0, desc="The wave speed")
kx = gf.add_param("kx", default=pi / 20, desc="The wave number in the x-direction")
ky = gf.add_param("ky", default=pi / 20, desc="The wave number in the y-direction")

# Fill in values
gf.fill_in(g[li, lj], flat_metric)
gf.fill_in(g[ui, uj], flat_metric)

x, y, z = gf.coords()

# Add the equations we want to evolve.
fun = gf.create_function("newwave_evo", ScheduleBin.Evolve)
fun.add_eqn(v_t, u)
fun.add_eqn(u_t, spd ** 2 * g[ui, uj] * div(v, li, lj))
print('*** ThornFunction wave_evo:')

fun.bake()

# Dump
fun.dump()

# Show tensortypes
fun.show_tensortypes()

# Again for wave_init
fun = gf.create_function("newwave_init", ScheduleBin.Init)
fun.add_eqn(v, sin(kx * x) * sin(ky * y))
fun.add_eqn(u, sympify(0))  # kx**2 * ky**2 * sin(kx * x) * sin(ky * y))
print('*** ThornFunction wave_init:')
fun.bake()
fun.dump()
fun.show_tensortypes()

fun = gf.create_function("refine", ScheduleBin.EstimateError)
regrid_error = gf.decl("regrid_error", [], Centering.CCC)
#fun.add_eqn(regrid_error, 2*v*v) #10/((x-20)**2 + (y-20)**2))
fun.add_eqn(regrid_error, 10/((x-20)**2 + (y-20)**2))
fun.bake(do_cse=False)

CppCarpetXWizard(gf).generate_thorn()
