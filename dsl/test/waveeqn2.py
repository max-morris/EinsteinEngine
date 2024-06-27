"""
The waveequation! It can't be solved too many times.
"""

from dsl.use_indices import *
from emit.ccl.interface.interface_visitor import InterfaceVisitor
from emit.ccl.param.param_visitor import ParamVisitor
from emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from emit.code.cpp.cpp_visitor import CppVisitor
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
    if i == j:
        return sympify(1)
    else:
        return sympify(0)


# Create a set of grid functions
gf = ThornDef("TestWave", "WaveEqn")

# Declare gfs
v_t = gf.decl("v_t", [li], Centering.VVC)
v = gf.decl("v", [li], Centering.VVC, rhs=v_t)
u_t = gf.decl("u_t", [], Centering.VVC)
u = gf.decl("u", [], Centering.VVC, rhs=u_t)

siter2 = gf.decl("siter2", [li, lj])
gf.add_sym(siter2[li, lj], li, lj)
iter1 = gf.decl("iter1", [li])

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

# Fill in the deriv variables with a function call
#
div1 = gf.declfun("div1", True)
divx = gf.declfun("divx", True)
divy = gf.declfun("divy", True)
divz = gf.declfun("divz", True)
div2 = gf.declfun("div2", True)
divxx = gf.declfun("divxx", True)
divyy = gf.declfun("divyy", True)
divzz = gf.declfun("divzz", True)


def to_div(out: Expr, j: Idx) -> Expr:
    n = to_num(j)
    ret: Any
    if n == 0:
        ret = divx(*out.args[:-1])
    elif n == 1:
        ret = divy(*out.args[:-1])
    elif n == 2:
        ret = divz(*out.args[:-1])
    else:
        assert False
    return cast(Expr, ret)


def to_div2(out: Expr, i: Idx, j: Idx) -> Expr:
    n = to_num(j)
    ret: Any
    if n == 0:
        ret = divxx(v)
    elif n == 1:
        ret = divyy(v)
    elif n == 2:
        ret = sympify(0)  # divzz(v)
    else:
        assert False
    return cast(Expr, ret)


gf.fill_in(iter1[lj], alt=div1(u, lj), f=to_div)
gf.fill_in(siter2[li, lj], alt=div2(v, li, lj), f=to_div2)

x, y, z = gf.coords()

# Add the equations we want to evolve.
fun = gf.create_function("newwave_evo", ScheduleBin.Evolve)
fun.add_eqn(v_t, u)
fun.add_eqn(u_t, spd ** 2 * g[ui, uj] * div2(v, li, lj))
print('*** ThornFunction wave_evo:')

# Ensure the equations make sense
fun.diagnose()

# Perform cse
# fun.cse()

# Dump
fun.dump()

# Show tensortypes
fun.show_tensortypes()

# Again for wave_init
fun = gf.create_function("newwave_init", ScheduleBin.Init)
fun.add_eqn(v, sin(kx * x) * sin(ky * y))
fun.add_eqn(u, sympify(0))  # kx**2 * ky**2 * sin(kx * x) * sin(ky * y))
print('*** ThornFunction wave_init:')
fun.diagnose()
# fun.cse()
fun.dump()
fun.show_tensortypes()

fun = gf.create_function("refine", ScheduleBin.EstimateError)
regrid_error = gf.decl("regrid_error", [], Centering.CCC)
fun.add_eqn(regrid_error, 10/((x-20)**2 + (y-20)**2))
fun.diagnose()

CppCarpetXWizard(gf).generate_thorn()
