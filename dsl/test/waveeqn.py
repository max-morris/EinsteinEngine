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
from generators.cpp_carpetx_generator import CppCarpetXGenerator
from nrpy.helpers.conditional_file_updater import ConditionalFileUpdater
import nrpy.helpers.conditional_file_updater as cfu
import os

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
p_t = gf.decl("p_t", [li], Centering.VVC)
p = gf.decl("p", [li], Centering.VVC, rhs=p_t)
p_d = gf.decl("p_d", [li, lj], Centering.VVC)
u_t = gf.decl("u_t", [], Centering.VVC)
u = gf.decl("u", [], Centering.VVC, rhs=u_t)
u_d = gf.decl("u_d", [ui], Centering.VVC)

siter2 = gf.decl("siter2", [li, lj])
gf.add_sym(siter2[li, lj], li, lj)
iter1 = gf.decl("iter1", [li])

# Declare the metric
g = gf.decl("g", [li, lj])
gf.add_sym(g[li, lj], li, lj)

# Declare params
spd = gf.add_param("spd", default=1.0, desc="The wave speed")
kx = gf.add_param("kx", default=1.0, desc="The wave number in the x-direction")
ky = gf.add_param("ky", default=1.0, desc="The wave number in the y-direction")

# Fill in values
gf.fill_in(g[li, lj], flat_metric)
gf.fill_in(g[ui, uj], flat_metric)

# Fill in with defaults
gf.fill_in(p[li], lambda _, i: mkSymbol(f"pD{to_num(i)}"))
gf.fill_in(p_t[li], lambda _, i: mkSymbol(f"p_tD{to_num(i)}"))

# Fill in the deriv variables with a function call
#
div1 = gf.declfun("div1", True)
divx = gf.declfun("divx", True)
divy = gf.declfun("divy", True)
divz = gf.declfun("divz", True)
gf.fill_in(p_d[li, lj], lambda _, i, j: div1(p[j], i))
gf.fill_in(u_d[li], lambda _, i: div1(u, i))


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
        ret = divx(p[i])
    elif n == 1:
        ret = divy(p[i])
    elif n == 2:
        ret = divz(p[i])
    else:
        assert False
    return cast(Expr, ret)


gf.fill_in(iter1[lj], alt=div1(u, lj), f=to_div)
gf.fill_in(siter2[li, lj], alt=div1(p[li], lj), f=to_div2)

x, y, z = gf.coords()

res = gf.do_subs(spd * g[ui, uj] * div1(p[lj], li))

# Add the equations we want to evolve.
fun = gf.create_function("wave_evo", ScheduleBin.EVOL)
fun.add_eqn(p_t[lj], spd * div1(u, lj))
fun.add_eqn(u_t, spd * g[ui, uj] * div1(p[lj], li))
print('*** ThornFunction wave_evo:')

# Ensure the equations make sense
fun.diagnose()

# Perform cse
fun.cse()

# Dump
fun.dump()

# Show tensortypes
fun.show_tensortypes()

# Again for wave_init
fun = gf.create_function("wave_init", ScheduleBin.INIT)
fun.add_eqn(u, sin(kx * x) * cos(ky * y))
fun.add_eqn(p[lj], sympify(0))
print('*** ThornFunction wave_init:')
fun.diagnose()
fun.cse()
fun.dump()
fun.show_tensortypes()


def cppCarpetXGenerator(gf):
    base_dir = os.path.join(gf.arrangement, gf.name)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "src"), exist_ok=True)

    carpetx_generator = CppCarpetXGenerator(gf)

    for fn_name in gf.thorn_functions.keys():
        print('=====================')
        code_tree = carpetx_generator.generate_function_code(fn_name)
        code = CppVisitor(carpetx_generator).visit(code_tree)
        #print(code)
        code_fname = os.path.join(base_dir, "src", carpetx_generator.get_src_file_name(fn_name))
        with ConditionalFileUpdater(code_fname) as fd:
            fd.write(code)

    print('== param.ccl ==')
    param_tree = carpetx_generator.generate_param_ccl()
    param_ccl = ParamVisitor().visit(param_tree)
    #print(param_ccl)
    param_ccl_fname = os.path.join(base_dir, "param.ccl")
    with ConditionalFileUpdater(param_ccl_fname) as fd:
        fd.write(param_ccl)

    print('== interface.ccl ==')
    interface_tree = carpetx_generator.generate_interface_ccl()
    interface_ccl = InterfaceVisitor().visit(interface_tree)
    #print(interface_ccl)
    interface_ccl_fname = os.path.join(base_dir, "interface.ccl")
    with ConditionalFileUpdater(interface_ccl_fname) as fd:
        fd.write(interface_ccl)

    print('== schedule.ccl ==')
    schedule_tree = carpetx_generator.generate_schedule_ccl()
    schedule_ccl = ScheduleVisitor().visit(schedule_tree)
    #print(schedule_ccl)
    schedule_ccl_fname = os.path.join(base_dir, "schedule.ccl")
    with ConditionalFileUpdater(schedule_ccl_fname) as fd:
        fd.write(schedule_ccl)

    print('== make.code.defn ==')
    makefile = carpetx_generator.generate_makefile()
    #print(makefile)
    makefile_fname = os.path.join(base_dir, "src/make.code.defn")
    with ConditionalFileUpdater(makefile_fname) as fd:
        fd.write(makefile)

cppCarpetXGenerator(gf)
