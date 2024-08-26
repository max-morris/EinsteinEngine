from dsl.use_indices import *
from dsl.sympywrap import mkMatrix, do_sympify, do_inv, do_diff, do_sympify
from emit.ccl.interface.interface_visitor import InterfaceVisitor
from emit.ccl.param.param_visitor import ParamVisitor
from emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from emit.code.cpp.cpp_visitor import CppVisitor
from typing import cast, Any
from sympy import Expr, Idx, cos, sin, sympify
from emit.code.code_tree import Centering
from generators.wizards import CppCarpetXWizard

# Create a set of grid functions
gf = ThornDef("TestRic", "TestRicVal")

a = gf.add_param("a", default=10.0, desc="Just a constant")
b = gf.add_param("b", default=0.2, desc="Just a constant")

# Declare gfs
g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
Ric = gf.decl("Ric", [la, lb], from_thorn="Ricci")
RicVal = gf.decl("RicVal", [la, lb])

gf.add_sym(g[li, lj], li, lj)
gf.add_sym(Ric[li, lj], li, lj)
gf.add_sym(RicVal[li, lj], li, lj)

gf.mk_subst(g[la, lb], mksymbol_for_tensor_xyz)
gf.mk_subst(RicVal[li,lj])
gf.mk_subst(Ric[li,lj])

x, y, z = gf.mk_coords()

fun = gf.create_function("MetricSet", ScheduleBin.Analysis, schedule_before=["setGL"])
fun.add_eqn(g[li,lj],mkMatrix([
 [a+b*x**2,        0, 0],
 [       0, a+b*x**2, 0],
 [       0,        0, 1]]))
fun.bake()

fun = gf.create_function("RicZero", ScheduleBin.Analysis, schedule_after=["setGL"])
fun.add_eqn(RicVal[l0,l0], Ric[l0,l0]-b*(-a + b*x**2)/(a + b*x**2)**2)
fun.bake()


CppCarpetXWizard(gf).generate_thorn()
