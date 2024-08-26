from dsl.use_indices import *
from dsl.sympywrap import mkMatrix, do_inv, do_sympify
from emit.ccl.interface.interface_visitor import InterfaceVisitor
from emit.ccl.param.param_visitor import ParamVisitor
from emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from emit.code.cpp.cpp_visitor import CppVisitor
from typing import cast, Any
from sympy import Expr, Idx, cos, sin
from emit.code.code_tree import Centering
from generators.wizards import CppCarpetXWizard

# Create a set of grid functions
gf = ThornDef("TestAffine", "Affine")
gf.set_div_stencil(3)

# Declare gfs
g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
x,y,z = gf.mk_coords()
G = gf.decl("Affine", [ua, lb, lc])
Ric = gf.decl("Ric", [la, lb])

gf.add_sym(g[li, lj], li, lj)
gf.add_sym(G[ua, lb, lc], lb, lc)
gf.add_sym(Ric[la,lb], la, lb)

gf.mk_subst(g[la, lb], mksymbol_for_tensor_xyz)

gmat = gf.get_matrix(g[la,lb])
imat = do_inv(gmat)
gf.mk_subst(g[ua, ub], imat)
gf.mk_subst(div(g[la, lb], lc)) # div(g[l0,l1],l2) -> gDD01_dD2
gf.mk_subst(div(g[ua, ub], lc))
gf.mk_subst(G[la, lb, lc], (div(g[la, lb], lc) + div(g[la, lc], lb) - div(g[lb, lc], la))/2)
gf.mk_subst(G[ud, lb, lc]) #, g[ud,ua]*G[la, lb, lc])
    
fun = gf.create_function("setAff", ScheduleBin.Analysis)

fun.add_eqn(G[ud, lb, lc], g[ud,ua]*G[la, lb, lc])
# Optimizations:
# Does it pull the 2nd G out of the sum of: G[ua, la, lb] * G[ub, li, lj]? Check
# -1*Foo should be -Foo
# Comment:
# Check non-diagonal metric

# Ensure the equations make sense
fun.bake(do_recycle_temporaries=True)

CppCarpetXWizard(gf).generate_thorn()
