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

set_dimension(4)

# Create a set of grid functions
gf = ThornDef("TestKerr", "Kerr")

# Declare gfs
g = gf.decl("g", [li, lj], Centering.VVC)
G = gf.decl("Affine", [ua, lb, lc], Centering.VVC)
Ric = gf.decl("Ric", [la, lb], Centering.VVC)

gf.add_sym(g[li, lj], li, lj)
gf.add_sym(Ric[li, lj], li, lj)
gf.add_sym(G[ua, lb, lc], lb, lc)

spin = False
a : Expr
if spin:
    # This is very slow
    a = gf.add_param("a", default=0.5, desc="The black hole spin")
else:
    a = do_sympify(0)
m = gf.add_param("m", default=0.5, desc="The black hole mass")
t, r, th, phi = gf.mk_coords()

sigma = r**2 + a**2*cos(th)**2
delta = r**2 - 2*m*r + a**2

gtt = -(1-2*m*r/sigma)
grr = sigma/delta
gqq = sigma
gpp = (r**2 + a**2 + (2*m*r**2*a**2/sigma)*sin(th)**2)*sin(th)**2
gtp = -4*m*r*a*sin(th)**2/sigma

gmat = mkMatrix([
[gtt,   0,   0, gtp],
[  0, grr,   0,   0],
[  0,   0, gqq,   0],
[gtp,   0,   0, gpp]])

gf.mk_subst(g[la,lb], gmat)
imat = do_inv(gmat)
gf.mk_subst(g[ua, ub], imat)
gf.mk_subst(G[la, lb, lc], (div(g[la, lb], lc) + div(g[la, lc], lb) - div(g[lb, lc], la))/2)
gf.mk_subst(G[ud, lb, lc], g[ud,ua]*G[la, lb, lc])
gf.mk_subst(Ric[li, lj],
             div(G[ua, li, lj], la) - div(G[ua, la, li], lj) + 
             G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])

for i in range(4):
    for j in range(i+1,4):
        ixi = [l0, l1, l2, l3][i]
        ixj = [l0, l1, l2, l3][j]
        print("Checking:",Ric[ixi,ixj])
        assert gf.do_subs(Ric[ixi, ixj]) == do_sympify(0)
