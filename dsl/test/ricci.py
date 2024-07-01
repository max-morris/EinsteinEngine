from dsl.use_indices import *
from emit.ccl.interface.interface_visitor import InterfaceVisitor
from emit.ccl.param.param_visitor import ParamVisitor
from emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from emit.code.cpp.cpp_visitor import CppVisitor
from typing import cast, Any
from sympy import Expr, Idx, cos, sin
from emit.code.code_tree import Centering
from generators.wizards import CppCarpetXWizard

# Create a set of grid functions
gf = ThornDef("TestRicci", "Ricci")

# Declare gfs
g = gf.decl("metric_lower", [li, lj], Centering.VVC)
G = gf.decl("Affine", [ua, lb, lc], Centering.VVC)
Ric = gf.decl("Ric", [la, lb], Centering.VVC)
iter3 = gf.decl("iter3", [la, lb, lc], Centering.VVC)
iter4 = gf.decl("iter4", [la, lb, lc, ld], Centering.VVC)

gf.add_sym(g[li, lj], li, lj)
gf.add_sym(G[ua, lb, lc], lb, lc)
gf.add_sym(iter3[ua, lb, lc], lb, lc)
gf.add_sym(iter4[la, lb, lc, ld], la, lb)

div1 = gf.declfun("div1", True)

# Need to automate these
gf.fill_in(g[la, lb], lambda _, i, j: mkSymbol(f"gDD{to_num(i)}{to_num(j)}"))
gf.fill_in(iter3[lc, la, lb], alt=div1(g[la, lb], lc),
           f=lambda _, a, b, c: mkSymbol(f"gDD{to_num(a)}{to_num(b)}_dD{to_num(c)}"))
gf.fill_in(g[ua, ub], lambda _, i, j: mkSymbol(f"gUU{to_num(i)}{to_num(j)}"))
gf.fill_in(iter3[lc, ua, ub], alt=div1(g[ua, ub], lc),
           f=lambda _, a, b, c: mkSymbol(f"gUU{to_num(a)}{to_num(b)}_dD{to_num(c)}"))
gf.fill_in(G[la, lb, lc], lambda _, a, b, c: mkSymbol(f"affDDD{to_num(a)}{to_num(b)}{to_num(c)}"))
gf.fill_in(G[ua, lb, lc], lambda _, a, b, c: mkSymbol(f"affUDD{to_num(a)}{to_num(b)}{to_num(c)}"))
gf.fill_in(Ric[la, lb], lambda _, a, b: mkSymbol(f"RicDD{to_num(a)}{to_num(b)}"))
gf.fill_in(iter4[la, lb, lc, ud], alt=div1(G[ud, la, lb], lc),
           f=lambda _, a, b, c, d: mkSymbol(f"affUDD{to_num(d)}{to_num(a)}{to_num(b)}_dD{to_num(c)}"))

fun = gf.create_function("setGL", ScheduleBin.Analysis)
fun.add_eqn(G[la, lb, lc], div1(g[lb, lc], la) + div1(g[la, lc], lb) - div1(g[la, lb], lc))
fun2 = gf.create_function("setGU", ScheduleBin.Analysis)
fun2.add_eqn(G[ua, lb, lc], g[ua, ud] * G[ld, lb, lc])
fun3 = gf.create_function("setRic", ScheduleBin.Analysis)
fun3.add_eqn(Ric[li, lj],
             div1(G[ua, li, lj], la) - div1(G[ua, la, li], lj) + G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[
                 ub, la, lj])

# Ensure the equations make sense
fun.diagnose()
fun2.diagnose()
fun3.diagnose()

CppCarpetXWizard(gf).generate_thorn()
