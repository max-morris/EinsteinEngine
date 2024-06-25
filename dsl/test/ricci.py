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

# Create a set of grid functions
gf = ThornDef("Ricci")

# Declare gfs
g = gf.decl("g", [li, lj], Centering.VVC)
G = gf.decl("G", [ua, lb,lc], Centering.VVC)
Ric = gf.decl("Ric", [la,lb], Centering.VVC)
iter3 = gf.decl("iter3", [la,lb,lc], Centering.VVC)
iter4 = gf.decl("iter4", [la,lb,lc,ld], Centering.VVC)

gf.add_sym(g[li, lj], li, lj)
gf.add_sym(G[ua, lb, lc], lb, lc)
gf.add_sym(iter3[ua, lb, lc], lb, lc)
gf.add_sym(iter4[la, lb, lc, ld], la, lb)

gf.fill_in(g[la,lb], lambda _,i,j : mkSymbol(f"gDD{to_num(i)}{to_num(j)}"))
gf.fill_in(iter3[lc,la,lb], alt=div1(g[la,lb], lc), f=lambda _,a,b,c: mkSymbol(f"gDD{to_num(a)}{to_num(b)}_dD{to_num(c)}"))
gf.fill_in(g[ua,ub], lambda _,i,j : mkSymbol(f"gUU{to_num(i)}{to_num(j)}"))
gf.fill_in(iter3[lc,ua,ub], alt=div1(g[ua,ub], lc), f=lambda _,a,b,c: mkSymbol(f"gUU{to_num(a)}{to_num(b)}_dD{to_num(c)}"))
gf.fill_in(G[la,lb,lc], lambda _,a,b,c: mkSymbol(f"affDDD{to_num(a)}{to_num(b)}{to_num(c)}"))
gf.fill_in(G[ua,lb,lc], lambda _,a,b,c: mkSymbol(f"affUDD{to_num(a)}{to_num(b)}{to_num(c)}"))
gf.fill_in(Ric[la,lb], lambda _,a,b: mkSymbol(f"RicDD{to_num(a)}{to_num(b)}"))
gf.fill_in(iter4[la,lb,lc,ud], alt=div1(G[ud,la,lb],lc), f=lambda _,a,b,c,d: mkSymbol(f"affUDD{to_num(d)}{to_num(a)}{to_num(b)}_dD{to_num(c)}"))

div1 = gf.declfun("div1", True)

fun = gf.create_function("setGL", ScheduleBin.ANALYSIS)
fun.add_eqn(G[la,lb,lc], div1(g[lb,lc], la) + div1(g[la,lc], lb) - div1(g[la,lb], lc))
fun2 = gf.create_function("setGU", ScheduleBin.ANALYSIS)
fun2.add_eqn(G[ua,lb,lc], g[ua,ud]*G[ld,lb,lc] )
fun3 = gf.create_function("setRic", ScheduleBin.ANALYSIS)
fun3.add_eqn(Ric[li,lj], div1(G[ua,li,lj],la) - div1(G[ua,la,li], lj) + G[ua,la,lb]*G[ub,li,lj] - G[ua,li,lb]*G[ub,la,lj])

# Ensure the equations make sense
fun.diagnose()
fun2.diagnose()
fun3.diagnose()
#xxx = div1(g[lb,lc], la) + div1(g[la,lc], lb) - div1(g[la,lb], lc)
#xxx = xxx.subs({la:l0, lb:l1, lc:l2})
#for k,v in gf.subs.items():
#    print(k,"-->",v)
#print(gf.do_subs(xxx))

carpetx_generator = CppCarpetXGenerator(gf)

for fn_name in gf.thorn_functions.keys():
    print('=====================')
    code_tree = carpetx_generator.generate_function_code(fn_name)
    code = CppVisitor().visit(code_tree)
    print(code)

print('== param.ccl ==')
param_tree = carpetx_generator.generate_param_ccl()
param_ccl = ParamVisitor().visit(param_tree)
print(param_ccl)

print('== interface.ccl ==')
interface_tree = carpetx_generator.generate_interface_ccl()
interface_ccl = InterfaceVisitor().visit(interface_tree)
print(interface_ccl)

print('== schedule.ccl ==')
schedule_tree = carpetx_generator.generate_schedule_ccl()
schedule_ccl = ScheduleVisitor().visit(schedule_tree)
print(schedule_ccl)

print('== make.code.defn ==')
makefile = carpetx_generator.generate_makefile()
print(makefile)
