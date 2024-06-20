"""
The waveequation! It can't be solved too many times.
"""

from dsl.use_indices import *
from emit.code.cpp.cpp_visitor import CppVisitor
from typing import cast, Any
from sympy import Expr, Idx, cos, sin
from emit.code.code_tree import Centering
from generators.cpp_carpetx_generator import CppCarpetXGenerator


def flat_metric(out: Expr, ni: Idx, nj: Idx) -> Expr:
    i = to_num(ni)
    j = to_num(nj)
    if i == j:
        return sympify(1)
    else:
        return sympify(0)


# Create a set of grid functions
gf = ThornDef("waveqn")

# Declare gfs
p = gf.decl("p", [li], Centering.VVC)
p_t = gf.decl("p_t", [li], Centering.VVC)
p_d = gf.decl("p_d", [li, lj], Centering.VVC)
u = gf.decl("u", [], Centering.VVC)
u_t = gf.declscalar("u_t")
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
fun = gf.create_function("wave_evo", "EVO")
fun.add_eqn(p_t[lj], spd * div1(u, lj)) # Evo should be an enum
fun.add_eqn(u_t, spd * g[ui, uj] * div1(p[lj], li))

fun = gf.create_function("wave_init", "INIT")
fun.add_eqn(u, sin(kx*x)*cos(ky*y) )
fun.add_eqn(p[lj], sympify(0))

# Ensure the equations make sense
fun.diagnose()

# Display the equations in final form
# gf.dump()

# Perform cse
fun.cse()

# Display again in case there are changes
fun.dump()

fun.show_tensortypes()

carpetx_generator = CppCarpetXGenerator('wave_evol', gf)
tree = carpetx_generator.generate_code()

visitor = CppVisitor()
code = visitor.visit(tree)

print(code)
