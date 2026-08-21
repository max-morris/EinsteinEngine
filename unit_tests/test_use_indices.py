#  Copyright (C) 2025-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
#
#  This file is part of the Einstein Engine (EinsteinEngine).
#
#  EinsteinEngine is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  EinsteinEngine is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Literal

from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sympy import IndexedBase, Symbol, Basic, Expr

from EinsteinEngine import *
from EinsteinEngine.common.sympywrap import *
from EinsteinEngine.frontend.definitions import *
from EinsteinEngine.frontend.dsl import symmetries
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef
from EinsteinEngine.frontend.dsl.use_indices import IndexContractionVisitor, InvalidIndexError, IndexTracker
from EinsteinEngine.frontend.dsl.finite_difference import DivMakerVisitor

val = mk_symbol("val")
x = mk_symbol("x")
y = mk_symbol("y")
z = mk_symbol("z")


def assert_eq(a: Expr, b: Expr) -> None:
    assert a is not None
    r = simplify(a - b)
    assert r == 0, f"{a} minus {b} !=0, instead {r}"


if __name__ == "__main__":
    gf = ThornDef("ARR", "TST")
    B = gf.decl("B", [lc, lb])
    gf._add_sym(B[la, lb], la, lb, 1)
    M = gf.decl("M", [la, lb])
    gf._add_sym(M[la, lb], la, lb, -1)
    V = gf.decl("V", [la])
    A = gf.decl("A", [ub, la, lc])
    gf._add_sym(A[ua, lb, lc], lb, lc, 1)


    kdelta = gf.mk_kdelta()

    ####
    fail_expr = mk_symbol("fail_expr")


    def testerr(gf: ThornDef, in_expr: Expr, result_expr: Expr) -> None:
        result_expr = gf._do_subs(result_expr)
        viz = IndexContractionVisitor(
            dict(),
            dimensionality=gf.einstein_notation.dimensionality,
            lookup_pair=gf.einstein_notation.lookup_pair,
        )
        try:
            expr, it = viz.visit(in_expr)
            expr = gf._do_subs(expr)
        except InvalidIndexError as iie:
            print(iie)
            it = IndexTracker(gf.einstein_notation.lookup_pair)
            expr = fail_expr
        zero_expr = simplify(expr - result_expr)
        if zero_expr == 0:
            result_color: Literal['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']
            if result_expr == fail_expr:
                result_color = "red"
            else:
                result_color = "green"
            print(colorize("success:", "green"), in_expr, colorize("->", "cyan"), colorize(result_expr, result_color))
        else:
            print(colorize("FAIL", "red"))
            print(colorize(in_expr, "yellow"))
            print(colorize(result_expr, "cyan"))
            print(colorize(expr, "green"))
            print(colorize(it, "blue"))
            raise Exception(colorize("Fail", "red"))


    testerr(gf, M[la, ub] * B[lb, uc], M[la, u0] * B[l0, uc] + M[la, u1] * B[l1, uc] + M[la, u2] * B[l2, uc])
    testerr(gf, M[la, ua] * B[ub, lb], (M[l0, u0] + M[l1, u1] + M[l2, u2]) * (B[u0, l0] + B[u1, l1] + B[u2, l2]))
    testerr(gf, sqrt(M[la, ua]) * B[ub, lb],
            sqrt(M[l0, u0] + M[l1, u1] + M[l2, u2]) * (B[u0, l0] + B[u1, l1] + B[u2, l2]))
    testerr(gf, M[la, ua] * (1 + B[ub, lb]),
            (M[l0, u0] + M[l1, u1] + M[l2, u2]) * (1 + B[u0, l0] + B[u1, l1] + B[u2, l2]))
    testerr(gf, M[la, ua] * B[la, ua], fail_expr)
    testerr(gf, M[ua, lb] + 1, fail_expr)
    testerr(gf, sqrt(V[ua]), fail_expr)
    testerr(gf, sqrt(V[ua] * V[la]), sqrt(V[u0] * V[l0] + V[u1] * V[l1] + V[u2] * V[l2]))
    testerr(gf, B[la, lb] * V[ua] * V[ub],
            B[l0, l0] * V[u0] ** 2 + B[l1, l1] * V[u1] ** 2 + B[l2, l2] * V[u2] ** 2 + 2 * B[l0, l1] * V[u0] * V[
                u1] + 2 * B[l1, l2] * V[u1] * V[u2] + 2 * B[l0, l2] * V[u0] * V[u2])
    testerr(gf, M[la, lb] * V[ua] * V[ub], zero)
    testerr(gf, A[ua, lb, la] * V[ub],
            (A[u0, l0, l0] + A[u1, l0, l1] + A[u2, l0, l2]) * V[u0] + (A[u0, l0, l1] + A[u1, l1, l1] + A[u2, l1, l2]) *
            V[u1] + (A[u0, l0, l2] + A[u1, l1, l2] + A[u2, l2, l2]) * V[u2])
    testerr(gf, div(A[ua, l0, l0], la), div(A[u0, l0, l0], l0) + div(A[u1, l0, l0], l1) + div(A[u2, l0, l0], l2))
    testerr(gf, div(A[ua, la, l0], l0), div(A[u0, l0, l0], l0) + div(A[u1, l0, l1], l0) + div(A[u2, l0, l2], l0))
    ####

    # Anti-Symmetric

    n = 0
    for out in gf.expand_eqn(mk_eq(M[la, lb], B[la, lb])):
        print(out)
        n += 1
    #assert n == get_dimension(), f"n = {n}"

    # Symmetric
    N = gf.decl("N", [la, lb])
    gf._add_sym(N[la, lb], la, lb, 1)

    n = 0
    for out in gf.expand_eqn(mk_eq(N[la, lb], B[la, lb])):
        print(out)
        n += 1
    #assert n == get_dimension() * (get_dimension() - 1)

    # Non-Symmetric
    Q = gf.decl("Q", [la, lb])
    gf.add_substitution_rule(Q[ua,ub])

    n = 0
    for out in gf.expand_eqn(mk_eq(Q[la, lb], B[la, lb])):
        print(out)
        n += 1
    #assert n == get_dimension() ** 2

    a = gf.decl("a", [])
    b = gf.decl("b", [])
    c = gf.decl("c", [])
    w = gf.decl("w", [])
    k = gf.decl("k", [la])
    o0 = gf.decl("o0", [])
    o1 = gf.decl("o1", [la])
    o2 = gf.decl("o2", [la])
    gf.add_substitution_rule(o2[ua])
    gf.add_substitution_rule(o1[ub])
    gf.add_substitution_rule(k[la])
    foofunc = gf.create_function("foo", ScheduleBin.Analysis)
    #foofunc.add_eqn(a, sympify(get_dimension()))
    foofunc.add_eqn(b, a + sympify(2))

    # Test of custom derivative operation mdiv
    mdiv = gf.mk_stencil("mdiv", la, (stencil(la) - stencil(0)) * DDI(la))
    foofunc.add_eqn( k[la],  mdiv(a ** 5 * w, la))

    # Upwind/downwind derivateve based on ol[la].
    osdiv = gf.mk_stencil("osdiv", la, h_step( kdelta[la,lb]*o1[ub])*finite_difference_stencil(4,1,1,la) +
                                       h_step(-kdelta[la,lb]*o1[ub])*finite_difference_stencil(4,1,-1,la))

    o0func = gf.create_function("set_o0", ScheduleBin.Analysis, schedule_before=["foo"])
    o0func.add_eqn(o0,
            h_step(x-1)*(x-1) + h_step(1-x)*(x+1) +
            h_step(y-1)*(y-1) + h_step(1-y)*(y+1) +
            h_step(z-1)*(z-1) + h_step(1-z)*(z+1)
    )

    # The result should be that o2[la] is 1 everywhere.
    foofunc.add_eqn(o2[la], osdiv(o0, la))
    foofunc.add_eqn(o2[ua], o2[lb]*kdelta[ua,ub])

    def getsym(a: IndexedBase) -> Symbol:
        b = a.args[0]
        assert isinstance(b, Symbol)
        return b


    # Baked equations refer to the underlying scalar Symbol of a declaration,
    # not the (zero-index) IndexedBase returned by decl(), so compare against
    # getsym(a)/getsym(w) here.
    a_sym = getsym(a)
    w_sym = getsym(w)
    kd0eqn = foofunc._eqn_list.eqns.get(mk_symbol("kD0"), None)
    assert kd0eqn == 5 * DXI * (-stencil(a_sym, 0, 0, 0) + stencil(a_sym, 1, 0, 0)) * a_sym ** 4 * w_sym + DXI * (
            -stencil(w_sym, 0, 0, 0) + stencil(w_sym, 1, 0, 0)) * a_sym ** 5


    # Now test functions
    fmax = gf.decl_fun("fmax", 2)
    foofunc.add_eqn(c, fmax(a, b))
    gf.bake()


dmv = DivMakerVisitor(div)
dmv2 = DivMakerVisitor(D)


def do_div(expr: Basic) -> Expr:
    r = dmv.visit(expr, no_idx)
    r = dmv2.visit(r, no_idx)
    assert isinstance(r, Expr)
    return r


if __name__ == "__main__":
    foo = mk_indexed_base("foo", (1,))
    gxx = mk_symbol("gxx")
    gxy = mk_symbol("gxy")
    gyy = mk_symbol("gyy")
    gzz = mk_symbol("gzz")
    gyz = mk_symbol("gyz")
    gxz = mk_symbol("gxz")
    f = mk_function("f")
    fp = mk_function("f'")

    expr1 = div(gxx ** 2, l0, l0)
    expr2 = 2 * div(gxx, l0) ** 2 + 2 * gxx * div(gxx, l0, l0)
    assert_eq(do_div(expr1), expr2)

    expr1 = - gyy * div(-gxz, l0) - gyy * div(gxz, l0)
    expr2 = zero
    assert_eq(do_div(expr1), expr2)

    expr1 = -2 * gxy * div(gxy, l2) + div(gxy ** 2, l2)
    expr2 = zero
    assert_eq(do_div(expr1), expr2)

    expr1 = div(-gxy ** 2, l2)
    expr2 = -2 * gxy * div(gxy, l2)
    assert_eq(do_div(expr1), expr2)

    expr1 = div(gxx * gyy - gxy ** 2, l2)
    expr2 = gxx * div(gyy, l2) + div(gxx, l2) * gyy - 2 * gxy * div(gxy, l2)
    assert_eq(do_div(expr1), expr2)

    assert_eq(do_div(div(x, l0)), one)
    assert_eq(do_div(div(y, l0)), zero)
    assert_eq(do_div(div(x ** 3, l0)), 3 * x ** 2)

    assert_eq(do_div(div(sin(x), l0)), cos(x))
    assert_eq(do_div(div(cos(x), l0)), -sin(x))
    assert_eq(do_div(div(tan(x), l0)), sec(x) ** 2)
    assert_eq(do_div(div(cot(x), l0)), -csc(x) ** 2)
    assert_eq(do_div(div(sec(x), l0)), sec(x) * tan(x))
    assert_eq(do_div(div(csc(x), l0)), -csc(x) * cot(x))

    assert_eq(do_div(div(sinh(x), l0)), cosh(x))
    assert_eq(do_div(div(cosh(x), l0)), sinh(x))
    assert_eq(do_div(div(tanh(x), l0)), sech(x) ** 2)
    assert_eq(do_div(div(coth(x), l0)), -csch(x) ** 2)
    assert_eq(do_div(div(sech(x), l0)), -sech(x) * tanh(x))
    assert_eq(do_div(div(csch(x), l0)), -csch(x) * coth(x))

    assert_eq(do_div(div(erf(x), l0)), 2 * exp(-x ** 2) / sqrt(pi))

    assert_eq(do_div(div(x ** 2 + x ** 3, l0)), 2 * x + 3 * x ** 2)
    assert_eq(do_div(div(x ** 2 + x ** 3, l1)), zero)
    assert_eq(do_div(div(1 / (2 + x ** 2), l0)), -2 * x / (2 + x ** 2) ** 2)
    assert_eq(do_div(div(x ** 3 / (2 + x ** 2), l0)), -2 * x ** 4 / (2 + x ** 2) ** 2 + 3 * x ** 2 / (2 + x ** 2))
    assert_eq(do_div(div(x ** 2 * sin(x), l0)), x * (x * cos(x) + 2 * sin(x)))
    assert_eq(do_div(div(sin(x ** 3), l0)), cos(x ** 3) * 3 * x ** 2)
    assert_eq(do_div(div(foo[la], l0)), div(foo[la], l0))
    assert_eq(do_div(div(div(foo[la], lb), lc)), div(foo[la], lb, lc))
    assert_eq(do_div(div(div(foo[la], lc), lb)), div(foo[la], lb, lc))
    assert_eq(do_div(div(div(foo[la], l0), l1)), div(foo[la], l0, l1))
    assert_eq(do_div(div(div(foo[la], l1), l0)), div(foo[la], l0, l1))
    assert_eq(do_div(x * div(x, l0)), x)
    assert_eq(do_div(x + div(x, l0)), x + 1)
    assert_eq(do_div(x + x * div(x, l0)), 2 * x)
    assert_eq(do_div(x * (x + x * div(x, l0))), 2 * x ** 2)
    assert_eq(do_div(x * (x / 2 + 3 * x * div(x, l0))), (7 / 2) * x ** 2)
    assert_eq(do_div(div(foo, l0)), div(foo, l0))
    assert_eq(do_div(div(gxx, l1) / 2 + div(gxy, l0)), div(gxx, l1) / 2 + div(gxy, l0))
    assert_eq(do_div(div(exp(x), l0)), exp(x))
    assert_eq(do_div(div(exp(x) / 2, l0)), exp(x) / 2)
    expr = (gxy * gyz - gxz * gyy) * (-div(gxx, l2) / 2 + div(gxz, l0))
    assert_eq(do_div(expr), expr)
    expr1 = (gxy * gyz - gxz * gyy + gzz) * (-div(gxx, l2) / 2 + div(gxz + gzz, l0))
    expr2 = (gxy * gyz - gxz * gyy + gzz) * (-div(gxx, l2) / 2 + div(gxz, l0) + div(gzz, l0))
    assert_eq(do_div(expr1), expr2)
    expr = (gxx * gyz - gxy * gxz) * (div(gxx, l1) - 2 * div(gxy, l0)) / 2 + (gxy * gyz - gxz * gyy) * div(gxx, l0) / 2
    assert_eq(do_div(expr), expr)
    expr1 = div(gxx * gyy, l0)
    expr2 = div(gxx, l0) * gyy + div(gyy, l0) * gxx
    assert_eq(do_div(expr1), expr2)
    expr1 = x ** 6 / 3 + sin(x) / x
    expr2 = 2 * x ** 5 + cos(x) / x - sin(x) / x ** 2
    assert_eq(do_div(div(expr1, l0)), expr2)
    expr1 = (x + sin(x)) * (1 / x + cos(x))
    expr2 = (1 + cos(x)) * (1 / x + cos(x)) - (1 / x ** 2 + sin(x)) * (x + sin(x))
    assert_eq(do_div(div(expr1, l0)), expr2)
    expr1 = 1 / (x + sin(x))
    expr2 = -(1 + cos(x)) / (x + sin(x)) ** 2
    assert_eq(do_div(div(expr1, l0)), expr2)
    assert_eq(do_div(div(sqrt(x), l0)), 1 / sqrt(x) / 2)
    assert_eq(do_div(D(f(x), l0)), fp(x))
    assert_eq(do_div(D(f(x ** 2 + y), l0)), 2 * x * fp(x ** 2 + y))
    assert_eq(do_div(D(f(x ** 2 + y), l1)), fp(x ** 2 + y))
    assert_eq(do_div(D(f(x ** 2 + y), l2)), zero)
    assert_eq(do_div(D(f(x + f(x)), l0)), fp(x + f(x)) * (1 + fp(x)))
