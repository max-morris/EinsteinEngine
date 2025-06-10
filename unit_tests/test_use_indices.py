from typing import Literal

from sympy import Expr, IndexedBase, Symbol

from EmitCactus import *
from EmitCactus.dsl.eqnlist import DXI
from EmitCactus.dsl.use_indices import IndexContractionVisitor, InvalidIndexError, IndexTracker, dimension, zero, \
    do_div, x, one, y
from nrpy.helpers.coloring import coloring_is_enabled as colorize


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

    ####
    fail_expr = mkSymbol("fail_expr")


    def testerr(gf: ThornDef, in_expr: Expr, result_expr: Expr) -> None:
        result_expr = gf.do_subs(result_expr)
        viz = IndexContractionVisitor(dict())
        try:
            expr, it = viz.visit(in_expr)
            expr = gf.do_subs(expr)
        except InvalidIndexError as iie:
            print(iie)
            it = IndexTracker()
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
    for out in gf.expand_eqn(mkEq(M[la, lb], B[la, lb])):
        print(out)
        n += 1
    assert n == dimension, f"n = {n}"

    # Symmetric
    N = gf.decl("N", [la, lb])
    gf._add_sym(N[la, lb], la, lb, 1)

    n = 0
    for out in gf.expand_eqn(mkEq(N[la, lb], B[la, lb])):
        print(out)
        n += 1
    assert n == dimension * (dimension - 1)

    # Non-Symmetric
    Q = gf.decl("Q", [la, lb])

    n = 0
    for out in gf.expand_eqn(mkEq(Q[la, lb], B[la, lb])):
        print(out)
        n += 1
    assert n == dimension ** 2

    a = gf.decl("a", [], declare_as_temp=True)
    b = gf.decl("b", [])
    c = gf.decl("c", [])
    k = gf.decl("k", [la])
    gf.add_substitution_rule(k[la])
    foofunc = gf.create_function("foo", ScheduleBin.Analysis)
    foofunc.add_eqn(a, sympify(dimension))
    foofunc.add_eqn(b, a + sympify(2))

    # Test of custom derivative operation mdiv
    mdiv = gf.mk_stencil("mdiv", la, (stencil(la) - stencil(0)) * DDI(la))
    foofunc.add_eqn(k[la], mdiv(a ** 5 * b, la))
    kd0eqn = foofunc.eqn_list.eqns.get(mkSymbol("kD0"), None)
    assert kd0eqn == 5 * DXI * (-stencil(a, 0, 0, 0) + stencil(a, 1, 0, 0)) * a ** 4 * b + DXI * (
            -stencil(b, 0, 0, 0) + stencil(b, 1, 0, 0)) * a ** 5


    def getsym(a: IndexedBase) -> Symbol:
        b = a.args[0]
        assert isinstance(b, Symbol)
        return b


    # Now test functions
    fmax = gf.decl_fun("fmax", 2)
    foofunc.add_eqn(c, fmax(a, b))
    foofunc.bake()
    assert foofunc.eqn_list.depends_on(getsym(c), getsym(a))
    assert foofunc.eqn_list.depends_on(getsym(c), getsym(b))

if __name__ == "__main__":
    foo = mkIndexedBase("foo", (1,))
    gxx = mkSymbol("gxx")
    gxy = mkSymbol("gxy")
    gyy = mkSymbol("gyy")
    gzz = mkSymbol("gzz")
    gyz = mkSymbol("gyz")
    gxz = mkSymbol("gxz")

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
