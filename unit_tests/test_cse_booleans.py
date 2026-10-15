#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
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

"""
Tests that CSE never makes a Boolean (a relational or a logical combination of them) into a temporary: a condition
shared by several Piecewise expressions stays inline where it is used. Run as a plain script.
"""

from sympy import And, Expr, Piecewise

from EinsteinEngine import *
from EinsteinEngine.common.cse_optimizations import CsePreset
from EinsteinEngine.common.sympywrap import cse, is_condition, mk_symbol
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef

a, b, c, x = (mk_symbol(n) for n in "abcx")


def expand(replacements: list[tuple[Expr, Expr]], reduced: list[Expr]) -> list[Expr]:
    """The reduced expressions with every replacement substituted back (in reverse order of definition)."""
    out = list(reduced)
    for lhs, rhs in reversed(replacements):
        out = [e.xreplace({lhs: rhs}) for e in out]
    return out


def check(exprs: list[Expr], **kw: object) -> tuple[list[tuple[Expr, Expr]], list[Expr]]:
    replacements, reduced = cse(exprs, **kw)  # type: ignore[arg-type]
    assert not any(is_condition(rhs) for _, rhs in replacements), replacements
    assert expand(replacements, reduced) == exprs, (replacements, reduced)
    return replacements, reduced


def test_shared_condition_stays_inline() -> None:
    # sympy's own cse makes `x > 0` a replacement here.
    _, reduced = check([Piecewise((a, x > 0), (b, True)), Piecewise((b, x > 0), (c, True))])
    assert all(e.has(x > 0) for e in reduced)


def test_condition_on_a_numeric_temporary() -> None:
    # a + b is a numeric temporary (kept); the shared condition on it is inlined and refers to it.
    exprs = [Piecewise((a, a + b > 0), (c, True)), Piecewise((c, a + b > 0), (a, True)), (a + b) ** 2]
    replacements, _ = check(exprs)
    assert any(rhs == a + b for _, rhs in replacements), replacements


def test_logical_combination() -> None:
    cond = And(x > 0, a < 1)
    check([Piecewise((a, cond), (b, True)), Piecewise((b, cond), (c, True))])


def test_with_optimizations() -> None:
    for preset in CsePreset:
        check([Piecewise((a * b + c, x > 0), (b, True)), Piecewise((a * b + c, x > 0), (c, True))],
              optimizations=preset())


def test_bake_with_shared_condition() -> None:
    # Before the fix, the global CSE made `s > 0` a temporary and the rebake failed.
    gf = ThornDef("ARR", "CSEBOOLEANS")
    u, v, s = gf.decl("u", []), gf.decl("v", []), gf.decl("s", [])
    o1, o2 = gf.decl("o1", []), gf.decl("o2", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(o1, Piecewise((u, s > 0), (v, True)) + u * v)
    fun.add_eqn(o2, Piecewise((v, s > 0), (u, True)) * u * v)
    gf.bake()
    rhses = [el.eqns[lhs] for el in fun.eqn_complex.eqn_lists for lhs in el.order]
    assert not any(is_condition(rhs) for rhs in rhses), rhses
    assert sum("s > 0" in str(rhs) for rhs in rhses) == 2, rhses  # the condition stays inline in both


if __name__ == "__main__":
    tests = [
        test_shared_condition_stays_inline,
        test_condition_on_a_numeric_temporary,
        test_logical_combination,
        test_with_optimizations,
        test_bake_with_shared_condition,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print("All CSE Boolean tests passed.")
