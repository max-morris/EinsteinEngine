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
shared by several Piecewise expressions stays inline where it is used, and its numeric operands stay shared.
Run as a plain script.
"""

from typing import Sequence

from typing import cast

from sympy import And, Basic, Expr, Piecewise, Symbol, cse as sympy_cse, exp
from sympy.core.parameters import evaluate

from EinsteinEngine import *
from EinsteinEngine.common.collect import op_count
from EinsteinEngine.common.cse_optimizations import CsePreset
from EinsteinEngine.common.sympywrap import CseOptimization, cse, is_condition, mk_symbol
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef
from EinsteinEngine.frontend.util import cse_isolate

a, b, c, d, x = (mk_symbol(n) for n in "abcdx")


def pw(*pairs: tuple[object, object]) -> Expr:
    """A typed Piecewise (conditions are Booleans, which EE's mk_piecewise does not accept)."""
    return cast(Expr, Piecewise(*pairs))  # type: ignore[no-untyped-call]


def expand(replacements: list[tuple[Symbol, Expr]], reduced: list[Expr]) -> list[Basic]:
    """The reduced expressions with every replacement substituted back (in reverse order of definition)."""
    out: list[Basic] = list(reduced)
    for lhs, rhs in reversed(replacements):
        out = [e.xreplace({lhs: rhs}) for e in out]  # type: ignore[no-untyped-call]
    return out


def check(exprs: list[Expr], optimizations: Sequence[CseOptimization] = ()) \
        -> tuple[list[tuple[Symbol, Expr]], list[Expr]]:
    replacements, reduced = cse(exprs, optimizations)
    assert not any(is_condition(rhs) for _, rhs in replacements), replacements
    assert expand(replacements, reduced) == exprs, (replacements, reduced)
    return replacements, reduced


def occurrences(sub: Basic, replacements: list[tuple[Symbol, Expr]], reduced: list[Expr]) -> int:
    """How many times `sub` is computed in the CSE output."""
    return sum(e.count(sub) for e in [rhs for _, rhs in replacements] + reduced)  # type: ignore[no-untyped-call]


def test_shared_condition_stays_inline() -> None:
    # sympy's own cse makes `x > 0` a replacement here.
    _, reduced = check([pw((a, x > 0), (b, True)), pw((b, x > 0), (c, True))])
    assert all(e.has(x > 0) for e in reduced)


def test_operands_stay_shared() -> None:
    # sympy extracts the whole relational, so its operand has no temporary of its own; it must get one.
    e = a * c + sqrt(a * b + c * d)
    exprs = [pw((a, e > 0), (b, True)), pw((b, e > 0), (c, True)), pw((c, e > 0), (d, True))]
    replacements, reduced = check(exprs)
    assert occurrences(e, replacements, reduced) == 1, (replacements, reduced)


def test_condition_on_a_numeric_temporary() -> None:
    # a + b is a numeric temporary (kept); the shared condition on it is inlined and refers to it.
    exprs = [pw((a, a + b > 0), (c, True)), pw((c, a + b > 0), (a, True)), (a + b) ** 2]
    replacements, reduced = check(exprs)
    assert occurrences(a + b, replacements, reduced) == 1, (replacements, reduced)


def test_temporary_using_a_condition() -> None:
    # A numeric temporary (the shared Piecewise) that contains a dropped condition.
    p = pw((a * b, x > 0), (c, True))
    check([p + d, p * d, pw((d, x > 0), (a, True))])


def test_nested_conditions() -> None:
    inner = a * b + a * c > 0
    outer = And(inner, x < 1)
    exprs = [pw((a, outer), (b, True)), pw((b, outer), (c, True)), pw((c, inner), (d, True)),
             pw((d, inner), (a, True))]
    replacements, reduced = check(exprs)
    assert occurrences(a * b + a * c, replacements, reduced) == 1, (replacements, reduced)


def test_with_optimizations() -> None:
    e = a * b + a * c
    for preset in CsePreset:
        replacements, _ = cse([pw((e + c, e > 0), (b, True)), pw((e * c, e > 0), (c, True))], preset())
        assert not any(is_condition(rhs) for _, rhs in replacements), (preset, replacements)


def test_structure_preserved() -> None:
    # Substituting a condition must not re-evaluate what it rebuilds: the result is sympy's own output (including the
    # postprocessors' work, e.g. CollectAfter's 3*(c + d + p)) with the conditions put back, node for node.
    shared = [3 * c + 3 * d + 3 * pw((a, x > 0), (b, True)), pw((b, x > 0), (a, True)) * d]
    for preset in CsePreset:
        raw_replacements, raw_reduced = sympy_cse(shared, optimizations=preset())  # type: ignore[no-untyped-call]
        conds = {lhs: rhs for lhs, rhs in raw_replacements if is_condition(rhs)}
        with evaluate(False):  # type: ignore[no-untyped-call]
            expected = [e.xreplace(conds) for e in raw_reduced]
        replacements, reduced = cse(shared, preset())
        assert reduced == expected and not replacements, (preset, reduced, expected)


def test_isolated_grid_variables() -> None:
    # Conditions on an isolated grid variable (cse_isolate wraps it in an undefined function, which is not hoisted).
    g = mk_symbol("g")
    exprs = [pw((a, g > 0), (b, True)) * g, pw((b, g > 0), (c, True)) + g,
             pw((c, g + a > 0), (a, True)), pw((a, g + a > 0), (b, True))]
    replacements, reduced = cse_isolate(exprs, symbols_to_isolate=[g])
    assert not any(is_condition(rhs) for _, rhs in replacements), replacements
    assert expand(replacements, reduced) == exprs, (replacements, reduced)
    assert occurrences(g + a, replacements, reduced) <= 1, (replacements, reduced)


def test_expensive_function_operand_shared() -> None:
    # exp(x) is a function of an atom but not trivial: as a condition operand it must be computed once.
    exprs = [pw((a, exp(x) > 1), (b, True)), pw((b, exp(x) > 1), (c, True))]
    replacements, reduced = check(exprs)
    assert occurrences(exp(x), replacements, reduced) == 1, (replacements, reduced)


def test_bake_with_shared_condition() -> None:
    # Before the fix, the global CSE made `s > 0` a temporary and the rebake failed. Both global-CSE paths.
    for isolate in (True, False):
        rhses = bake(isolate, shared=True)
        assert not any(is_condition(rhs) for rhs in rhses), rhses
        assert sum(" > 0" in str(rhs) for rhs in rhses) == 2, rhses  # the condition stays inline in both


def test_bake_structure_preserved() -> None:
    # Inlining a shared condition must cost no more than an unshared one, through the whole bake (with isolation,
    # cse_isolate's restoring substitution must not undo the default preset's factoring of o1).
    for isolate in (True, False):
        shared = sum(op_count(rhs) for rhs in bake(isolate, shared=True))
        unshared = sum(op_count(rhs) for rhs in bake(isolate, shared=False))
        assert shared <= unshared, (isolate, shared, unshared)


_bakes = 0


def bake(isolate: bool, shared: bool) -> list[Expr]:
    global _bakes
    _bakes += 1
    gf = ThornDef("ARR", f"CSEBOOLEANS{_bakes}")
    u, v, s = gf.decl("u", []), gf.decl("v", []), gf.decl("s", [])
    o1, o2 = gf.decl("o1", []), gf.decl("o2", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(o1, 3 * u + 3 * v + 3 * pw((u, s > 0), (v, True)))
    fun.add_eqn(o2, pw((v, s > 0 if shared else s > 1), (u, True)) * u * v)
    gf.bake(cse_isolate_grid_variables=isolate)
    return [el.eqns[lhs] for el in fun.eqn_complex.eqn_lists for lhs in el.order]


if __name__ == "__main__":
    tests = [
        test_shared_condition_stays_inline,
        test_operands_stay_shared,
        test_condition_on_a_numeric_temporary,
        test_temporary_using_a_condition,
        test_nested_conditions,
        test_with_optimizations,
        test_structure_preserved,
        test_isolated_grid_variables,
        test_expensive_function_operand_shared,
        test_bake_with_shared_condition,
        test_bake_structure_preserved,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print("All CSE Boolean tests passed.")
