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
Tests for the CSE optimizations (EinsteinEngine.common.cse_optimizations).
Run as a plain script.
"""

import os
import tempfile
from typing import Any

from sympy import Expr, Rational, cse as sympy_cse

from EinsteinEngine import *
from EinsteinEngine.common.collect import op_count
from EinsteinEngine.common.cse_optimizations import cse_basic, cse_collect_greedy
from EinsteinEngine.common.sympywrap import cse, mk_symbol, sqrt
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef, ThornFunction
from EinsteinEngine.intermediate.eqn_ordering import prioritize_rare_symbols
from EinsteinEngine.tuning.probe import derive_add_eqn_order

a, b, c, x = (mk_symbol(n) for n in "abcx")


def test_cse_basic_matches_sympy() -> None:
    # cse_basic() folds sympy's two "basic" pairs into one, which must not change what CSE produces.
    for exprs in [[(a - b) * c + (a - b) ** 2, c * (a - b)],
                  [x * a - x * b + (a - b) / c, sqrt(a - b) * (b - a)],
                  [a - b, -(a - b) * Rational(3, 2) + c * (b - a)]]:
        assert cse(exprs, [cse_basic()]) == sympy_cse(exprs, optimizations="basic"), exprs  # type: ignore[no-untyped-call]
        assert cse(exprs) == sympy_cse(exprs), exprs  # type: ignore[no-untyped-call]


_thorn_counter = 0


def bake_thorn(**opts: Any) -> ThornFunction:
    """A function whose sums share grid variables in some of their terms, baked with `opts`."""
    global _thorn_counter
    _thorn_counter += 1
    gf = ThornDef("ARR", f"CSEOPTS{_thorn_counter}")
    u, v, w, s = gf.decl("u", []), gf.decl("v", []), gf.decl("w", []), gf.decl("s", [])
    o1, o2, o3 = gf.decl("o1", []), gf.decl("o2", []), gf.decl("o3", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(o1, u * sin(s) + u * cos(s) + v * w)
    fun.add_eqn(o2, v * sin(s) + v * cos(s) + u * w * sin(s))
    fun.add_eqn(o3, (u - v) * w + (u - v) * s + w * s * sin(s))
    gf.bake(**opts)
    return fun


def eqns_of(fun: ThornFunction) -> list[tuple[str, str]]:
    return [(str(lhs), str(el.eqns[lhs])) for el in fun.eqn_complex.eqn_lists for lhs in el.order]


def total_ops(fun: ThornFunction) -> int:
    return sum(op_count(rhs) for el in fun.eqn_complex.eqn_lists for rhs in el.eqns.values())


def test_default_is_no_optimizations() -> None:
    assert eqns_of(bake_thorn()) == eqns_of(bake_thorn(cse_optimizations=[]))


def test_basic_optimizations() -> None:
    fun = bake_thorn(cse_optimizations=[cse_basic()])
    assert {"o1", "o2", "o3"} <= {lhs for lhs, _ in eqns_of(fun)}


def test_collect_greedy_reduces_operations() -> None:
    plain = total_ops(bake_thorn())
    collected = total_ops(bake_thorn(cse_optimizations=[cse_collect_greedy()]))
    assert collected < plain, (collected, plain)


def test_collect_greedy_budget() -> None:
    plain = eqns_of(bake_thorn())
    # A zero budget makes no moves.
    assert eqns_of(bake_thorn(cse_optimizations=[cse_collect_greedy(max_iter=0)])) == plain
    assert eqns_of(bake_thorn(cse_optimizations=[cse_collect_greedy(max_iter=1)])) != plain
    assert eqns_of(bake_thorn(cse_optimizations=[cse_collect_greedy()])) == \
        eqns_of(bake_thorn(cse_optimizations=[cse_collect_greedy(max_iter=25)]))


def test_both_cse_levels_take_the_option() -> None:
    for level in (CseOptimizationLevel.Optimal, CseOptimizationLevel.Fast):
        plain = total_ops(bake_thorn(cse_optimization_level=level))
        collected = total_ops(bake_thorn(cse_optimization_level=level, cse_optimizations=[cse_collect_greedy()]))
        assert collected < plain, (level, collected, plain)


DERIVE_RECIPE = """\
from EinsteinEngine import *
from EinsteinEngine.tuning.tuning import get_optional_tuning_param
gf = ThornDef("ARR", "CSEOPTSDERIVE")
u, v, w, s = gf.decl("u", []), gf.decl("v", []), gf.decl("w", []), gf.decl("s", [])
o1, o2, o3 = gf.decl("o1", []), gf.decl("o2", []), gf.decl("o3", [])
fun = gf.create_function("fn", ScheduleBin.Evolve)
fun.add_eqn(o1, u * sin(s) + u * cos(s) + v * w)
fun.add_eqn(o2, v * sin(s) + v * cos(s) + u * w * sin(s))
fun.add_eqn(o3, (u - v) * w + (u - v) * s + w * s * sin(s))
gf.bake(cse_optimizations=get_optional_tuning_param("cse_optimizations", []),
        functions={"fn": {"early_ordering_fn": get_optional_tuning_param("early_ordering_fn", None),
                          "ordering_fn": get_optional_tuning_param("ordering_fn", prioritize_rare_symbols)}})
"""


def test_derivation_takes_the_option() -> None:
    # A rank_by_post_population order is derived from a run of the recipe that gets the tuner's other probe params
    # (tuning.build_experiment), so a tuner that sets the CSE optimizations ranks the add_eqn calls by that CSE.
    calls = [0]

    def counting(expr: Expr) -> Expr:
        calls[0] += 1
        return expr

    bake_thorn(cse_optimizations=[(counting, None)])
    one_bake = calls[0]
    assert one_bake > 0
    calls[0] = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "derive.py")
        with open(path, "w") as fh:
            fh.write(DERIVE_RECIPE)
        derive_add_eqn_order(path, prioritize_rare_symbols, {"cse_optimizations": [(counting, None)]})
    assert calls[0] == one_bake, (calls[0], one_bake)


if __name__ == "__main__":
    tests = [
        test_cse_basic_matches_sympy,
        test_default_is_no_optimizations,
        test_basic_optimizations,
        test_collect_greedy_reduces_operations,
        test_collect_greedy_budget,
        test_both_cse_levels_take_the_option,
        test_derivation_takes_the_option,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print("All CSE optimization tests passed.")
