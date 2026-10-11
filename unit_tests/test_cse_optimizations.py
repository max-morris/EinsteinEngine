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
import warnings
from typing import Any

from sympy import Expr, Rational, cse as sympy_cse

from EinsteinEngine import *
from EinsteinEngine.common.collect import op_count
from EinsteinEngine.common.cse_optimizations import CsePreset, cse_basic, cse_collect_greedy, default_cse_optimizations
from EinsteinEngine.common.sympywrap import cse, mk_symbol, sqrt
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef, ThornFunction
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
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


def test_default_is_basic_collect_after() -> None:
    assert default_cse_optimizations() == CsePreset.BasicCollectAfter()
    assert eqns_of(bake_thorn()) == eqns_of(bake_thorn(cse_optimizations=CsePreset.BasicCollectAfter()))
    assert eqns_of(bake_thorn()) != eqns_of(bake_thorn(cse_optimizations=[]))


def test_basic_optimizations() -> None:
    fun = bake_thorn(cse_optimizations=[cse_basic()])
    assert {"o1", "o2", "o3"} <= {lhs for lhs, _ in eqns_of(fun)}


def test_collect_greedy_reduces_operations() -> None:
    plain = total_ops(bake_thorn(cse_optimizations=[]))
    collected = total_ops(bake_thorn(cse_optimizations=[cse_collect_greedy()]))
    assert collected < plain, (collected, plain)


def test_collect_greedy_budget() -> None:
    plain = eqns_of(bake_thorn(cse_optimizations=[]))
    # A zero budget makes no moves.
    assert eqns_of(bake_thorn(cse_optimizations=[cse_collect_greedy(max_iter=0)])) == plain
    assert eqns_of(bake_thorn(cse_optimizations=[cse_collect_greedy(max_iter=1)])) != plain
    assert eqns_of(bake_thorn(cse_optimizations=[cse_collect_greedy()])) == \
        eqns_of(bake_thorn(cse_optimizations=[cse_collect_greedy(max_iter=25)]))


def test_both_cse_paths_take_the_option() -> None:
    for isolate in (True, False):
        plain = total_ops(bake_thorn(cse_isolate_grid_variables=isolate, cse_optimizations=[]))
        collected = total_ops(bake_thorn(cse_isolate_grid_variables=isolate, cse_optimizations=[cse_collect_greedy()]))
        assert collected < plain, (isolate, collected, plain)


def test_deprecated_cse_optimization_level() -> None:
    for level, isolate in ((CseOptimizationLevel.Optimal, True), (CseOptimizationLevel.Fast, False)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old = eqns_of(bake_thorn(cse_optimization_level=level))
        assert any(issubclass(w.category, DeprecationWarning) and "cse_isolate_grid_variables" in str(w.message)
                   for w in caught), caught
        assert old == eqns_of(bake_thorn(cse_isolate_grid_variables=isolate))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            bake_thorn(cse_optimization_level=CseOptimizationLevel.Fast, cse_isolate_grid_variables=True)
        except DslException as e:
            assert "contradicts" in str(e), e
        else:
            raise AssertionError("a contradictory cse_optimization_level was accepted")


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


def test_collect_after_cse() -> None:
    exprs = [x * a + x * b + c * (a - b), mk_symbol("y") * a + mk_symbol("y") * b + (a - b) ** 2]
    plain_subs, plain_exprs = cse(exprs)
    subs, reduced = cse(exprs, [cse_collect_greedy(after_cse=True)])
    # Collecting after the CSE leaves the sharing it found alone...
    assert [lhs for lhs, _ in subs] == [lhs for lhs, _ in plain_subs]
    # ...and only makes each expression cheaper, so the whole is never worse than plain CSE.
    ops = sum(op_count(e) for e in reduced) + sum(op_count(rhs) for _, rhs in subs)
    plain_ops = sum(op_count(e) for e in plain_exprs) + sum(op_count(rhs) for _, rhs in plain_subs)
    assert ops < plain_ops, (ops, plain_ops)
    # Collecting before the CSE can expose sharing that collecting after cannot: here a + b.
    before_subs, _ = cse(exprs, [cse_collect_greedy()])
    assert a + b in [rhs for _, rhs in before_subs], before_subs
    assert total_ops(bake_thorn(cse_optimizations=[cse_collect_greedy(after_cse=True)])) < \
        total_ops(bake_thorn(cse_optimizations=[]))


def test_presets() -> None:
    # The order is what a tuner indexes; appending is fine, reordering is not.
    assert [p.name for p in CsePreset] == ["Off", "Basic", "CollectBefore", "CollectBeforeBasic", "CollectAfter",
                                           "BasicCollectAfter"]
    assert CsePreset.CollectAfter(max_iter=7) == [cse_collect_greedy(max_iter=7, after_cse=True)]
    assert CsePreset.BasicCollectAfter(max_iter=7) == [cse_collect_greedy(max_iter=7, after_cse=True), cse_basic()]
    assert CsePreset.Off() == []
    assert CsePreset.Basic() == [cse_basic()]
    assert CsePreset.CollectBefore() == [cse_collect_greedy()]
    assert CsePreset.CollectBeforeBasic(max_iter=7) == [cse_collect_greedy(max_iter=7), cse_basic()]
    assert eqns_of(bake_thorn(cse_optimizations=CsePreset.CollectBeforeBasic(max_iter=3))) == \
        eqns_of(bake_thorn(cse_optimizations=[cse_collect_greedy(max_iter=3), cse_basic()]))


if __name__ == "__main__":
    tests = [
        test_cse_basic_matches_sympy,
        test_default_is_basic_collect_after,
        test_basic_optimizations,
        test_collect_greedy_reduces_operations,
        test_collect_greedy_budget,
        test_both_cse_paths_take_the_option,
        test_deprecated_cse_optimization_level,
        test_derivation_takes_the_option,
        test_collect_after_cse,
        test_presets,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print("All CSE optimization tests passed.")
