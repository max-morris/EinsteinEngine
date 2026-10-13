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
Tests for rank_by_post_population: the ranking of add_eqn calls by median post-CSE position, and the run of a recipe
that finds those positions (probe.derive_add_eqn_order). Run as a plain script.
"""

import functools
import os
import runpy
import tempfile
import textwrap
from collections.abc import Callable
from typing import Any

from sympy import Symbol

from EinsteinEngine.common.describe_param import describe_param
from EinsteinEngine.intermediate.eqn_grouping import median_rank_key, rank_add_eqn_calls
from EinsteinEngine.intermediate.eqn_ordering import add_eqn_order, lexicographical_order, prioritize_rare_symbols
from EinsteinEngine.tuning import probe, tuning
from EinsteinEngine.tuning.probe import RankByPostPopulation, derive_add_eqn_order, rank_by_post_population

PRELUDE = """\
from EinsteinEngine import *
from EinsteinEngine.tuning.tuning import get_optional_tuning_param
"""


def sym(name: str) -> Symbol:
    return Symbol(name)  # type: ignore[no-untyped-call]


def expect_runtime_error(fn: Callable[[], object], fragment: str) -> None:
    try:
        fn()
    except RuntimeError as e:
        assert fragment in str(e), f"Expected '{fragment}' in the message, got: {e}"
        return
    raise AssertionError(f"Expected a RuntimeError mentioning '{fragment}'")


def write_recipe(directory: str, name: str, body: str) -> str:
    path = os.path.join(directory, f"{name}.py")
    with open(path, "w") as fh:
        fh.write(PRELUDE + textwrap.dedent(body))
    return path


def run_recipe(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run the recipe as a trial would, with `params` as its tuning params, and return its globals."""
    tuning._tuning_params = params
    try:
        return runpy.run_path(path, run_name="__main__")
    finally:
        tuning._tuning_params = None


def pre_population_origins(fun: Any) -> list[list[int]]:
    """For each list, the origins of its equations in pre-population order, each group once."""
    result: list[list[int]] = list()
    for el in fun.eqn_complex.eqn_lists:
        seen: list[int] = list()
        for lhs in el.eqn_pre_population_order:
            if (origin := el.eqn_origin.get(lhs)) is not None and origin not in seen:
                seen.append(origin)
        result.append(seen)
    return result


def test_median_rank_key() -> None:
    assert median_rank_key([3, 1, 2], 7) == (2.0, 7)
    assert median_rank_key([4, 1], 0) == (2.5, 0)
    assert median_rank_key([], 1) == (float('inf'), 1)
    assert median_rank_key([10**6], 0) < median_rank_key([], 0)


def test_rank_add_eqn_calls() -> None:
    """A hand-computed ranking: medians, ties, calls without surviving equations, several lists, empty calls."""
    a, b, c, d, e, f, g, t = (sym(n) for n in "abcdefgt")
    # List 0's order is [t, a, b, c, d, e]; t (e.g. a CSE temp) has no origin. Call 0 = {c} (median 3), call 1 =
    # {a, e} (median (1 + 5) / 2 = 3), call 2 = {b, d} (median (2 + 4) / 2 = 3). All three tie at 3, so they stay in
    # call-index order. Call 3 = {f}, which is not in the order (optimized away), so it goes last in this list.
    origins0 = {a: 1, b: 2, c: 0, d: 2, e: 1, f: 3}
    order0 = [t, a, b, c, d, e]
    assert rank_add_eqn_calls([origins0], [order0], 4) == [0, 1, 2, 3]
    # Move b and d earlier: call 2's median becomes (0 + 1) / 2 = 0.5, ahead of calls 0 and 1.
    assert rank_add_eqn_calls([origins0], [[b, d, t, a, c, e]], 4) == [2, 0, 1, 3]
    # A second list with calls 4 and 5 is ranked on its own and appended; call 6 produced no equation at all.
    origins1 = {g: 5, sym("h"): 4}
    assert rank_add_eqn_calls([origins0, origins1], [order0, [g, sym("h")]], 7) == [0, 1, 2, 3, 5, 4, 6]
    # A call with no equation at all (call 0 here) ranks with the last list's calls that have none in the order (call
    # 3), by call index, as in the offline derivation, which gave both the median inf.
    assert rank_add_eqn_calls([{a: 1, b: 2, f: 3}], [[b, a]], 4) == [2, 1, 0, 3]
    assert rank_add_eqn_calls([{a: 1}, {b: 2, f: 3}], [[a], [b]], 4) == [1, 2, 0, 3]


WORKED_EXAMPLE = """\
gf = ThornDef("ARR", "WORKED")
v = gf.decl("v", [li])
u, a, src = gf.decl("u", []), gf.decl("a", []), gf.decl("src", [])
fun = gf.create_function("fn", ScheduleBin.Evolve)
fun.add_eqn(u, src)
fun.add_eqn(v[li], [src * 2, src * 3, src * 4])
fun.add_eqn(a, u + 1)
gf.bake(do_cse=False, ordering_fn=get_optional_tuning_param("ordering_fn", lexicographical_order),
        functions={"fn": {"early_ordering_fn": get_optional_tuning_param("early_ordering_fn", None)}})
"""


def test_worked_example(tmp: str) -> None:
    """Calls 0: u = src, 1: v[li] = [...], 2: a = u + 1. Lexicographically the order is [u, a, vD0, vD1, vD2] (u moved
    in front of a, which reads it), so the medians are 0, 3 and 1, and the order [0, 2, 1]."""
    path = write_recipe(tmp, "worked", WORKED_EXAMPLE)
    assert derive_add_eqn_order(path, lexicographical_order) == ("fn", (0, 2, 1))
    assert probe.active_params() is None
    # The derived order drives the early locus like any add_eqn_order.
    fun = run_recipe(path, {"early_ordering_fn": add_eqn_order([0, 2, 1])})["fun"]
    assert pre_population_origins(fun) == [[0, 2, 1]], pre_population_origins(fun)


def test_pull_out_temps_do_not_count(tmp: str) -> None:
    """
    Only the equations an add_eqn call produced count toward its median, as in the offline derivation (which
    snapshotted the list around each call). The pull-out temporary inherits its call's origin but does not count.

    Calls 0: qc = pull_out(src * src) * 2, 1: qb = src + 1, 2: qa = src * 3. Lexicographically the order is
    [pull_out_0, qa, qb, qc] (pull_out_0 sorts first and qc reads it anyway). Counting qc alone, the medians are 3, 2
    and 1, so the order is [2, 1, 0]. Counting the pull-out temporary too, call 0's median would be (0 + 3) / 2 = 1.5,
    and the order [2, 0, 1].
    """
    path = write_recipe(tmp, "pull_out", """\
        gf = ThornDef("ARR", "PULLOUTCOUNT")
        qa, qb, qc, src = gf.decl("qa", []), gf.decl("qb", []), gf.decl("qc", []), gf.decl("src", [])
        fun = gf.create_function("fn", ScheduleBin.Evolve)
        fun.add_eqn(qc, pull_out(src * src) * 2)
        fun.add_eqn(qb, src + 1)
        fun.add_eqn(qa, src * 3)
        gf.bake(do_cse=False, ordering_fn=get_optional_tuning_param("ordering_fn", lexicographical_order),
                early_ordering_fn=get_optional_tuning_param("early_ordering_fn", None))
        """)
    assert derive_add_eqn_order(path, lexicographical_order) == ("fn", (2, 1, 0))


def test_whole_thorn(tmp: str) -> None:
    """
    Two functions sharing subexpressions (so global CSE makes global temporaries, with synthetic functions, as well as
    local and tile ones), pull-outs in both, a manual soft split and an auto hard split in the ranked function. The
    derivation runs without the split predicate and with global CSE over both functions. The manual soft split keeps
    calls 0 and 1 in list 0 and calls 2 and 3 in list 1, where prioritize_rare_symbols puts call 3 first.
    """
    path = write_recipe(tmp, "whole_thorn", """\
        gf = ThornDef("ARR", "WHOLETHORN")
        v = gf.decl("v", [li])
        a, b, c, d, e, src, src2 = (gf.decl(n, []) for n in ("a", "b", "c", "d", "e", "src", "src2"))
        p = gf.add_param("p", 1.0, "a param")
        f1 = gf.create_function("f1", ScheduleBin.Evolve,
                                auto_hard_split_predicate=get_optional_tuning_param("hard", lambda i: i == 0))
        f1.add_eqn(a, sin(src) * cos(src) + pull_out(src * src))
        f1.add_eqn(v[li], [sin(src2) * p, cos(src2) * cos(src), sin(src) * cos(src2) * exp(src)])
        f1.soft_split()
        f1.add_eqn(b, exp(src2) * cos(src) + sin(src) * cos(src) * v[l0])
        f1.add_eqn(c, src * p + exp(src2) * cos(src) + a)
        f2 = gf.create_function("f2", ScheduleBin.Evolve)
        f2.add_eqn(d, sin(src) * cos(src2) * exp(src) + exp(src2) * cos(src) + pull_out(src2 * src2))
        f2.add_eqn(e, d * sin(src) * cos(src) + cos(src2) * cos(src))
        # No CSE optimizations: the assertions below depend on the derived order differing from the recipe
        #  order, which it does for this thorn's plain CSE.
        gf.bake(do_cse=True, temporary_promotion_strategy=promote_all(), cse_optimizations=[],
                functions={"f1": {"early_ordering_fn": get_optional_tuning_param("early_ordering_fn", None),
                                  "ordering_fn": get_optional_tuning_param("ordering_fn", maximize_symbol_reuse)}})
        """)
    # The recipe's own hard split predicate is not evaluated while deriving.
    assert derive_add_eqn_order(path, prioritize_rare_symbols) == ("f1", (0, 1, 3, 2))
    f1 = run_recipe(path, {"early_ordering_fn": add_eqn_order([0, 1, 3, 2])})["f1"]
    assert pre_population_origins(f1) == [[0], [1, 3, 2]], pre_population_origins(f1)  # the soft split merges


def test_derivation_errors(tmp: str) -> None:
    ignores = write_recipe(tmp, "ignores_ordering_fn", """\
        gf = ThornDef("ARR", "IGNORES")
        a, src = gf.decl("a", []), gf.decl("src", [])
        gf.create_function("fn", ScheduleBin.Evolve).add_eqn(a, sin(src))
        gf.bake(early_ordering_fn=get_optional_tuning_param("early_ordering_fn", None))
        """)
    expect_runtime_error(lambda: derive_add_eqn_order(ignores, prioritize_rare_symbols), "never read")
    two = write_recipe(tmp, "two_functions", """\
        gf = ThornDef("ARR", "TWOFNS")
        a, b, src = gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
        gf.create_function("f1", ScheduleBin.Evolve).add_eqn(a, sin(src))
        gf.create_function("f2", ScheduleBin.Evolve).add_eqn(b, cos(src))
        gf.bake(ordering_fn=get_optional_tuning_param("ordering_fn", maximize_symbol_reuse),
                early_ordering_fn=get_optional_tuning_param("early_ordering_fn", None))
        """)
    expect_runtime_error(lambda: derive_add_eqn_order(two, prioritize_rare_symbols), "exactly one function")
    assert probe.active_params() is None


def test_repr() -> None:
    ranked = rank_by_post_population(prioritize_rare_symbols)
    assert isinstance(ranked, RankByPostPopulation) and ranked.ordering_fn is prioritize_rare_symbols
    expected = 'rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols)'
    assert repr(ranked) == expected, repr(ranked)
    assert describe_param(ranked) == expected
    partial_fn = functools.partial(prioritize_rare_symbols, complexity_factor=0.5)
    assert repr(rank_by_post_population(partial_fn)) == \
        'rank_by_post_population(partial(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols, complexity_factor=0.5))'


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            for test in (test_median_rank_key, test_rank_add_eqn_calls, test_repr):
                test()
                print(f"ok: {test.__name__}")
            for tmp_test in (test_worked_example, test_pull_out_temps_do_not_count, test_whole_thorn,
                             test_derivation_errors):
                tmp_test(tmp)
                print(f"ok: {tmp_test.__name__}")
        finally:
            os.chdir(cwd)
    print("All rank_by_post_population tests passed.")
