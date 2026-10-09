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
Tests for rank_by_post_population: the ranking of add_eqn calls by median post-CSE position, the trial bake that finds
those positions, its isolation from the real bake, and the report to the tuning probe. Run as a plain script.
"""

import functools
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Optional

from sympy import Expr, Symbol

from EinsteinEngine import *
from EinsteinEngine.common.describe_param import describe_param
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef, ThornFunction
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.intermediate.eqn_grouping import median_rank_key, rank_add_eqn_calls
from EinsteinEngine.intermediate.eqn_ordering import RankByPostPopulation
from EinsteinEngine.intermediate.eqnlist import EqnList
from EinsteinEngine.tuning import probe


def sym(name: str) -> Symbol:
    return Symbol(name)  # type: ignore[no-untyped-call]


def expect_dsl_exception(fn: Callable[[], object], fragment: str) -> None:
    try:
        fn()
    except DslException as e:
        assert fragment in str(e), f"Expected '{fragment}' in the message, got: {e}"
        return
    raise AssertionError(f"Expected a DslException mentioning '{fragment}'")


def eqns_of(fun: ThornFunction) -> list[list[tuple[str, str]]]:
    return [[(str(lhs), str(el.eqns[lhs])) for lhs in el.order] for el in fun.eqn_complex.eqn_lists]


def pre_population_origins(fun: ThornFunction) -> list[list[int]]:
    """For each list, the origins of its equations in pre-population order, each group once."""
    result: list[list[int]] = list()
    for el in fun.eqn_complex.eqn_lists:
        seen: list[int] = list()
        for lhs in el.eqn_pre_population_order:
            if (origin := el.eqn_origin.get(lhs)) is not None and origin not in seen:
                seen.append(origin)
        result.append(seen)
    return result


class Recorder:
    """A split predicate that records the positions it is queried with."""

    def __init__(self, fire: Callable[[int], bool]) -> None:
        self.fire = fire
        self.calls: list[int] = list()

    def __call__(self, i: int) -> bool:
        self.calls.append(i)
        return self.fire(i)


@contextmanager
def recorded_probe_hooks() -> Iterator[dict[str, list[Any]]]:
    """Record the engine's calls to the probe hooks (which are no-ops outside a probe) while in the block."""
    calls: dict[str, list[Any]] = {'derived': [], 'split_positions': [], 'bake_finished': []}
    saved = probe.report_derived_order, probe.report_split_positions, probe.bake_finished

    def derived(function_name: str, order: Sequence[int]) -> None:
        calls['derived'].append((function_name, list(order)))

    def split_positions(predicate: object, function_name: str, count: int) -> None:
        calls['split_positions'].append((function_name, count))

    def bake_finished() -> None:
        calls['bake_finished'].append(None)

    probe.report_derived_order, probe.report_split_positions, probe.bake_finished = derived, split_positions, bake_finished
    try:
        yield calls
    finally:
        probe.report_derived_order, probe.report_split_positions, probe.bake_finished = saved


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


def build_worked_example(name: str, early: Optional[EqnOrderingFn]) -> ThornFunction:
    """The docstring's worked example: calls 0: u = src, 1: v[li] = [...], 2: a = u + 1, ordered lexicographically."""
    gf = ThornDef("ARR", name)
    v = gf.decl("v", [li])
    u, a, src = gf.decl("u", []), gf.decl("a", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(u, src)
    fun.add_eqn(v[li], [src * 2, src * 3, src * 4])
    fun.add_eqn(a, u + 1)
    gf.bake(do_cse=False, ordering_fn=lexicographical_order, functions={"fn": {"early_ordering_fn": early}})
    return fun


def test_worked_example() -> None:
    with recorded_probe_hooks() as calls:
        fun = build_worked_example("WORKED", rank_by_post_population(lexicographical_order))
    assert calls['derived'] == [("fn", [0, 2, 1])], calls['derived']
    # The derived order drives the early locus exactly like add_eqn_order([0, 2, 1]).
    assert pre_population_origins(fun) == [[0, 2, 1]], pre_population_origins(fun)
    twin = build_worked_example("WORKEDTWIN", add_eqn_order([0, 2, 1]))
    assert eqns_of(fun) == eqns_of(twin)
    assert calls['bake_finished'] == [None]


def test_pull_out_temps_do_not_count() -> None:
    """
    Only the equations an add_eqn call produced count toward its median, as in the offline derivation (which
    snapshotted the list around each call). The pull-out temporary inherits its call's origin but does not count.

    Calls 0: qc = pull_out(src * src) * 2, 1: qb = src + 1, 2: qa = src * 3. Lexicographically the trial's order is
    [pull_out_0, qa, qb, qc] (pull_out_0 sorts first and qc reads it anyway). Counting qc alone, the medians are 3, 2
    and 1, so the order is [2, 1, 0]. Counting the pull-out temporary too, call 0's median would be (0 + 3) / 2 = 1.5,
    and the order [2, 0, 1].
    """
    gf = ThornDef("ARR", "PULLOUTCOUNT")
    qa, qb, qc, src = gf.decl("qa", []), gf.decl("qb", []), gf.decl("qc", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(qc, pull_out(src * src) * 2)
    fun.add_eqn(qb, src + 1)
    fun.add_eqn(qa, src * 3)
    with recorded_probe_hooks() as calls:
        gf.bake(do_cse=False, ordering_fn=lexicographical_order,
                functions={"fn": {"early_ordering_fn": rank_by_post_population(lexicographical_order)}})
    assert calls['derived'] == [("fn", [2, 1, 0])], calls['derived']
    el = fun.eqn_complex.eqn_lists[0]
    assert el.eqn_origin[sym("pull_out_0")] == 0, "the pull-out temporary should inherit its call's origin"


def build_isolation_thorn(name: str, early: Optional[EqnOrderingFn]) -> tuple[ThornDef, Recorder, Recorder]:
    """
    Two functions sharing subexpressions (so global CSE makes global temporaries, with synthetic functions, as well as
    local and tile ones), pull-outs in both, a manual soft split and an early hard auto split in the ranked function.
    """
    gf = ThornDef("ARR", name)
    v = gf.decl("v", [li])
    a, b, c, d, e, src, src2 = (gf.decl(n, []) for n in ("a", "b", "c", "d", "e", "src", "src2"))
    p = gf.add_param("p", 1.0, "a param")
    hard, soft = Recorder(lambda i: i == 1), Recorder(lambda i: False)
    f1 = gf.create_function("f1", ScheduleBin.Evolve, auto_hard_split_predicate=hard, auto_soft_split_predicate=soft)
    f1.add_eqn(a, sin(src) * cos(src) + pull_out(src * src))
    f1.add_eqn(v[li], [sin(src2) * p, cos(src2) * cos(src), sin(src) * cos(src2) * exp(src)])
    f1.soft_split()
    f1.add_eqn(b, exp(src2) * cos(src) + sin(src) * cos(src) * v[l0])
    f1.add_eqn(c, src * p + exp(src2) * cos(src) + a)
    f2 = gf.create_function("f2", ScheduleBin.Evolve)
    f2.add_eqn(d, sin(src) * cos(src2) * exp(src) + exp(src2) * cos(src) + pull_out(src2 * src2))
    f2.add_eqn(e, d * sin(src) * cos(src) + cos(src2) * cos(src))
    gf.bake(do_cse=True, temporary_promotion_strategy=promote_all(),
            functions={"f1": {"early_ordering_fn": early}})
    return gf, hard, soft


def thorn_state(gf: ThornDef) -> dict[str, Any]:
    """Everything a bake leaves behind that later stages or the generated code read."""
    functions: dict[str, Any] = dict()
    for name, fn in sorted(gf.functions.items()):
        ec = fn.eqn_complex
        functions[name] = {
            'eqns': eqns_of(fn),
            'hard_splits': sorted(ec._hard_splits),
            'loops': dict(fn.source_annotations.loops),
            'eqn_annotations': {i: {str(k): v for k, v in d.items()} for i, d in fn.source_annotations.eqns.items()},
            'tile_temps': sorted(map(str, ec.tile_temporaries)),
            'origins': [sorted((str(k), o) for k, o in el.eqn_origin.items()) for el in ec.eqn_lists],
            'recipe_orders': [[str(k) for k in el.eqn_recipe_order] for el in ec.eqn_lists],
            'pre_population_orders': [[str(k) for k in el.eqn_pre_population_order] for el in ec.eqn_lists],
            'ordering_fns': [el.ordering_fn for el in ec.eqn_lists],
        }
    return {
        'functions': functions,
        'tile_temporaries': sorted(map(str, gf.tile_temporaries)),
        'global_temporaries': sorted(map(str, gf.global_temporaries)),
        'cse_temp_kinds': sorted((str(k), str(v)) for k, v in gf.cse_temp_kinds.items()),
        'unique_name_counter': gf._unique_name_counter,
        'centering': sorted((k, str(v)) for k, v in gf.centering.items()),
        'declarations': sorted(gf.declarations),
        'synthetic_fns': sorted(str(fn.name) for fns in gf.synthetic_fns.values() for fn in fns),
    }


def test_trial_isolation() -> None:
    """The real bake is identical to one with add_eqn_order(<the derived order>), temporary names included, and the
    trial calls no auto split predicate and no probe hook."""
    with recorded_probe_hooks() as ranked_calls:
        ranked, ranked_hard, ranked_soft = build_isolation_thorn("ISORANKED", rank_by_post_population(prioritize_rare_symbols))
    [(name, derived)] = ranked_calls['derived']
    # Not the recipe order, so the ranking matters: the manual soft split keeps calls 0 and 1 in list 0 and calls 2 and
    # 3 in list 1, where prioritize_rare_symbols puts call 3 first.
    assert (name, derived) == ("f1", [0, 1, 3, 2]), (name, derived)
    assert pre_population_origins(ranked.functions["f1"]) == [[0, 1], [3, 2]], pre_population_origins(ranked.functions["f1"])

    with recorded_probe_hooks() as explicit_calls:
        explicit, explicit_hard, explicit_soft = build_isolation_thorn("ISOEXPLICIT", add_eqn_order(derived))
    assert explicit_calls['derived'] == []

    ranked_state, explicit_state = thorn_state(ranked), thorn_state(explicit)
    for key in explicit_state:
        assert ranked_state[key] == explicit_state[key], f"{key}:\n{ranked_state[key]}\n!=\n{explicit_state[key]}"
    # The test exercises what it should: global temporaries with synthetic functions, and pull-outs in both functions.
    assert len(explicit_state['global_temporaries']) > 0 and len(explicit_state['synthetic_fns']) > 0
    assert explicit_state['unique_name_counter'] == 2

    # The predicates were queried once per early position (the ranked function's two lists have 2 groups each), by
    # the real bake only; the probe saw the same split reports and one bake_finished.
    assert ranked_hard.calls == explicit_hard.calls == [0, 1, 2, 3], ranked_hard.calls
    assert ranked_soft.calls == explicit_soft.calls
    assert ranked_calls['split_positions'] == explicit_calls['split_positions'] == [("f1", 4), ("f1", 4)]
    assert ranked_calls['bake_finished'] == explicit_calls['bake_finished'] == [None]


def test_trial_restores_after_failure() -> None:
    """If the trial raises, the error propagates and every function is left as it was, unbaked."""
    gf = ThornDef("ARR", "TRIALFAIL")
    a, b, src = gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
    hard = Recorder(lambda i: False)
    f1 = gf.create_function("f1", ScheduleBin.Evolve, auto_hard_split_predicate=hard)
    f1.add_eqn(a, sin(src))
    f2 = gf.create_function("f2", ScheduleBin.Evolve)
    f2.add_eqn(b, cos(src) + pull_out(src * src))

    def exploding(eqns: dict[Symbol, Expr], eqn_list: EqnList) -> Iterator[Symbol]:
        raise RuntimeError("boom")

    complexes = {name: fn.eqn_complex for name, fn in gf.functions.items()}
    try:
        gf.bake(functions={"f1": {"early_ordering_fn": rank_by_post_population(exploding)}})
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("expected the trial to raise")
    for name, fn in gf.functions.items():
        assert fn.eqn_complex is complexes[name]
        assert not fn.been_baked and not fn.eqn_complex.been_baked
        assert all(not el.been_baked for el in fn.eqn_complex.eqn_lists)
    assert f1._auto_hard_split_predicate is hard and hard.calls == []
    assert gf._unique_name_counter == 0 and len(gf.tile_temporaries) == 0 and len(gf.cse_temp_kinds) == 0


def test_bake_wide() -> None:
    """Passed bake-wide, rank_by_post_population derives each function's own order, from one trial."""
    gf = ThornDef("ARR", "BAKEWIDE")
    names = ("qa", "qb", "qc", "ra", "rb")
    qa, qb, qc, ra, rb = (gf.decl(n, []) for n in names)
    src = gf.decl("src", [])
    f1 = gf.create_function("f1", ScheduleBin.Evolve)
    f1.add_eqn(qc, src * 2)
    f1.add_eqn(qa, src * 3)
    f1.add_eqn(qb, src * 4)
    f2 = gf.create_function("f2", ScheduleBin.Evolve)
    f2.add_eqn(rb, src * 5)
    f2.add_eqn(ra, src * 6)
    with recorded_probe_hooks() as calls:
        gf.bake(do_cse=False, ordering_fn=lexicographical_order,
                early_ordering_fn=rank_by_post_population(lexicographical_order))
    assert sorted(calls['derived']) == [("f1", [1, 2, 0]), ("f2", [1, 0])], calls['derived']
    assert pre_population_origins(f1) == [[1, 2, 0]] and pre_population_origins(f2) == [[1, 0]]


def test_misuse() -> None:
    expect_dsl_exception(lambda: rank_by_post_population(bayesian_optimization), "Bayesian")
    expect_dsl_exception(lambda: rank_by_post_population(functools.partial(bayesian_optimization, exploration_iter=1)),
                         "Bayesian")
    expect_dsl_exception(lambda: rank_by_post_population(rank_by_post_population(prioritize_rare_symbols)),
                         "cannot be nested")

    ranked = rank_by_post_population(prioritize_rare_symbols)
    expect_dsl_exception(lambda: ranked({}, None), "not an ordering function by itself")  # type: ignore[arg-type]

    def fresh(name: str) -> tuple[ThornDef, ThornFunction]:
        gf = ThornDef("ARR", name)
        a, src = gf.decl("a", []), gf.decl("src", [])
        fun = gf.create_function("fn", ScheduleBin.Evolve)
        fun.add_eqn(a, sin(src))
        return gf, fun

    gf, fun = fresh("MISUSE1")
    expect_dsl_exception(lambda: gf.bake(ordering_fn=ranked), "can only be the early_ordering_fn")
    gf, fun = fresh("MISUSE2")
    expect_dsl_exception(lambda: gf.bake(pre_population_ordering_fn=ranked), "can only be the early_ordering_fn")
    gf, fun = fresh("MISUSE3")
    expect_dsl_exception(lambda: fun._early_bake(early_ordering_fn=ranked), "cannot be passed to _early_bake directly")
    # Misuse next to a valid early rank_by_post_population fails before the trial bake: no order is derived.
    gf, fun = fresh("MISUSE4")
    with recorded_probe_hooks() as calls:
        expect_dsl_exception(lambda: gf.bake(early_ordering_fn=ranked, ordering_fn=ranked),
                             "can only be the early_ordering_fn")
    assert calls['derived'] == [], calls


def test_repr() -> None:
    ranked = rank_by_post_population(prioritize_rare_symbols)
    assert isinstance(ranked, RankByPostPopulation)
    expected = 'rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols)'
    assert repr(ranked) == expected, repr(ranked)
    assert describe_param(ranked) == expected
    partial_fn = functools.partial(prioritize_rare_symbols, complexity_factor=0.5)
    assert repr(rank_by_post_population(partial_fn)) == \
        'rank_by_post_population(partial(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols, complexity_factor=0.5))'
    assert ranked.ordering_fn is prioritize_rare_symbols


if __name__ == "__main__":
    tests = [
        test_median_rank_key,
        test_rank_add_eqn_calls,
        test_worked_example,
        test_pull_out_temps_do_not_count,
        test_trial_isolation,
        test_trial_restores_after_failure,
        test_bake_wide,
        test_misuse,
        test_repr,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print("All rank_by_post_population tests passed.")
