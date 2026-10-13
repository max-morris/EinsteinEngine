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
Tests for auto splits at the post-population locus (EinsteinEngine.intermediate.post_population_split).

Cut positions depend on the post-CSE order, so each test bakes an unsplit twin of its recipe first and derives the
positions from the twin's lists.
"""

import contextlib
import functools
import io
import os
import tempfile
from typing import Any, Callable, Iterator, Optional

from sympy import Expr, IndexedBase, Symbol

from EinsteinEngine import *
from EinsteinEngine.common.sympywrap import do_subs, free_symbols, sympify
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef, ThornFunction
from EinsteinEngine.intermediate.eqnlist import EqnList
from EinsteinEngine.intermediate.split_locus import SplitLocus
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy
from EinsteinEngine.tuning import probe

HardPredicate = Callable[[int], bool]
SoftPredicate = Callable[[int], bool | SoftSplitRetainmentStrategy]
Recipe = Callable[[ThornDef, Optional[HardPredicate], Optional[SoftPredicate]], None]

_thorn_counter = 0


def _thorn_name(tag: str) -> str:
    global _thorn_counter
    _thorn_counter += 1
    return f"TSTPP{_thorn_counter}{tag}"


def build(recipe: Recipe, tag: str, *,
          hard: Optional[HardPredicate] = None,
          soft: Optional[SoftPredicate] = None,
          **bake_opts: Any) -> ThornDef:
    gf = ThornDef("ARR", _thorn_name(tag))
    recipe(gf, hard, soft)
    gf.bake(**bake_opts)
    return gf


def mk_pp_function(gf: ThornDef, name: str, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> ThornFunction:
    return gf.create_function(name, ScheduleBin.Evolve, auto_split_locus=SplitLocus.PostPopulation,
                              auto_hard_split_predicate=hard, auto_soft_split_predicate=soft)


def decl(gf: ThornDef, names: str) -> Iterator[IndexedBase]:
    return (gf.decl(n, []) for n in names)


# a = (uv+1)^2, b = (uv+1)^3 + a, c = (uv+1) b: one Local CSE temp shared by every equation.
def recipe_shared(gf: ThornDef, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> None:
    a, b, c, u, v = decl(gf, "abcuv")
    fun = mk_pp_function(gf, "f", hard, soft)
    fun.add_eqn(a, (u * v + 1) ** 2)
    fun.add_eqn(b, (u * v + 1) ** 3 + a)
    fun.add_eqn(c, (u * v + 1) * b)


# A chain of Local CSE temps (some reading others) shared across b and c, plus a that shares only one of them.
def recipe_chain(gf: ThornDef, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> None:
    a, b, c, u, v = decl(gf, "abcuv")
    fun = mk_pp_function(gf, "f", hard, soft)
    fun.add_eqn(a, u ** 2 + sin(u))
    fun.add_eqn(b, sin(v + 2) ** 3 * cos(u * v + 1))
    fun.add_eqn(c, sin(v + 2) ** 2 * u * cos(u * v + 1) + b)


# a shares nothing with b and c, which share a CSE temp.
def recipe_disjoint(gf: ThornDef, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> None:
    a, b, c, u, v, w = decl(gf, "abcuvw")
    fun = mk_pp_function(gf, "f", hard, soft)
    fun.add_eqn(a, w ** 2 + 3)
    fun.add_eqn(b, sin(u * v + 1) ** 3)
    fun.add_eqn(c, sin(u * v + 1) ** 2 * b)


# recipe_shared with a manual soft split after a.
def recipe_manual_soft(gf: ThornDef, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> None:
    a, b, c, u, v = decl(gf, "abcuv")
    fun = mk_pp_function(gf, "f", hard, soft)
    fun.add_eqn(a, (u * v + 1) ** 2)
    fun.soft_split()
    fun.add_eqn(b, (u * v + 1) ** 3 + a)
    fun.add_eqn(c, (u * v + 1) * b)


# A pull-out temp read by two later equations.
def recipe_pull_out(gf: ThornDef, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> None:
    a, b, u, v = decl(gf, "abuv")
    fun = mk_pp_function(gf, "f", hard, soft)
    fun.add_eqn(a, pull_out(sin(u) * cos(v) + u) * v)
    fun.add_eqn(b, u * (sin(u) * cos(v) + u))


# Two functions sharing subexpressions, so that promote_all() creates Global temps and synthetic functions.
def recipe_two_functions(gf: ThornDef, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> None:
    a, b, c, d, u, v = decl(gf, "abcduv")
    f = mk_pp_function(gf, "f", hard, soft)
    f.add_eqn(a, sin(u * v + 1) ** 2 + cos(u + v))
    f.add_eqn(b, sin(u * v + 1) ** 3 * cos(u + v) + a)
    g = gf.create_function("g", ScheduleBin.Evolve)
    g.add_eqn(c, sin(u * v + 1) ** 2 * cos(u + v) ** 2)
    g.add_eqn(d, sin(u * v + 1) * v)


# The shape of an enforce function: X' = X/(X+Y), Y' = Y/(X+Y), plus an unrelated a. A cut between the two writers
#  would make the second one read the already overwritten X.
def mk_recipe_overwrite(locus: SplitLocus) -> Recipe:
    def recipe(gf: ThornDef, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> None:
        X, Y, a, u = decl(gf, "XYau")
        Xp, Yp = gf.overwrite(X), gf.overwrite(Y)
        fun = gf.create_function("f", ScheduleBin.Evolve, auto_split_locus=locus,
                                 auto_hard_split_predicate=hard, auto_soft_split_predicate=soft)
        fun.add_eqn(a, u * 2 + 1)
        fun.add_eqn(Xp, X / (X + Y))
        fun.add_eqn(Yp, Y / (X + Y))
    return recipe


# add_eqn calls that read later calls: call 0 reads b, which call 1 writes, and calls 2 and 3 depend on each other
#  (vD1 reads c, and c reads vD0). At the early locus without an early ordering function each call is its own position,
#  so a hard cut after call 0 or after call 2 would put a read before its write.
def mk_recipe_backward(locus: SplitLocus) -> Recipe:
    def recipe(gf: ThornDef, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> None:
        v = gf.decl("v", [li])
        a, b, c, u = decl(gf, "abcu")
        fun = gf.create_function("f", ScheduleBin.Evolve, auto_split_locus=locus,
                                 auto_hard_split_predicate=hard, auto_soft_split_predicate=soft)
        fun.add_eqn(a, b * 2 + sin(u))
        fun.add_eqn(b, u * 3 + cos(u))
        fun.add_eqn(v[li], [u * sin(u), c * 2, u * 3])
        fun.add_eqn(c, v[l0] * 3 + a)
    return recipe


def temps_first(eqns: dict[Symbol, Expr], eqn_list: EqnList) -> Iterator[Symbol]:
    """Synthetic symbols first, then the rest, each in recipe order (dependencies are repaired by the bake)."""
    order = list(recipe_order(eqns, eqn_list))
    yield from (lhs for lhs in order if lhs in eqn_list.synthetic_symbols)
    yield from (lhs for lhs in order if lhs not in eqn_list.synthetic_symbols)


def sym(name: str) -> Symbol:
    return mk_symbol(name)


def function_order(gf: ThornDef, name: str = "f") -> list[Symbol]:
    """The order positions run over: every list's order, concatenated."""
    return [lhs for el in gf.functions[name].eqn_complex.eqn_lists for lhs in el.order]


def local_temps(gf: ThornDef, el: EqnList) -> set[Symbol]:
    return {lhs for lhs in el.eqns if lhs in el.synthetic_symbols and lhs not in gf.tile_temporaries}


def check_consistent(gf: ThornDef, name: str = "f") -> None:
    """The per-list bookkeeping is sane, no dead copies exist, and _calc_tile_temps does not fire its assertion."""
    ec = gf.functions[name].eqn_complex
    tile_temps = ec.tile_temporaries  # Runs _calc_tile_temps.
    all_locals: set[Symbol] = set()
    for el in ec.eqn_lists:
        all_locals |= local_temps(gf, el)

    for idx, el in enumerate(ec.eqn_lists):
        written = set(el.eqns.keys())
        read: set[Symbol] = set()
        for rhs in el.eqns.values():
            read |= free_symbols(rhs)

        assert set(el.order) == written, f"list {idx}: order {el.order} does not match eqns {written}"
        assert local_temps(gf, el) <= read, f"list {idx}: dead copies {local_temps(gf, el) - read}"
        assert not (el.inputs & written), f"list {idx}: inputs {el.inputs & written} are written"
        assert el.outputs <= written, f"list {idx}: outputs {el.outputs - written} are never written"
        assert el.uninitialized_tile_temporaries <= written
        assert not (el.preinitialized_tile_temporaries & written)
        for s in read:
            assert s in written or s in el.params or s in el.inputs or s in el.preinitialized_tile_temporaries, \
                f"list {idx}: {s} is read but is neither written, a param, an input, nor a tile temp"
            if s in all_locals:
                assert s in written, f"list {idx}: Local CSE temp {s} is read without being computed in the list"
        for s in el.preinitialized_tile_temporaries:
            assert s in tile_temps
            assert any(s in other.uninitialized_tile_temporaries for other in ec.eqn_lists[:idx]), \
                f"list {idx}: tile temp {s} is not written in an earlier list"


def simulate(gf: ThornDef, name: str = "f", **inputs: float) -> dict[Symbol, Expr]:
    """
    Run a function's lists in order on scalar inputs. Values of a list's temporaries that are neither tile temps nor
    declared variables are discarded when the list ends, so reading such a value outside its list fails. Writing an
    overwrite X' also stores into X, as the generated code does, so a later read of X sees the new value.
    """
    ec = gf.functions[name].eqn_complex
    tile_temps = ec.tile_temporaries
    env: dict[Symbol, Expr] = {sym(k): sympify(v) for k, v in inputs.items()}
    for el in ec.eqn_lists:
        for lhs in el.order:
            env[lhs] = sympify(float(do_subs(el.eqns[lhs], env).evalf()))
            if "'" in str(lhs):
                env[sym(str(lhs).replace("'", ""))] = env[lhs]
        for t in el.temporaries - tile_temps:
            if str(t) not in gf.declarations:
                del env[t]
    return env


def assert_same_outputs(gf1: ThornDef, gf2: ThornDef, outputs: str, name: str = "f") -> None:
    inputs = dict(u=0.3, v=-1.7, w=2.1)
    r1 = simulate(gf1, name, **inputs)
    r2 = simulate(gf2, name, **inputs)
    for o in outputs:
        v1, v2 = float(r1[sym(o)]), float(r2[sym(o)])
        assert abs(v1 - v2) <= 1e-12 * max(1.0, abs(v2)), f"{o}: {v1} != {v2}"


def generate_carpetx(gf: ThornDef, base_dir: str) -> None:
    wizard = CppCarpetXWizard(gf, CppCarpetXGenerator(gf))
    wizard.base_dir = os.path.join(base_dir, gf.arrangement, gf.name)
    wizard.generate_thorn()


def test_hard_cut_through_chain() -> None:
    twin = build(recipe_chain, "Twin")
    order = function_order(twin)
    twin_el = twin.functions["f"].eqn_complex.eqn_lists[0]
    twin_locals = local_temps(twin, twin_el)
    assert len(twin_locals) >= 3, f"The recipe should produce a chain of Local CSE temps, got {twin_locals}"

    b = sym("b")
    cut = order.index(b)
    assert cut < len(order) - 1
    gf = build(recipe_chain, "Hard", hard=lambda i: i == cut)
    ec = gf.functions["f"].eqn_complex
    check_consistent(gf)

    assert len(ec.eqn_lists) == 2
    el0, el1 = ec.eqn_lists
    roots_before = {lhs for lhs in order[:cut + 1] if lhs not in twin_locals}
    roots_after = {lhs for lhs in order[cut + 1:] if lhs not in twin_locals}
    assert set(el0.eqns) - twin_locals == roots_before
    assert set(el1.eqns) - twin_locals == roots_after

    # The Local temps each piece needs are recomputed there; temps read on both sides are duplicated.
    duplicated = local_temps(gf, el0) & local_temps(gf, el1)
    assert len(duplicated) > 0, "Some Local temp should be needed on both sides of the cut"
    assert local_temps(gf, el0) | local_temps(gf, el1) <= twin_locals

    # b crosses the hard cut, so it becomes a tile temp.
    assert b in ec.tile_temporaries
    assert b in el0.uninitialized_tile_temporaries and b in el1.preinitialized_tile_temporaries

    # The piece keeps the order that was cut.
    assert [lhs for lhs in el0.order if lhs in order[:cut + 1]] == [lhs for lhs in order[:cut + 1] if lhs in el0.eqns]

    assert_same_outputs(gf, twin, "abc")

    with tempfile.TemporaryDirectory() as tmp:
        generate_carpetx(gf, tmp)
        src_dir = os.path.join(tmp, gf.arrangement, gf.name, "src")
        assert len(os.listdir(src_dir)) > 0


def test_temp_needed_only_after_cut_moves() -> None:
    twin = build(recipe_disjoint, "Twin", ordering_fn=temps_first)
    order = function_order(twin)
    twin_locals = local_temps(twin, twin.functions["f"].eqn_complex.eqn_lists[0])
    a = sym("a")
    cut = order.index(a)
    moved = {t for t in twin_locals if order.index(t) < cut}
    assert len(moved) > 0, f"temps_first should put a Local temp before a, got {order}"

    gf = build(recipe_disjoint, "Move", hard=lambda i: i == cut, ordering_fn=temps_first)
    ec = gf.functions["f"].eqn_complex
    check_consistent(gf)

    assert len(ec.eqn_lists) == 2
    el0, el1 = ec.eqn_lists
    assert set(el0.eqns) == {a}, f"Only a should remain before the cut, got {el0.order}"
    assert moved <= set(el1.eqns)
    assert len(ec.tile_temporaries) == 0

    assert_same_outputs(gf, twin, "abc")


def test_soft_cut_merged() -> None:
    x0 = sym("x0")
    a = sym("a")

    for retainment, predicate_strategy, expect_mangled in [
        (retain_none(), None, True),
        (retain_all(), None, False),
        (retain_none(), retain_all(), False),
    ]:
        twin = build(recipe_shared, "SoftTwin", soft_split_retainment_strategy=retainment)
        order = function_order(twin)
        cut = order.index(a)
        twin_el = twin.functions["f"].eqn_complex.eqn_lists[0]
        assert x0 in local_temps(twin, twin_el)

        soft_value: bool | SoftSplitRetainmentStrategy = predicate_strategy if predicate_strategy is not None else True

        def soft(i: int, cut: int = cut, value: bool | SoftSplitRetainmentStrategy = soft_value) -> bool | SoftSplitRetainmentStrategy:
            return value if i == cut else False

        gf = build(recipe_shared, "Soft", soft=soft, soft_split_retainment_strategy=retainment)
        ec = gf.functions["f"].eqn_complex
        check_consistent(gf)

        assert len(ec.eqn_lists) == 1, "The soft cut should be merged away"
        mangled = {lhs for lhs in ec.eqn_lists[0].eqns if "_ss" in str(lhs)}
        if expect_mangled:
            assert {str(m) for m in mangled} == {"x0_ss0"}, f"x0 should be forgotten and recomputed, got {mangled}"
        else:
            assert len(mangled) == 0, f"x0 should be retained, got {mangled}"
            assert set(ec.eqn_lists[0].eqns) == set(twin_el.eqns)

        assert_same_outputs(gf, twin, "abc")


def test_promotion_to_tile() -> None:
    x0 = sym("x0")
    a = sym("a")

    for strategy, expect_tile in [(promote_none(), False), (promote_all(), True), (promote_all(TempKind.Tile), True)]:
        twin = build(recipe_shared, "PromTwin", temporary_promotion_strategy=strategy)
        assert twin.cse_temp_kinds[x0] == TempKind.Local
        order = function_order(twin)
        cut = order.index(a)

        def hard(i: int, cut: int = cut) -> bool:
            return i == cut

        gf = build(recipe_shared, "Prom", hard=hard, temporary_promotion_strategy=strategy)
        ec = gf.functions["f"].eqn_complex
        check_consistent(gf)
        assert len(ec.eqn_lists) == 2
        el0, el1 = ec.eqn_lists

        assert x0 in el0.eqns
        if expect_tile:
            assert x0 not in el1.eqns, "A promoted temp is computed only in the first piece that needs it"
            assert x0 in gf.tile_temporaries and x0 in ec.tile_temporaries
            assert x0 in el0.uninitialized_tile_temporaries and x0 in el1.preinitialized_tile_temporaries
        else:
            assert x0 in el1.eqns, "Under promote_none a temp needed on both sides is duplicated"
            assert x0 not in gf.tile_temporaries and x0 not in ec.tile_temporaries

        assert_same_outputs(gf, twin, "abc")


def test_soft_cut_never_promotes() -> None:
    """A soft cut generates no tile temps: a Local temp needed across it is duplicated, and the merge decides."""
    x0 = sym("x0")
    a = sym("a")

    for retainment, expect_mangled in [(retain_none(), True), (retain_all(), False)]:
        twin = build(recipe_shared, "SoftPromTwin", temporary_promotion_strategy=promote_all(),
                     soft_split_retainment_strategy=retainment)
        order = function_order(twin)
        cut = order.index(a)
        def at_cut(i: int, cut: int = cut) -> bool:
            return i == cut

        gf = build(recipe_shared, "SoftProm", soft=at_cut, temporary_promotion_strategy=promote_all(),
                   soft_split_retainment_strategy=retainment)
        ec = gf.functions["f"].eqn_complex
        check_consistent(gf)
        assert len(ec.eqn_lists) == 1
        assert x0 not in gf.tile_temporaries and len(ec.tile_temporaries) == 0
        mangled = {str(lhs) for lhs in ec.eqn_lists[0].eqns if "_ss" in str(lhs)}
        assert mangled == ({"x0_ss0"} if expect_mangled else set()), mangled
        assert_same_outputs(gf, twin, "abc")

    # A temp needed on both sides of a hard cut and across a soft cut is promoted, because of the hard cut.
    b = sym("b")
    twin = build(recipe_shared, "MixPromTwin", temporary_promotion_strategy=promote_all())
    order = function_order(twin)
    cut_a, cut_b = order.index(a), order.index(b)
    gf = build(recipe_shared, "MixProm", hard=lambda i: i == cut_b, soft=lambda i: i == cut_a,
               temporary_promotion_strategy=promote_all())
    ec = gf.functions["f"].eqn_complex
    check_consistent(gf)
    assert len(ec.eqn_lists) == 2
    assert x0 in gf.tile_temporaries and x0 in ec.eqn_lists[0].uninitialized_tile_temporaries
    assert_same_outputs(gf, twin, "abc")


def test_existing_tile_temp_moves_to_reader() -> None:
    """A CSE tile temp is never left alone in a piece that does not read it; it moves to its first reader."""
    def recipe(gf: ThornDef, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> None:
        a, b, u, v = decl(gf, "abuv")
        fun = mk_pp_function(gf, "f", hard, soft)
        fun.add_eqn(a, (u * v + 1) ** 2)
        fun.soft_split()
        fun.add_eqn(b, (u * v + 1) ** 3 + a)

    x0 = sym("x0")
    opts: dict[str, Any] = dict(temporary_promotion_strategy=promote_all(),
                                ordering_fn=functools.partial(pre_population_order, exclude_synthetic_symbols=True))
    twin = build(recipe, "LoneTwin", **opts)
    assert function_order(twin)[0] == x0
    for splitmaxxing in (False, True):
        gf = build(recipe, "Lone", hard=lambda i: i == 0, splitmaxxing=splitmaxxing, **opts)
        check_consistent(gf)
        for el in gf.functions["f"].eqn_complex.eqn_lists:
            assert set(el.eqns) != {x0}, "The tile temp must not be computed alone"
        if not splitmaxxing:
            assert_same_outputs(gf, twin, "ab")


def test_cse_off() -> None:
    a = sym("a")
    twin = build(recipe_shared, "NoCseTwin", do_cse=False)
    order = function_order(twin)
    assert len(twin.cse_temp_kinds) == 0 and twin.cse_promotion_predicate is None
    cut = order.index(a)

    gf = build(recipe_shared, "NoCse", hard=lambda i: i == cut, do_cse=False)
    ec = gf.functions["f"].eqn_complex
    check_consistent(gf)
    assert len(ec.eqn_lists) == 2
    assert ec.eqn_lists[0].order == order[:cut + 1]
    assert ec.eqn_lists[1].order == order[cut + 1:]
    assert a in ec.tile_temporaries

    assert_same_outputs(gf, twin, "abc")


def test_synthetic_functions_skipped() -> None:
    reports: list[tuple[object, str, int]] = list()
    queried: list[int] = list()

    twin = build(recipe_two_functions, "SynTwin", temporary_promotion_strategy=promote_all())
    synthetic = set(twin.functions) - {"f", "g"}
    assert len(synthetic) > 0, "promote_all() should have created synthetic functions"
    count = len(function_order(twin))
    cut = 0 if count > 1 else -1

    def hard(i: int) -> bool:
        queried.append(i)
        return i == cut

    original_report = probe.report_split_positions

    def record(predicate: object, function_name: str, count: int) -> None:
        reports.append((predicate, function_name, count))

    probe.report_split_positions = record
    try:
        gf = build(recipe_two_functions, "Syn", hard=hard, temporary_promotion_strategy=promote_all())
    finally:
        probe.report_split_positions = original_report

    assert reports == [(hard, "f", count)], f"Only f should report its positions, got {reports}"
    assert queried == list(range(count)), f"The predicate should see positions 0..{count - 1}, got {queried}"
    assert set(gf.functions) - {"f", "g"} == synthetic
    for name in synthetic | {"g"}:
        assert len(gf.functions[name].eqn_complex.eqn_lists) == 1
        check_consistent(gf, name)
    check_consistent(gf)
    if cut >= 0:
        assert len(gf.functions["f"].eqn_complex.eqn_lists) == 2


def test_cut_at_list_end() -> None:
    # A hard cut after the final equation of a list turns the boundary to the next list into a hard one.
    twin = build(recipe_manual_soft, "EndTwin")
    twin_ec = twin.functions["f"].eqn_complex
    assert len(twin_ec.eqn_lists) == 1, "The manual soft split is merged away in the twin"

    # Record the list lengths the predicates see, which are those before the merge.
    seen: dict[str, ThornDef] = dict()
    lengths: list[int] = list()

    def recipe(gf: ThornDef, hard: Optional[HardPredicate], soft: Optional[SoftPredicate]) -> None:
        seen["gf"] = gf
        recipe_manual_soft(gf, hard, soft)

    def record_lengths(i: int) -> bool:
        if i == 0:
            lengths.extend(len(el.order) for el in seen["gf"].functions["f"].eqn_complex.eqn_lists)
        return False

    build(recipe, "EndLen", hard=record_lengths)
    assert len(lengths) == 2
    end_of_list_0 = lengths[0] - 1

    x0 = sym("x0")
    gf = build(recipe_manual_soft, "End", hard=lambda i: i == end_of_list_0)
    ec = gf.functions["f"].eqn_complex
    check_consistent(gf)
    assert len(ec.eqn_lists) == 2, "The soft boundary became hard, so nothing is merged"
    assert set(ec.eqn_lists[0].eqns) == {x0, sym("a")}
    assert x0 in ec.eqn_lists[1].eqns
    assert_same_outputs(gf, twin, "abc")

    # A cut after the very last position changes nothing.
    last = sum(lengths) - 1
    gf = build(recipe_manual_soft, "Last", hard=lambda i: i == last)
    ec = gf.functions["f"].eqn_complex
    check_consistent(gf)
    assert len(ec.eqn_lists) == 1
    assert ec.eqn_lists[0].eqns == twin_ec.eqn_lists[0].eqns


def test_pull_out_temp_crosses_hard_cut() -> None:
    twin = build(recipe_pull_out, "PullTwin")
    order = function_order(twin)
    pulled = [lhs for lhs in order if str(lhs).startswith("pull_out")]
    assert len(pulled) == 1, f"Expected one pull-out temp, got {order}"
    p = pulled[0]
    cut = order.index(p)

    gf = build(recipe_pull_out, "Pull", hard=lambda i: i == cut)
    ec = gf.functions["f"].eqn_complex
    check_consistent(gf)
    assert len(ec.eqn_lists) == 2
    el0, el1 = ec.eqn_lists
    assert p in el0.eqns and p not in el1.eqns, "Named symbols are never duplicated"
    assert p in ec.tile_temporaries and p in el1.preinitialized_tile_temporaries
    # The first loop writes no grid function, so the tile temp's centering must have been inferred.
    assert gf.get_centering_from_var_name(str(p)) is not None
    assert_same_outputs(gf, twin, "ab")

    with tempfile.TemporaryDirectory() as tmp:
        generate_carpetx(gf, tmp)


def test_overwrite_never_separated() -> None:
    """
    At every locus, a split requested between an overwrite X' and a later read of X is ignored (the predicates are
    still queried at every position), so no loop reads an already overwritten variable.
    """
    inputs = dict(X=0.3, Y=1.7, u=-0.4)
    for locus in (SplitLocus.Early, SplitLocus.PrePopulation, SplitLocus.PostPopulation):
        recipe = mk_recipe_overwrite(locus)
        twin = build(recipe, f"OW{locus.name}Twin")
        expected = simulate(twin, "f", **inputs)

        for kind in ("hard", "soft"):
            calls: list[int] = list()

            def everywhere(i: int) -> bool:
                calls.append(i)
                return True

            gf = build(recipe, f"OW{locus.name}{kind}", **{kind: everywhere})
            ec = gf.functions["f"].eqn_complex
            check_consistent(gf)
            assert calls == list(range(len(calls))) and len(calls) >= 2, calls

            [writers_list] = {idx for idx, el in enumerate(ec.eqn_lists) for lhs in el.eqns if str(lhs) in {"X'", "Y'"}}
            if locus is SplitLocus.Early and kind == "hard":
                # The cut between a and the writers is legitimate and still made.
                assert len(ec.eqn_lists) == 2 and writers_list == 1, [el.order for el in ec.eqn_lists]

            result = simulate(gf, "f", **inputs)
            for out in ("X'", "Y'", "a"):
                assert abs(float(result[sym(out)]) - float(expected[sym(out)])) <= 1e-12, (locus, kind, out)

            with tempfile.TemporaryDirectory() as tmp:
                generate_carpetx(gf, tmp)
                with open(os.path.join(tmp, gf.arrangement, gf.name, "src", f"{gf.name}_f.cpp")) as f:
                    loops = f.read().split("// f loop")[1:]
                stored: set[str] = set()
                for loop in loops:
                    for var in stored:
                        assert f"access({var}," not in loop, f"{locus.name}/{kind}: a loop reads {var} after it was stored"
                    stored |= {var for var in "XY" if f"store({var}," in loop}


def test_backward_reads_never_hard_cut() -> None:
    """
    A hard cut that would put a read before its write (add_eqn calls that read later calls, at the early locus without
    an early ordering function) is ignored with a warning, while the predicates are still queried at every position;
    soft cuts there are kept, since the merge rejoins the loops. The pre- and post-population loci cut baked orders,
    which respect dependencies, so they never refuse a cut for this reason.
    """
    inputs = dict(u=-0.4)
    outputs = ("a", "b", "c", "vD0", "vD1", "vD2")
    for locus in (SplitLocus.Early, SplitLocus.PrePopulation, SplitLocus.PostPopulation):
        recipe = mk_recipe_backward(locus)
        twin = build(recipe, f"BW{locus.name}Twin")
        expected = simulate(twin, "f", **inputs)

        for kind in ("hard", "soft"):
            calls: list[int] = list()

            def everywhere(i: int) -> bool:
                calls.append(i)
                return True

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                gf = build(recipe, f"BW{locus.name}{kind}", **{kind: everywhere})
            ec = gf.functions["f"].eqn_complex
            assert calls == list(range(len(calls))) and len(calls) >= 4, calls
            warnings = [line for line in out.getvalue().splitlines() if "ignoring the auto splits" in line]

            if locus is SplitLocus.Early and kind == "hard":
                # Positions 0 and 2 are refused; the cut after call 1 is legitimate, and the one after call 3 is at the end.
                assert len(warnings) == 1 and "positions [0, 2] of the Early locus" in warnings[0] \
                       and "put reads of b, c before their writes." in warnings[0], warnings
                named = [sorted(str(lhs) for lhs in el.eqns if str(lhs) in outputs) for el in ec.eqn_lists]
                assert named == [["a", "b"], ["c", "vD0", "vD1", "vD2"]], named
            else:
                assert len(warnings) == 0, (locus, kind, warnings)

            result = simulate(gf, "f", **inputs)
            for name in outputs:
                assert abs(float(result[sym(name)]) - float(expected[sym(name)])) <= 1e-12, (locus, kind, name)


def test_no_cut_is_identity() -> None:
    twin = build(recipe_chain, "IdTwin")
    gf = build(recipe_chain, "Id", hard=lambda i: False, soft=lambda i: False)
    twin_ec = twin.functions["f"].eqn_complex
    ec = gf.functions["f"].eqn_complex
    assert len(ec.eqn_lists) == len(twin_ec.eqn_lists)
    for el, twin_el in zip(ec.eqn_lists, twin_ec.eqn_lists):
        assert el.order == twin_el.order
        assert el.eqns == twin_el.eqns


if __name__ == "__main__":
    tests = [
        test_hard_cut_through_chain,
        test_temp_needed_only_after_cut_moves,
        test_soft_cut_merged,
        test_promotion_to_tile,
        test_soft_cut_never_promotes,
        test_existing_tile_temp_moves_to_reader,
        test_cse_off,
        test_synthetic_functions_skipped,
        test_cut_at_list_end,
        test_pull_out_temp_crosses_hard_cut,
        test_overwrite_never_separated,
        test_backward_reads_never_hard_cut,
        test_no_cut_is_identity,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print("All post-population split tests passed.")
