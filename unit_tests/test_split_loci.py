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
Tests for the early and pre-population ordering/split loci and for the EqnComplex refinement primitive.
Run as a plain script.
"""

import contextlib
import functools
import io
import os
import tempfile
from typing import Any, Callable, Iterator, Optional

from sympy import Symbol

from EinsteinEngine import *
from EinsteinEngine.common.sympywrap import *
from EinsteinEngine.frontend.definitions import *
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef, ThornFunction
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.intermediate.eqn_grouping import CutHazard, cut_hazards
from EinsteinEngine.intermediate.eqn_ordering import pre_cse_stand_in
from EinsteinEngine.intermediate.eqnlist import EqnListPiece, SplitBoundary
from EinsteinEngine.intermediate.intermediate_exception import IntermediateException
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy

x = mk_symbol("x")


def lists_of(fun: ThornFunction) -> list[list[str]]:
    """The LHSes of each of the function's lists, in baked order."""
    return [[str(lhs) for lhs in el.order] for el in fun.eqn_complex.eqn_lists]


def eqns_of(fun: ThornFunction) -> list[list[tuple[str, str]]]:
    return [[(str(lhs), str(el.eqns[lhs])) for lhs in el.order] for el in fun.eqn_complex.eqn_lists]


def keys(d: dict[Symbol, int]) -> list[str]:
    return [str(k) for k in d.keys()]


def expect_dsl_exception(fn: Callable[[], object], fragment: str) -> None:
    try:
        fn()
    except DslException as e:
        assert fragment in str(e), f"Expected '{fragment}' in the message, got: {e}"
        return
    raise AssertionError(f"Expected a DslException mentioning '{fragment}'")


class Recorder:
    """A split predicate that records the positions it is queried with."""

    def __init__(self, fire: Callable[[int], bool]) -> None:
        self.fire = fire
        self.calls: list[int] = list()

    def __call__(self, i: int) -> bool:
        self.calls.append(i)
        return self.fire(i)


def test_no_reorder_equivalence() -> None:
    """Deferred early-locus predicates give the same lists as splitting right after each add_eqn (the old behavior)."""
    def hard(i: int) -> bool:
        return i in (1, 4)

    def soft(i: int) -> bool | SoftSplitRetainmentStrategy:
        return retain_all() if i == 2 else (i == 5)

    def build(name: str, deferred: bool) -> ThornFunction:
        gf = ThornDef("ARR", name)
        v = gf.decl("v", [li])
        a, b, c, d, e, f, g, src = [gf.decl(n, []) for n in ("a", "b", "c", "d", "e", "f", "g", "src")]
        p = gf.add_param("p", 1.0, "a param")
        if deferred:
            fun = gf.create_function("fn", ScheduleBin.Evolve, auto_hard_split_predicate=hard, auto_soft_split_predicate=soft)
        else:
            fun = gf.create_function("fn", ScheduleBin.Evolve)

        adds: list[Callable[[], None]] = [
            lambda: fun.add_eqn(a, src * p + sin(src)),
            lambda: fun.add_eqn(v[li], [a * 2, a * src, sin(a) * cos(src)]),
            lambda: fun.add_eqn(b, v[l0] * v[l1] + sin(src)),
            lambda: fun.add_eqn(c, b * p + sin(src) * cos(src)),
            lambda: fun.add_eqn(d, c + v[l2] * a),
            lambda: fun.add_eqn(e, d * src + sin(src) * cos(src)),
            lambda: fun.add_eqn(f, e * p),
            lambda: fun.add_eqn(g, f + e + sin(src)),
        ]
        for i, add in enumerate(adds):
            add()
            if i == 3:
                fun.split_loop()  # A manual split that no auto split coincides with
            if not deferred:
                # The old behavior: predicates evaluated in add_eqn, right after the call.
                if hard(i):
                    fun.split_loop()
                elif isinstance(s := soft(i), bool):
                    if s:
                        fun.soft_split()
                else:
                    fun.soft_split(s)

        gf.bake(do_cse=True, temporary_promotion_strategy=promote_all(), soft_split_retainment_strategy=retain_none())
        return fun

    old, new = build("NOREORDEROLD", False), build("NOREORDERNEW", True)
    assert eqns_of(old) == eqns_of(new), f"{eqns_of(old)}\n!=\n{eqns_of(new)}"
    assert old.eqn_complex._hard_splits == new.eqn_complex._hard_splits
    assert dict(old.source_annotations.loops) == dict(new.source_annotations.loops)
    assert old.eqn_complex.tile_temporaries == new.eqn_complex.tile_temporaries


def test_early_add_eqn_order() -> None:
    gf = ThornDef("ARR", "EARLYORDER")
    a, b, c, d, src = [gf.decl(n, []) for n in ("a", "b", "c", "d", "src")]
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(a, src)
    fun.add_eqn(b, src * 2)
    fun.add_eqn(c, a + b)
    fun.add_eqn(d, src * 3)
    fun._early_bake(early_ordering_fn=add_eqn_order([3, 1, 0, 2]), ordering_fn=pre_population_order)
    el = fun.eqn_complex.eqn_lists[0]
    assert keys(el.eqn_pre_population_order) == ["d", "b", "a", "c"], keys(el.eqn_pre_population_order)
    assert keys(el.eqn_recipe_order) == ["a", "b", "c", "d"]
    assert list(el.eqn_origin.values()) == [0, 1, 2, 3]
    assert lists_of(fun) == [["d", "b", "a", "c"]]


def test_early_key_order_dependency_repair() -> None:
    """Reversing the calls cannot put c before the a and b it reads."""
    gf = ThornDef("ARR", "EARLYKEY")
    a, b, c, d, src = [gf.decl(n, []) for n in ("a", "b", "c", "d", "src")]
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(a, src)
    fun.add_eqn(b, src * 2)
    fun.add_eqn(c, a + b)
    fun.add_eqn(d, src * 3)
    fun._early_bake(early_ordering_fn=add_eqn_key_order(lambda i: -i), ordering_fn=pre_population_order)
    assert lists_of(fun) == [["d", "b", "a", "c"]], lists_of(fun)


def test_scc_super_groups() -> None:
    """v's group reads b, and b's group reads v0: the two groups form one super-group, i.e., one early position."""
    gf = ThornDef("ARR", "EARLYSCC")
    v = gf.decl("v", [li])
    b, w, src = gf.decl("b", []), gf.decl("w", []), gf.decl("src", [])
    hard = Recorder(lambda i: i == 0)
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_hard_split_predicate=hard)
    fun.add_eqn(v[li], [src, b * 2, src * 3])
    fun.add_eqn(b, v[l0] * 3)
    fun.add_eqn(w, src * 5)
    fun._early_bake(early_ordering_fn=add_eqn_order([2, 1, 0]), ordering_fn=pre_population_order)
    assert hard.calls == [0, 1], hard.calls
    assert lists_of(fun) == [["w"], ["vD0", "b", "vD1", "vD2"]], lists_of(fun)


def test_overwrite_anti_edge() -> None:
    """A group reading X stays before a group writing X', although no free symbol connects them."""
    gf = ThornDef("ARR", "EARLYOVERWRITE")
    X, r, src = gf.decl("X", []), gf.decl("r", []), gf.decl("src", [])
    Xp = gf.overwrite(X)
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(r, X * 2)
    fun.add_eqn(Xp, src)
    fun._early_bake(early_ordering_fn=add_eqn_order([1, 0]), ordering_fn=pre_population_order)
    assert keys(fun.eqn_complex.eqn_lists[0].eqn_pre_population_order) == ["r", "X'"]


def test_manual_splits_never_crossed() -> None:
    gf = ThornDef("ARR", "EARLYMANUAL")
    a, b, c, d, src = [gf.decl(n, []) for n in ("a", "b", "c", "d", "src")]
    hard = Recorder(lambda i: i == 2)
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_hard_split_predicate=hard)
    fun.add_eqn(a, src)
    fun.add_eqn(b, src * 2)
    fun.split_loop("custom")
    fun.add_eqn(c, src * 3)
    fun.add_eqn(d, src * 4)
    fun._early_bake(early_ordering_fn=add_eqn_key_order(lambda i: -i), ordering_fn=pre_population_order)
    # Each list is reordered on its own; positions run across lists in the new order: [b a] [d | c].
    assert hard.calls == [0, 1, 2, 3], hard.calls
    assert lists_of(fun) == [["b", "a"], ["d"], ["c"]], lists_of(fun)
    assert fun.source_annotations.loops[1] == "custom"
    assert fun.source_annotations.loops[2] == "fn loop 2"
    assert fun.eqn_complex._hard_splits == {1, 2}


def test_trailing_split_dropped() -> None:
    gf = ThornDef("ARR", "EARLYTRAILING")
    a, b, src = gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_hard_split_predicate=lambda i: i == 1)
    fun.add_eqn(a, src)
    fun.add_eqn(b, a * 2)
    fun._early_bake(ordering_fn=recipe_order)
    assert lists_of(fun) == [["a", "b"]], lists_of(fun)


def test_boundary_folding() -> None:
    # An auto hard split right before a manual soft split: the empty piece's hard boundary wins its kind, and the
    # manual split's custom annotation survives.
    gf = ThornDef("ARR", "FOLDHARD")
    a, b, c, src = gf.decl("a", []), gf.decl("b", []), gf.decl("c", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_hard_split_predicate=lambda i: i == 1)
    fun.add_eqn(a, src)
    fun.add_eqn(b, src * 2)
    fun.soft_split(annotation="manual soft")
    fun.add_eqn(c, src * 3)
    fun._early_bake(ordering_fn=recipe_order)
    assert lists_of(fun) == [["a", "b"], ["c"]], lists_of(fun)
    assert fun.eqn_complex._hard_splits == {1}
    assert fun.source_annotations.loops[1] == "manual soft"

    # An auto soft split right before a manual hard split: the manual (inherited) boundary wins with its annotation.
    gf = ThornDef("ARR", "FOLDSOFT")
    a, b, c, src = gf.decl("a", []), gf.decl("b", []), gf.decl("c", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_soft_split_predicate=lambda i: i == 1)
    fun.add_eqn(a, src)
    fun.add_eqn(b, src * 2)
    fun.split_loop("manual hard")
    fun.add_eqn(c, src * 3)
    fun._early_bake(ordering_fn=recipe_order)
    assert lists_of(fun) == [["a", "b"], ["c"]], lists_of(fun)
    assert fun.eqn_complex._hard_splits == {1}
    assert fun.source_annotations.loops[1] == "manual hard"

    # Direct refine: an empty piece is dropped and its boundary folds into the next piece's.
    def mk_direct(name: str) -> tuple[ThornFunction, Symbol, Symbol]:
        gf = ThornDef("ARR", name)
        a, b, src = gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
        fun = gf.create_function("fn", ScheduleBin.Evolve)
        fun.add_eqn(a, src)
        fun.add_eqn(b, src * 2)
        [sa, sb] = list(fun.eqn_complex.eqn_lists[0].eqns.keys())
        return fun, sa, sb

    # Between two soft boundaries, the later boundary is kept.
    fun, sa, sb = mk_direct("FOLDDIRECTDROP")
    forget, keep = retain_none(), retain_all()
    fun._refine([[EqnListPiece((sa,), None),
                  EqnListPiece((), SplitBoundary(soft=True, retainment_strategy=forget)),
                  EqnListPiece((sb,), SplitBoundary(soft=True, retainment_strategy=keep, annotation="later"))]])
    ec = fun.eqn_complex
    assert [len(el.eqns) for el in ec.eqn_lists] == [1, 1] and ec._hard_splits == set()
    assert ec._soft_split_retainment_strategies == {1: keep}
    assert fun.source_annotations.loops[1] == "later"

    # A hard boundary on either side: the empty piece is dropped, and the hard boundary wins.
    for first, second in ((SplitBoundary(soft=False, annotation="hard"), SplitBoundary(soft=True, annotation="soft")),
                          (SplitBoundary(soft=True, annotation="soft"), SplitBoundary(soft=False, annotation="hard"))):
        fun, sa, sb = mk_direct("FOLDDIRECTHARD")
        fun._refine([[EqnListPiece((sa,), None), EqnListPiece((), first), EqnListPiece((sb,), second)]])
        ec = fun.eqn_complex
        assert [len(el.eqns) for el in ec.eqn_lists] == [1, 1] and ec._hard_splits == {1}
        assert fun.source_annotations.loops[1] == "hard"

    # An empty trailing piece is dropped even when soft.
    fun, sa, sb = mk_direct("FOLDDIRECTTRAIL")
    fun._refine([[EqnListPiece((sa, sb), None), EqnListPiece((), SplitBoundary(soft=True))]])
    assert [len(el.eqns) for el in fun.eqn_complex.eqn_lists] == [2]


def test_leading_custom_annotation_kept() -> None:
    """
    A split_loop() before the first add_eqn leaves an empty first list. When a cut drops it, the list that becomes the
    first one loses its boundary but keeps its custom annotation.
    """
    for locus in (SplitLocus.Early, SplitLocus.PrePopulation):
        gf = ThornDef("ARR", f"LEADANNOT{locus.name.upper()}")
        a, b, c, src = gf.decl("a", []), gf.decl("b", []), gf.decl("c", []), gf.decl("src", [])
        fun = gf.create_function("fn", ScheduleBin.Evolve, auto_split_locus=locus,
                                 auto_hard_split_predicate=lambda i: i == 0)
        fun.split_loop(annotation="Custom")
        fun.add_eqn(a, src)
        fun.add_eqn(b, src * 2)
        fun.add_eqn(c, src * 3)
        fun._early_bake(ordering_fn=recipe_order)
        assert lists_of(fun) == [["a"], ["b", "c"]], lists_of(fun)
        assert fun.eqn_complex._hard_splits == {1}
        assert dict(fun.source_annotations.loops) == {0: "Custom", 1: "fn loop 1"}, dict(fun.source_annotations.loops)

    # Direct refine: the annotation folded in from dropped pieces survives on the new first list, too.
    gf = ThornDef("ARR", "LEADANNOTDIRECT")
    a, b, src = gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.split_loop(annotation="Custom")
    fun.add_eqn(a, src)
    fun.add_eqn(b, src * 2)
    [sa, sb] = list(fun.eqn_complex.eqn_lists[1].eqns.keys())
    fun._refine([[EqnListPiece((), None)],
                 [EqnListPiece((), None), EqnListPiece((sa,), SplitBoundary(soft=False)),
                  EqnListPiece((sb,), SplitBoundary(soft=False))]])
    assert [len(el.eqns) for el in fun.eqn_complex.eqn_lists] == [1, 1]
    assert dict(fun.source_annotations.loops) == {0: "Custom", 1: "fn loop 1"}, dict(fun.source_annotations.loops)


def test_empty_soft_piece_dropped_before_manual_soft_split() -> None:
    """
    An auto soft split after the last group before a manual soft_split() would leave an empty list; at every locus it
    is dropped and the manual split's boundary is kept.
    """
    def build(name: str, locus: SplitLocus) -> ThornFunction:
        gf = ThornDef("ARR", name)
        a, b, c, src = gf.decl("a", []), gf.decl("b", []), gf.decl("c", []), gf.decl("src", [])
        fun = gf.create_function("fn", ScheduleBin.Evolve, auto_split_locus=locus,
                                 auto_soft_split_predicate=lambda i: retain_none() if i == 1 else False)
        fun.add_eqn(a, sin(src) * cos(src) + src)
        fun.add_eqn(b, sin(src) * cos(src) * 2)
        fun.soft_split(retain_all())
        fun.add_eqn(c, sin(src) * cos(src) * 3)
        fun._early_bake(ordering_fn=recipe_order)
        return fun

    for locus in (SplitLocus.Early, SplitLocus.PrePopulation):
        fun = build(f"EMPTYSOFT{locus.name.upper()}", locus)
        ec = fun.eqn_complex
        assert ec._hard_splits == set()
        assert lists_of(fun) == [["a", "b"], ["c"]], lists_of(fun)
        assert set(ec._soft_split_retainment_strategies) == {1}


def test_merge_keeps_provenance() -> None:
    """A soft-split merge keeps the recipe positions, pre-population order, origins, and params of absorbed equations."""
    gf = ThornDef("ARR", "MERGEPROVENANCE")
    a, b, c, d, src = [gf.decl(n, []) for n in ("a", "b", "c", "d", "src")]
    p = gf.add_param("p", 1.0, "p")
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(a, sin(src))
    fun.soft_split()
    fun.add_eqn(b, a * cos(src))
    fun.add_eqn(c, src * p)
    fun.add_eqn(d, b + c)
    gf.bake(do_cse=True, soft_split_retainment_strategy=retain_none(),
            early_ordering_fn=add_eqn_order([0, 2, 1, 3]), ordering_fn=pre_population_order)
    [el] = fun.eqn_complex.eqn_lists
    names = {"a", "b", "c", "d"}
    named = {str(lhs): pos for lhs, pos in el.eqn_recipe_order.items() if str(lhs) in names}
    assert named == {"a": 0, "b": 1, "c": 2, "d": 3}, named
    positions = list(el.eqn_recipe_order.values())
    assert positions == sorted(positions), "Recipe order keys must stay in recipe-position order"
    assert [k for k in keys(el.eqn_pre_population_order) if k in names] == ["a", "c", "b", "d"]
    assert {str(k): v for k, v in el.eqn_origin.items() if str(k) in names} == {"a": 0, "b": 1, "c": 2, "d": 3}
    assert el.eqn_recorded_params[mk_symbol("c")] == frozenset({p})
    assert [lhs for lhs in lists_of(fun)[0] if lhs in names] == ["a", "c", "b", "d"]


def test_early_cuts_see_new_order() -> None:
    gf = ThornDef("ARR", "EARLYCUTORDER")
    a, b, c, src = gf.decl("a", []), gf.decl("b", []), gf.decl("c", []), gf.decl("src", [])
    soft = Recorder(lambda i: i == 0)
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_soft_split_predicate=soft)
    fun.add_eqn(a, src)
    fun.add_eqn(b, src * 2)
    fun.add_eqn(c, src * 3)
    fun._early_bake(early_ordering_fn=add_eqn_order([2, 0, 1]), ordering_fn=pre_population_order)
    assert soft.calls == [0, 1, 2]
    assert lists_of(fun) == [["c"], ["a", "b"]], lists_of(fun)
    assert fun.eqn_complex._hard_splits == set()
    assert fun.source_annotations.loops[1] == "fn loop 1 (soft split)"


def test_soft_queried_only_if_hard_declines() -> None:
    gf = ThornDef("ARR", "HARDFIRST")
    a, b, c, src = gf.decl("a", []), gf.decl("b", []), gf.decl("c", []), gf.decl("src", [])
    hard = Recorder(lambda i: i == 0)
    soft = Recorder(lambda i: True)
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_hard_split_predicate=hard, auto_soft_split_predicate=soft)
    fun.add_eqn(a, src)
    fun.add_eqn(b, src * 2)
    fun.add_eqn(c, src * 3)
    fun._early_bake(ordering_fn=recipe_order)
    assert hard.calls == [0, 1, 2] and soft.calls == [1, 2], (hard.calls, soft.calls)
    assert lists_of(fun) == [["a"], ["b"], ["c"]]
    assert fun.eqn_complex._hard_splits == {1}


def test_pre_population_ordering() -> None:
    gf = ThornDef("ARR", "PREPOPORDER")
    b, a, c, src = gf.decl("b", []), gf.decl("a", []), gf.decl("c", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(c, src * 3)
    fun.add_eqn(b, src * 2)
    fun.add_eqn(a, src)
    fun._early_bake(pre_population_ordering_fn=lexicographical_order, ordering_fn=recipe_order)
    el = fun.eqn_complex.eqn_lists[0]
    assert lists_of(fun) == [["a", "b", "c"]], lists_of(fun)
    assert keys(el.eqn_pre_population_order) == ["a", "b", "c"]
    assert keys(el.eqn_recipe_order) == ["c", "b", "a"]
    # ordering_fn is reset to the post-population function after the pre-CSE bake.
    assert el.ordering_fn is recipe_order


def test_pre_population_cuts() -> None:
    def build(name: str, hard: Optional[Callable[[int], bool]], bake: bool = False) -> tuple[ThornDef, ThornFunction]:
        gf = ThornDef("ARR", name)
        b, c, d, src = gf.decl("b", []), gf.decl("c", []), gf.decl("d", []), gf.decl("src", [])
        fun = gf.create_function("fn", ScheduleBin.Evolve, auto_split_locus=SplitLocus.PrePopulation,
                                 auto_hard_split_predicate=hard)
        fun.add_eqn(b, pull_out(src * src) + 1)
        fun.add_eqn(c, b * 2)
        fun.add_eqn(d, c + src)
        if bake:
            gf.bake(ordering_fn=recipe_order)
        else:
            fun._early_bake(ordering_fn=recipe_order)
        return gf, fun

    _, unsplit = build("PREPOPUNSPLIT", None)
    [order] = lists_of(unsplit)
    assert len(order) == 4 and order[0].startswith("pull_out"), order

    hard = Recorder(lambda i: i in (0, 2))
    gf, fun = build("PREPOPSPLIT", hard)
    assert hard.calls == [0, 1, 2, 3], hard.calls
    assert lists_of(fun) == [order[0:1], order[1:3], order[3:4]], lists_of(fun)
    assert all(el.been_baked for el in fun.eqn_complex.eqn_lists)
    assert all(el.ordering_fn is recipe_order for el in fun.eqn_complex.eqn_lists)
    # The pull-out temp crosses the first cut; it inherits its origin and gets a centering for its tile temp.
    first_el = fun.eqn_complex.eqn_lists[0]
    [pull_out_temp] = list(first_el.eqns.keys())
    assert first_el.eqn_origin[pull_out_temp] == 0
    assert gf.centering[str(pull_out_temp)] == Centering.VVV

    # Through a full bake, the pull-out temp is promoted to a tile temp and the thorn generates.
    gf, fun = build("PREPOPSPLITGEN", lambda i: i in (0, 2), bake=True)
    ec = fun.eqn_complex
    [pull_out_temp] = [lhs for lhs in ec.eqn_lists[0].eqns if str(lhs).startswith("pull_out")]
    assert pull_out_temp in ec.tile_temporaries
    assert pull_out_temp in ec.eqn_lists[0].uninitialized_tile_temporaries
    with tempfile.TemporaryDirectory() as tmp:
        wizard = CppCarpetXWizard(gf, CppCarpetXGenerator(gf))
        wizard.base_dir = os.path.join(tmp, gf.arrangement, gf.name)
        wizard.generate_thorn()
        with open(os.path.join(wizard.base_dir, "src", f"{gf.name}_fn.cpp")) as f:
            code = f.read()
        assert str(pull_out_temp) in code


def test_params_repartitioning() -> None:
    gf = ThornDef("ARR", "PREPOPPARAMS")
    a, b, src = gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
    p = gf.add_param("p", 1.0, "p")
    q = gf.add_param("q", 2.0, "q")
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_split_locus=SplitLocus.PrePopulation,
                             auto_hard_split_predicate=lambda i: i == 0)
    fun.add_eqn(a, src * p)
    fun.add_eqn(b, src * q)
    fun._early_bake(ordering_fn=recipe_order)
    el0, el1 = fun.eqn_complex.eqn_lists
    assert p in el0.params and q not in el0.params
    assert q in el1.params and p not in el1.params
    assert all(d in el0.params and d in el1.params for d in (DXI, DYI, DZI))


def test_validation_errors() -> None:
    def build(name: str) -> tuple[ThornFunction, Symbol, Symbol, Symbol]:
        gf = ThornDef("ARR", name)
        t, u = gf.decl("tmp", []), gf.decl("u", [])
        p = gf.add_param("p", 1.0, "p")
        fun = gf.create_function("fn", ScheduleBin.Evolve)
        fun.add_eqn(t, sin(x) * p)
        fun.add_eqn(u, t * 2)
        fun._early_bake(ordering_fn=recipe_order)
        [st, su] = list(fun.eqn_complex.eqn_lists[0].eqns.keys())
        return fun, st, su, p

    def refine(fun: ThornFunction, st: Symbol, su: Symbol) -> None:
        fun._refine([[EqnListPiece((st,), None), EqnListPiece((su,), SplitBoundary(soft=False))]])

    # The analytic seed keeps u's write region (Everywhere, since t is analytic) after t moves to an earlier loop.
    fun, st, su, p = build("VALIDOK")
    assert fun.eqn_complex.eqn_lists[0].write_decls[su] is IntentRegion.Everywhere
    refine(fun, st, su)
    fun.eqn_complex.rebake_refined()
    assert fun.eqn_complex.eqn_lists[1].write_decls[su] is IntentRegion.Everywhere

    # Without the seed, the write region would change.
    fun, st, su, p = build("VALIDREGION")
    refine(fun, st, su)
    fun.eqn_complex.eqn_lists[1].analytic_seed.clear()
    expect_dsl_exception(lambda: fun.eqn_complex.rebake_refined(), "inferred write region")

    # A param that goes missing from a piece can only be a bug in refine().
    fun, st, su, p = build("VALIDPARAM")
    refine(fun, st, su)
    fun.eqn_complex.eqn_lists[0].params.discard(p)
    try:
        fun.eqn_complex.rebake_refined()
    except AssertionError as e:
        assert "parameters" in str(e), str(e)
    else:
        raise AssertionError("Expected an AssertionError about the missing parameter")

    # A temporary written in a loop over the interior but read in a loop over everything.
    gf = ThornDef("ARR", "VALIDTILE")
    t, a, b, src = gf.decl("tmp", []), gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(t, sin(x))
    fun.add_eqn(a, src * 2)
    fun.add_eqn(b, t * 2)
    fun._early_bake(ordering_fn=recipe_order)
    [st, sa, sb] = list(fun.eqn_complex.eqn_lists[0].eqns.keys())
    fun._refine([[EqnListPiece((st, sa), None), EqnListPiece((sb,), SplitBoundary(soft=False))]])
    expect_dsl_exception(lambda: fun.eqn_complex.rebake_refined(), "only computes over Interior")


def test_overwrite_validation() -> None:
    """A refinement that makes a loop read X after an earlier loop cut from the same list wrote X' is rejected."""
    gf = ThornDef("ARR", "VALIDOVERWRITE")
    X, Y = gf.decl("X", []), gf.decl("Y", [])
    Xp, Yp = gf.overwrite(X), gf.overwrite(Y)
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(Xp, X / (X + Y))
    fun.add_eqn(Yp, Y / (X + Y))
    fun._early_bake(ordering_fn=recipe_order)
    [sxp, syp] = list(fun.eqn_complex.eqn_lists[0].eqns.keys())
    fun._refine([[EqnListPiece((sxp,), None), EqnListPiece((syp,), SplitBoundary(soft=False))]])
    expect_dsl_exception(lambda: fun.eqn_complex.rebake_refined(), "has already overwritten")

    # The refinement is consumed even when validation fails, so a second rebake fails loudly.
    try:
        fun.eqn_complex.rebake_refined()
    except IntermediateException as e:
        assert "without a preceding refine" in str(e), str(e)
    else:
        raise AssertionError("Expected an IntermediateException for a second rebake_refined()")


def test_cut_hazards() -> None:
    """Backward reads forbid hard cuts only, over every position between the reader and the writer; overwrites forbid
    every cut between the overwrite and a later read of an earlier version."""
    a, b, c, X, Xp = [mk_symbol(n) for n in ("a", "b", "c", "X", "X'")]
    # Elements: 0 writes a and reads c (written by element 2); 1 writes X'; 2 writes c; 3 writes b and reads X.
    hazards = cut_hazards([{c}, set(), set(), {X}], [{a}, {Xp}, {c}, {b}])
    overwrite_hazard = CutHazard(X, overwrite=Xp)
    assert hazards == {0: {CutHazard(c)}, 1: {CutHazard(c), overwrite_hazard}, 2: {overwrite_hazard}}, hazards
    assert CutHazard(c).forbids(soft=False) and not CutHazard(c).forbids(soft=True)
    assert overwrite_hazard.forbids(soft=False) and overwrite_hazard.forbids(soft=True)
    # A dependency-respecting order has no backward reads.
    assert cut_hazards([set(), {a}, {a, b}], [{a}, {b}, {c}]) == dict()


def test_early_backward_reads() -> None:
    """
    At the early locus without an early ordering function, the add_eqn calls are cut in recipe order, which need not
    respect their dependencies: a hard cut that would put a read before its write is ignored with a warning (the
    predicates are still queried everywhere), a soft cut is kept, and an early ordering function makes the calls one
    super-group, so the question never arises.
    """
    def build(name: str, early_ordering_fn: Optional[EqnOrderingFn] = None, **kwargs: Any) -> tuple[ThornFunction, str]:
        gf = ThornDef("ARR", name)
        v = gf.decl("v", [li])
        a, a2, b, src = gf.decl("a", []), gf.decl("a2", []), gf.decl("b", []), gf.decl("src", [])
        fun = gf.create_function("fn", ScheduleBin.Evolve, **kwargs)
        fun.add_eqn(a, b * 2)  # reads b, which the next call writes
        fun.add_eqn(b, src * 3)
        fun.add_eqn(v[li], [src, a2 * 2, src * 3])  # reads a2, which the next call writes from vD0
        fun.add_eqn(a2, v[l0] * 3 + a)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fun._early_bake(early_ordering_fn=early_ordering_fn, ordering_fn=pre_population_order)
        return fun, out.getvalue()

    hard = Recorder(lambda i: True)
    fun, out = build("EARLYBACKHARD", auto_hard_split_predicate=hard)
    assert hard.calls == [0, 1, 2, 3], hard.calls
    # Positions 0 and 2 are refused, 1 is cut, and the cut at 3, after the last call, leaves an empty piece.
    assert lists_of(fun) == [["b", "a"], ["vD0", "a2", "vD1", "vD2"]], lists_of(fun)
    assert "fn: ignoring the auto splits at positions [0, 2] of the Early locus, which would put reads of a2, b before " \
           "their writes." in out, out

    soft = Recorder(lambda i: i == 0)
    fun, out = build("EARLYBACKSOFT", auto_soft_split_predicate=soft)
    assert soft.calls == [0, 1, 2, 3], soft.calls
    assert lists_of(fun) == [["a"], ["b", "vD0", "a2", "vD1", "vD2"]], lists_of(fun)
    assert fun.eqn_complex.needs_merge() and "ignoring" not in out, out

    # With an early ordering function, the mutually dependent calls 2 and 3 form one super-group (one position), and
    # b's call is ordered before a's.
    hard = Recorder(lambda i: True)
    fun, out = build("EARLYBACKORDERED", add_eqn_order([0, 1, 2, 3]), auto_hard_split_predicate=hard)
    assert hard.calls == [0, 1, 2], hard.calls
    assert lists_of(fun) == [["b"], ["a"], ["vD0", "a2", "vD1", "vD2"]], lists_of(fun)
    assert "ignoring" not in out, out


def test_manual_backward_split_rejected() -> None:
    """A manual split_loop() between a read and the later equation that writes the value is a clear DslException."""
    gf = ThornDef("ARR", "MANUALBACK")
    a, b, src = gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(a, b * 2)
    fun.split_loop()
    fun.add_eqn(b, src * 3)
    expect_dsl_exception(lambda: gf.bake(), "'b' is written in loop 1 after it is read in loop(s) [0]")


def test_duplicate_lhs_in_segment() -> None:
    gf = ThornDef("ARR", "DUPLHS")
    a, src = gf.decl("a", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_hard_split_predicate=lambda i: i == 0)
    fun.add_eqn(a, src)
    expect_dsl_exception(lambda: fun.add_eqn(a, src * 2), "bake time")

    # A manual split still separates two assignments to the same LHS.
    fun.split_loop()
    fun.add_eqn(a, src * 2)


def test_bayesian_stand_in_before_cse() -> None:
    assert pre_cse_stand_in(bayesian_optimization) is prioritize_rare_symbols
    assert pre_cse_stand_in(functools.partial(bayesian_optimization, exploration_iter=1)) is prioritize_rare_symbols
    assert pre_cse_stand_in(maximize_symbol_reuse) is maximize_symbol_reuse

    # Used as the early ordering function, Bayesian optimization is replaced before it can run.
    gf = ThornDef("ARR", "EARLYBAYES")
    a, b, src = gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(a, src)
    fun.add_eqn(b, sin(src))

    def exploding_bayesian(eqns: object, eqn_list: object) -> Iterator[Symbol]:
        raise AssertionError("Bayesian optimization must not run before CSE")

    fun._early_bake(early_ordering_fn=exploding_bayesian, pre_population_ordering_fn=exploding_bayesian,
                    ordering_fn=recipe_order)
    assert sorted(lists_of(fun)[0]) == ["a", "b"]


def test_add_eqn_order_validation() -> None:
    expect_dsl_exception(lambda: add_eqn_order([0, 1, 0]), "more than once")
    expect_dsl_exception(lambda: add_eqn_order([0, -1]), "not a valid")

    gf = ThornDef("ARR", "ORDERRANGE")
    a, src = gf.decl("a", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(a, src)
    expect_dsl_exception(lambda: fun._early_bake(early_ordering_fn=add_eqn_order([1, 0])), "out of range")


def test_unknown_function_bake_options() -> None:
    """Per-function bake options must name an existing function, so a typo cannot silently drop them."""
    gf = ThornDef("ARR", "UNKNOWNFN")
    a, src = gf.decl("a", []), gf.decl("src", [])
    fun = gf.create_function("my_rhs", ScheduleBin.Evolve)
    fun.add_eqn(a, src)
    expect_dsl_exception(lambda: gf.bake(functions={"my_rsh": {"ordering_fn": recipe_order}}), "my_rsh")


def test_full_bake_with_cse() -> None:
    """Early reordering and pre-population cuts go through a full bake with global CSE and a soft-split merge."""
    for locus in (SplitLocus.Early, SplitLocus.PrePopulation):
        gf = ThornDef("ARR", f"FULLBAKE{locus.name.upper()}")
        v = gf.decl("v", [li])
        a, b, c, src = gf.decl("a", []), gf.decl("b", []), gf.decl("c", []), gf.decl("src", [])
        fun = gf.create_function("fn", ScheduleBin.Evolve, auto_split_locus=locus,
                                 auto_hard_split_predicate=lambda i: i == 1,
                                 auto_soft_split_predicate=lambda i: i == 2)
        fun.add_eqn(a, sin(src) * cos(src) + pull_out(src * src))
        fun.add_eqn(v[li], [a * sin(src), a * cos(src), sin(src) * cos(src)])
        fun.add_eqn(b, v[l0] + v[l1] * sin(src))
        fun.add_eqn(c, b + v[l2] * sin(src) * cos(src))
        gf.bake(do_cse=True, temporary_promotion_strategy=promote_all(),
                functions={"fn": {"early_ordering_fn": add_eqn_order([0, 3, 1, 2]) if locus is SplitLocus.Early else None,
                                  "pre_population_ordering_fn": pre_population_order}})
        all_lhses = {str(lhs) for el in fun.eqn_complex.eqn_lists for lhs in el.eqns}
        assert {"a", "b", "c", "vD0", "vD1", "vD2"} <= all_lhses, all_lhses
        assert not fun.needs_merge()


if __name__ == "__main__":
    tests = [
        test_no_reorder_equivalence,
        test_early_add_eqn_order,
        test_early_key_order_dependency_repair,
        test_scc_super_groups,
        test_overwrite_anti_edge,
        test_manual_splits_never_crossed,
        test_trailing_split_dropped,
        test_boundary_folding,
        test_leading_custom_annotation_kept,
        test_empty_soft_piece_dropped_before_manual_soft_split,
        test_merge_keeps_provenance,
        test_early_cuts_see_new_order,
        test_soft_queried_only_if_hard_declines,
        test_pre_population_ordering,
        test_pre_population_cuts,
        test_params_repartitioning,
        test_validation_errors,
        test_overwrite_validation,
        test_cut_hazards,
        test_early_backward_reads,
        test_manual_backward_split_rejected,
        test_duplicate_lhs_in_segment,
        test_bayesian_stand_in_before_cse,
        test_add_eqn_order_validation,
        test_unknown_function_bake_options,
        test_full_bake_with_cse,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print("All split locus tests passed.")
