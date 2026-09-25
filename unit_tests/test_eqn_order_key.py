#  Copyright (C) 2026 Steven R. Brandt and other Einstein Engine contributors.
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

"""Tests for auto_order_key: buffering add_eqn and replaying in a tuned order."""

from typing import Callable, Optional

from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef


def _build(order_key: Optional[Callable[[int], float]]) -> list[str]:
    """Add four independent equations and report the order they really landed in."""
    gf = ThornDef("ARR", f"TST{id(order_key) % 10000}")
    a = gf.decl("a", [])
    b = gf.decl("b", [])
    c = gf.decl("c", [])
    d = gf.decl("d", [])
    src = gf.decl("src", [])

    fun = gf.create_function("f", ScheduleBin.Evolve, auto_order_key=order_key)
    #  Independent of one another, so any permutation is legal.
    fun.add_eqn(a, src * 1)
    fun.add_eqn(b, src * 2)
    fun.add_eqn(c, src * 3)
    fun.add_eqn(d, src * 4)

    fun._flush_buffered_eqns()
    return [str(s) for s in fun.eqn_complex.eqn_lists[0].eqn_insertion_order]


def test_no_key_keeps_source_order() -> None:
    """Without a key nothing is buffered and behaviour is unchanged."""
    assert _build(None) == ['a', 'b', 'c', 'd']
    print("test_no_key_keeps_source_order: PASS")


def test_key_reorders_equations() -> None:
    """Equations are added in the order the key asks for, not as written."""
    #  Key is called with the ORIGINAL 1-based position; negating reverses.
    assert _build(lambda i: -float(i)) == ['d', 'c', 'b', 'a']

    #  An arbitrary permutation: positions 1,2,3,4 -> keys putting 3rd first.
    keys = {1: 0.9, 2: 0.1, 3: 0.05, 4: 0.5}
    assert _build(lambda i: keys[i]) == ['c', 'b', 'd', 'a']
    print("test_key_reorders_equations: PASS")


def test_equal_keys_keep_source_order() -> None:
    """Ties fall back to source order, so the permutation is always defined.

    A sampler can repeat a value; without the tie-break the resulting order
    would depend on sort implementation details rather than on the keys.
    """
    assert _build(lambda i: 0.0) == ['a', 'b', 'c', 'd']
    print("test_equal_keys_keep_source_order: PASS")


def test_splits_fire_at_positions_in_the_new_order() -> None:
    """The split predicate must see the replayed position, not the written one.

    This is the whole point of the feature: the function has to behave as
    though the equations had been written in the tuned order, which includes
    where its loops get split.
    """
    gf = ThornDef("ARR", "TSTSPLIT")
    a, b, c, d = (gf.decl(n, []) for n in "abcd")
    src = gf.decl("src", [])

    seen: list[int] = []

    def hard_split_after_second(i: int) -> bool:
        seen.append(i)
        return i == 2

    fun = gf.create_function("f", ScheduleBin.Evolve,
                             auto_order_key=lambda i: -float(i),
                             auto_hard_split_predicate=hard_split_after_second)
    fun.add_eqn(a, src * 1)
    fun.add_eqn(b, src * 2)
    fun.add_eqn(c, src * 3)
    fun.add_eqn(d, src * 4)

    assert seen == [], "predicate must not run while equations are buffered"

    fun._flush_buffered_eqns()

    assert seen == [1, 2, 3, 4], f"predicate should see replay positions, saw {seen}"
    lists = fun.eqn_complex.eqn_lists
    assert len(lists) == 2, f"expected one split into 2 loops, got {len(lists)}"
    #  Reversed order is d,c,b,a and the split lands after the 2nd -> d,c | b,a
    assert [str(s) for s in lists[0].eqn_insertion_order] == ['d', 'c']
    assert [str(s) for s in lists[1].eqn_insertion_order] == ['b', 'a']
    print("test_splits_fire_at_positions_in_the_new_order: PASS")


def test_dependencies_are_respected() -> None:
    """A key asking for an impossible order yields the closest valid one.

    Within a single loop a late dependency is repaired, but once a split falls
    between an equation and what it reads, it is not -- generation then fails.
    For the Z4c RHS only ~0.18% of random permutations respect the dependency
    graph, so treating the keys as raw sort keys would spend a search almost
    entirely on failed trials. They are used as priorities in a topological
    sort instead, which makes every draw valid.
    """
    gf = ThornDef("ARR", "TSTDEP")
    a, b, c = (gf.decl(n, []) for n in "abc")
    src = gf.decl("src", [])

    #  c <- b <- a, so the only valid order is a, b, c.
    fun = gf.create_function("f", ScheduleBin.Evolve,
                             auto_order_key=lambda i: -float(i))   # asks for c, b, a
    fun.add_eqn(a, src * 1)
    fun.add_eqn(b, a * 2)
    fun.add_eqn(c, b * 3)
    fun._flush_buffered_eqns()

    got = [str(s) for s in fun.eqn_complex.eqn_lists[0].eqn_insertion_order]
    assert got == ['a', 'b', 'c'], (
        f"dependencies must win over the key; wanted a,b,c got {got}")
    print("test_dependencies_are_respected: PASS")


def test_key_still_chooses_among_valid_orders() -> None:
    """Where dependencies allow a choice, the key decides it.

    Otherwise the topological sort would ignore the tuner entirely and the knob
    would do nothing.
    """
    def build(tag: str, key: Callable[[int], float]) -> list[str]:
        g = ThornDef("ARR", f"TSTFREE{tag}")
        #  Not x/y/z/t -- those are the coordinate symbols and are predeclared.
        aa, bb, pp, qq = (g.decl(n, []) for n in ("a", "b", "p", "q"))
        ss = g.decl("src", [])
        f = g.create_function("f", ScheduleBin.Evolve, auto_order_key=key)
        f.add_eqn(aa, ss * 1)
        f.add_eqn(bb, aa * 2)   # b depends on a; p and q are free
        f.add_eqn(pp, ss * 3)
        f.add_eqn(qq, ss * 4)
        f._flush_buffered_eqns()
        return [str(sym) for sym in f.eqn_complex.eqn_lists[0].eqn_insertion_order]

    early = build("E", lambda i: {1: 0.8, 2: 0.9, 3: 0.1, 4: 0.2}[i])
    late = build("L", lambda i: {1: 0.1, 2: 0.2, 3: 0.8, 4: 0.9}[i])

    assert early == ['p', 'q', 'a', 'b'], early
    assert late == ['a', 'b', 'p', 'q'], late
    #  a before b in both: the dependency holds regardless of the key.
    for got in (early, late):
        assert got.index('a') < got.index('b'), got
    print("test_key_still_chooses_among_valid_orders: PASS")


if __name__ == "__main__":
    test_no_key_keeps_source_order()
    test_key_reorders_equations()
    test_equal_keys_keep_source_order()
    test_splits_fire_at_positions_in_the_new_order()
    test_dependencies_are_respected()
    test_key_still_chooses_among_valid_orders()
    print("All equation order-key tests passed.")
