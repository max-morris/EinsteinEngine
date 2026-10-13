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
Author-level equation groups for the early ordering locus. A group is the scalar equations of one add_eqn call (those
sharing an `eqn_origin`); an equation without an origin is its own group. The early locus reorders whole groups, never
the equations inside one. Groups in a dependency cycle (possible even when the scalar equations are acyclic) are
condensed into one super-group.

A group reading X must precede a group writing X' (see `DslFrontend.overwrite`), though no free symbol connects them.

Ordering functions run here directly, not through `EqnList.order_builder`, so any annotations they yield are ignored.
"""

from __future__ import annotations

import heapq
from statistics import median
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

from sympy import Symbol

from EinsteinEngine.intermediate.dependencies import Dependencies
from EinsteinEngine.intermediate.eqn_ordering import EqnOrderingFn

if TYPE_CHECKING:
    from EinsteinEngine.intermediate.eqnlist import EqnList


def _group_key(eqn_list: EqnList, lhs: Symbol) -> int | Symbol:
    return eqn_list.eqn_origin.get(lhs, lhs)


def author_groups(eqn_list: EqnList, order: Iterable[Symbol]) -> list[tuple[Symbol, ...]]:
    """
    Partition the equations of `eqn_list` into author-level groups. Groups appear in the order their first member
    appears in `order`, and members keep their relative order from `order`.
    """
    groups: dict[int | Symbol, list[Symbol]] = dict()
    for lhs in order:
        if lhs in eqn_list.eqns:
            groups.setdefault(_group_key(eqn_list, lhs), list()).append(lhs)
    return [tuple(members) for members in groups.values()]


def scalar_positions(eqn_list: EqnList, ordering_fn: EqnOrderingFn) -> dict[Symbol, int]:
    """
    The 0-based position of each equation in the order `ordering_fn` yields; equations it does not yield rank last, in
    recipe order. The ordering function may read `eqn_list.complexity`, so run `_run_preliminary_complexity_analysis`
    first.
    """
    positions: dict[Symbol, int] = dict()
    for item in ordering_fn(eqn_list.eqns, eqn_list):
        lhs = item[0] if isinstance(item, tuple) else item
        if lhs in eqn_list.eqns and lhs not in positions:
            positions[lhs] = len(positions)
    for lhs in eqn_list.eqn_recipe_order:
        if lhs in eqn_list.eqns and lhs not in positions:
            positions[lhs] = len(positions)
    return positions


def median_rank_key(positions: Collection[int], tiebreak: int) -> tuple[float, int]:
    """
    The sort key of a group whose members sit at `positions`: their median, then `tiebreak`. An empty group ranks last.
    Shared by `order_groups` and `rank_add_eqn_calls` so that the two agree on medians and ties.
    """
    return (float(median(positions)) if len(positions) > 0 else float('inf')), tiebreak


def rank_add_eqn_calls(origins_by_list: Sequence[Mapping[Symbol, int]],
                       orders: Sequence[Sequence[Symbol]],
                       origin_count: int) -> list[int]:
    """
    Rank a function's add_eqn calls by the median position of their equations in the post-CSE orders.

    - origins_by_list[i]: maps list i's equations to the 0-based index of the add_eqn call that produced them. It must
      be snapshotted before the bake, so that pull-out and CSE temporaries do not count.
    - orders[i]: list i's order after the post-CSE rebake; equations missing from it do not count.
    - origin_count: the function's number of add_eqn calls.

    Each list's calls are sorted by `median_rank_key`, and the rankings are concatenated in list order. Calls that
    produced no equation at all rank with the last list's, so the result is a permutation of range(origin_count).
    """
    present = {origin for origins in origins_by_list for origin in origins.values()}
    ranked: list[int] = list()
    for list_idx, (origins, order) in enumerate(zip(origins_by_list, orders, strict=True)):
        position = {lhs: idx for idx, lhs in enumerate(order)}
        # A call with no surviving equation still gets an (empty) entry, so that it ranks last.
        members: dict[int, list[int]] = dict()
        for lhs, origin in origins.items():
            here = members.setdefault(origin, list())
            if lhs in position:
                here.append(position[lhs])
        if list_idx == len(origins_by_list) - 1:
            for origin in range(origin_count):
                if origin not in present:
                    members[origin] = list()
        ranked.extend(sorted(members, key=lambda origin: median_rank_key(members[origin], origin)))
    if len(origins_by_list) == 0:
        ranked.extend(range(origin_count))
    return ranked


def overwrite_version(sym: Symbol) -> tuple[str, int]:
    """For an overwrite symbol X'' return ('X', 2); for a plain symbol X return ('X', 0)."""
    name = str(sym)
    return name.replace("'", ""), name.count("'")


@dataclass(frozen=True)
class CutHazard:
    """
    A reason not to cut at a position (see `cut_hazards`).

    - An overwrite (`overwrite` is X'): the later loop would read `read`, an earlier version of X, after it was
      overwritten. This forbids every cut.
    - A backward read (`overwrite` is None): the earlier loop would read `read` before the later loop computes it.
      This forbids hard cuts only, since merge_soft_splits rejoins a soft cut's loops and the rebake puts the write
      first.
    """
    read: Symbol
    overwrite: Optional[Symbol] = field(default=None, kw_only=True)

    def forbids(self, *, soft: bool) -> bool:
        return self.overwrite is not None or not soft


def cut_hazards(reads: Sequence[Collection[Symbol]],
                writes: Sequence[Collection[Symbol]]) -> dict[int, set[CutHazard]]:
    """
    Map each position p ("cut after element p") that a `CutHazard` forbids, at least for hard cuts, to its hazards.
    Element i reads `reads[i]` and writes `writes[i]`. A baked order has no backward reads, but the recipe order of
    add_eqn calls may.
    """
    forbidden: dict[int, set[CutHazard]] = defaultdict(set)

    # Backward reads. A symbol read by the element that writes it is not a hazard (a cut never splits an element).
    writer_of = {sym: idx for idx, syms in enumerate(writes) for sym in syms}
    for reader_idx, syms in enumerate(reads):
        for sym in syms:
            if (writer_idx := writer_of.get(sym)) is not None and writer_idx > reader_idx:
                for position in range(reader_idx, writer_idx):
                    forbidden[position].add(CutHazard(sym))

    # Overwrites.
    overwritten = {base for syms in writes for base, version in map(overwrite_version, syms) if version > 0}
    if len(overwritten) == 0:
        return dict(forbidden)

    readers: dict[str, list[tuple[int, int, Symbol]]] = {base: list() for base in overwritten}
    for idx, syms in enumerate(reads):
        for sym in syms:
            base, version = overwrite_version(sym)
            if base in readers:
                readers[base].append((idx, version, sym))

    for writer_idx, syms in enumerate(writes):
        for sym in syms:
            base, version = overwrite_version(sym)
            if version == 0:
                continue
            for reader_idx, read_version, read_sym in readers[base]:
                if read_version < version and reader_idx > writer_idx:
                    for position in range(writer_idx, reader_idx):
                        forbidden[position].add(CutHazard(read_sym, overwrite=sym))
    return dict(forbidden)


def _strongly_connected_components(successors: list[set[int]]) -> list[list[int]]:
    """Tarjan's algorithm, iterative so that long dependency chains cannot exhaust the recursion limit."""
    n = len(successors)
    counter = 0
    indices = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = list()
    components: list[list[int]] = list()

    def visit(v: int) -> None:
        nonlocal counter
        indices[v] = low[v] = counter
        counter += 1
        stack.append(v)
        on_stack[v] = True

    for root in range(n):
        if indices[root] != -1:
            continue
        visit(root)
        work = [(root, iter(sorted(successors[root])))]
        while len(work) > 0:
            v, it = work[-1]
            descended = False
            for w in it:
                if indices[w] == -1:
                    visit(w)
                    work.append((w, iter(sorted(successors[w]))))
                    descended = True
                    break
                elif on_stack[w]:
                    low[v] = min(low[v], indices[w])
            if descended:
                continue

            work.pop()
            if len(work) > 0:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])

            if low[v] == indices[v]:
                component: list[int] = list()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                components.append(component)

    return components


def order_groups(eqn_list: EqnList, ordering_fn: EqnOrderingFn) -> list[tuple[Symbol, ...]]:
    """
    Rank the author-level groups of `eqn_list` by the median position of their members in the order `ordering_fn`
    yields, and sort them topologically by rank. Returns the super-groups in order, each with its members in recipe
    order; flattening the result gives the new pre-population order.
    """
    if len(eqn_list.eqns) == 0:
        return list()

    recipe_positions = eqn_list.eqn_recipe_order
    positions = scalar_positions(eqn_list, ordering_fn)

    groups = author_groups(eqn_list, sorted(eqn_list.eqns.keys(), key=lambda lhs: recipe_positions[lhs]))
    group_of = {lhs: group_idx for group_idx, members in enumerate(groups) for lhs in members}

    # successors[a] holds the groups that must come after group a.
    successors: list[set[int]] = [set() for _ in groups]

    dependencies = Dependencies(eqn_list.eqns)
    group_reads: list[set[Symbol]] = list()
    for group_idx, members in enumerate(groups):
        reads: set[Symbol] = set()
        for lhs in members:
            for dep in dependencies.dependencies.get(lhs, set()):
                reads.add(dep)
                if (dep_group := group_of.get(dep)) is not None and dep_group != group_idx:
                    successors[dep_group].add(group_idx)
        group_reads.append(reads)

    # Reader of X before writer of X' (and of X' before X'', etc.).
    for writer_idx, members in enumerate(groups):
        for lhs in members:
            base, primes = overwrite_version(lhs)
            if primes == 0:
                continue
            for reader_idx, reads in enumerate(group_reads):
                if reader_idx == writer_idx:
                    continue
                if any((v := overwrite_version(sym))[0] == base and v[1] < primes for sym in reads):
                    successors[reader_idx].add(writer_idx)

    components = _strongly_connected_components(successors)
    component_of = {group_idx: comp_idx for comp_idx, component in enumerate(components) for group_idx in component}

    component_members: list[tuple[Symbol, ...]] = list()
    component_keys: list[tuple[float, int]] = list()
    for component in components:
        members = tuple(sorted((lhs for group_idx in component for lhs in groups[group_idx]),
                               key=lambda lhs: recipe_positions[lhs]))
        component_members.append(members)
        component_keys.append(median_rank_key([positions[lhs] for lhs in members], recipe_positions[members[0]]))

    component_successors: list[set[int]] = [set() for _ in components]
    indegree = [0] * len(components)
    for group_idx, succs in enumerate(successors):
        for succ in succs:
            a, b = component_of[group_idx], component_of[succ]
            if a != b and b not in component_successors[a]:
                component_successors[a].add(b)
                indegree[b] += 1

    heap = [(component_keys[idx], idx) for idx in range(len(components)) if indegree[idx] == 0]
    heapq.heapify(heap)
    result: list[tuple[Symbol, ...]] = list()
    while len(heap) > 0:
        _, idx = heapq.heappop(heap)
        result.append(component_members[idx])
        for succ in sorted(component_successors[idx]):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                heapq.heappush(heap, (component_keys[succ], succ))

    assert len(result) == len(components), "The condensed group dependency graph should be acyclic"
    return result
