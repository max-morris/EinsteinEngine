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
Author-level equation groups, used by the early ordering locus.

A group is the set of scalar equations created by one author-level ``add_eqn`` call, i.e., those sharing an
``eqn_origin``; equations without an origin (e.g., from synthetic functions) are each their own group. The early locus
reorders whole groups, never the equations inside one:

1. The ordering function is run over the list's scalar equations, and each group is ranked by the median position of
   its members in that order (ties go to recipe order).
2. Groups can depend on each other in a cycle even when the scalar equations are acyclic. Such cycles are condensed
   into strongly connected super-groups, ranked by the median over the union of their members.
3. The super-groups are topologically sorted by priority (Kahn's algorithm on a heap): the lowest-ranked super-group
   whose dependencies have all been emitted comes next.

Besides the dependencies through RHS free symbols, a group reading ``X`` gets an explicit "reader before overwriter"
edge to a group writing ``X'`` (see ``DslFrontend.overwrite``), since no free symbol connects the two. For the same
reason, `overwrite_hazards` tells the split loci where a cut would separate a writer of ``X'`` from a later reader of
``X``.

Ordering functions run here directly, never through ``EqnList.order_builder`` or ``set_eqn_annotation``; any
annotations they yield are ignored.
"""

from __future__ import annotations

import heapq
from statistics import median
from collections import defaultdict
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Iterable

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
    The 0-based position of each equation in the order `ordering_fn` yields. Annotations are stripped, repeats are
    ignored, and equations the function does not yield rank last, in recipe order.

    The ordering function may read ``eqn_list.complexity``, so run ``_run_preliminary_complexity_analysis`` first.
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


def overwrite_version(sym: Symbol) -> tuple[str, int]:
    """For an overwrite symbol X'' return ('X', 2); for a plain symbol X return ('X', 0)."""
    name = str(sym)
    return name.replace("'", ""), name.count("'")


def overwrite_hazards(reads: Sequence[Collection[Symbol]],
                      writes: Sequence[Collection[Symbol]]) -> dict[int, set[tuple[Symbol, Symbol]]]:
    """
    The cut positions in a sequence of elements that an overwrite forbids. Element i reads `reads[i]` and writes
    `writes[i]`, and position p means "cut after element p". A cut at p is forbidden if an element at or before p writes
    a version of X (X', X'', ...) and an element after p reads an earlier version (X, X', ...), since the later loop
    would read the overwritten value. Maps each forbidden position to the (overwrite, earlier version read) pairs that
    forbid it.
    """
    overwritten = {base for syms in writes for base, version in map(overwrite_version, syms) if version > 0}
    if len(overwritten) == 0:
        return dict()

    # readers[base]: the (element, version, symbol) triples that read a version of base.
    readers: dict[str, list[tuple[int, int, Symbol]]] = {base: list() for base in overwritten}
    for idx, syms in enumerate(reads):
        for sym in syms:
            base, version = overwrite_version(sym)
            if base in readers:
                readers[base].append((idx, version, sym))

    forbidden: dict[int, set[tuple[Symbol, Symbol]]] = defaultdict(set)
    for writer_idx, syms in enumerate(writes):
        for sym in syms:
            base, version = overwrite_version(sym)
            if version == 0:
                continue
            for reader_idx, read_version, read_sym in readers[base]:
                if read_version < version and reader_idx > writer_idx:
                    for position in range(writer_idx, reader_idx):
                        forbidden[position].add((sym, read_sym))
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
    Rank the author-level groups of `eqn_list` by `ordering_fn` and sort them topologically by rank.

    Returns the resulting super-groups in order; each is a single group unless groups formed a dependency cycle.
    Members of a super-group are in recipe order. Flattening the result gives the new pre-population order.
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
        component_keys.append((float(median(positions[lhs] for lhs in members)), recipe_positions[members[0]]))

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
