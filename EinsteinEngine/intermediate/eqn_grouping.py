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
reason, `cut_hazards` tells the split loci where a cut would separate a writer of ``X'`` from a later reader of ``X``;
it also finds the cuts that would put a read before its write, which the early locus meets without an early ordering
function, since the recipe order of the add_eqn calls need not respect their dependencies.

Ordering functions run here directly, never through ``EqnList.order_builder`` or ``set_eqn_annotation``; any
annotations they yield are ignored.
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


def median_rank_key(positions: Collection[int], tiebreak: int) -> tuple[float, int]:
    """
    The sort key of a group whose members sit at `positions` in some order: their median position, then `tiebreak`
    (lower first). A group with no positions (e.g., whose equations were all optimized away) ranks after every other.
    Used both to rank groups at the early locus (`order_groups`) and to derive an early order from a trial bake
    (`rank_add_eqn_calls`), so that the two agree on medians and ties.
    """
    return (float(median(positions)) if len(positions) > 0 else float('inf')), tiebreak


def rank_add_eqn_calls(origins_by_list: Sequence[Mapping[Symbol, int]],
                       orders: Sequence[Sequence[Symbol]],
                       origin_count: int) -> list[int]:
    """
    Rank a function's add_eqn calls by the median position of their equations; this is the ranking step of
    ``rank_by_post_population`` (see there and ``DslFrontend._derive_early_orders``).

    - origins_by_list[i]: for list i, the equations each add_eqn call produced, mapped to the call's 0-based index.
      The trial bake snapshots this before the bake, so that pull-out temporaries (which inherit the origin of the
      equation they were pulled out of) and CSE temporaries (which have none) do not count.
    - orders[i]: list i's order after the trial's post-CSE rebake. An equation missing from it does not count.
    - origin_count: the function's number of add_eqn calls.

    Within each list, the calls are sorted by `median_rank_key` of their equations' positions, ties going to the lower
    call index; a call none of whose equations is in the order goes last (in call-index order). The lists' rankings
    are concatenated in list order. A call that produced no equation at all ranks with the last list's calls that
    have none in the order, by call index, as in the offline derivation (which ranked one list and gave every call
    without a position the median ``inf``). So the result is a permutation of ``range(origin_count)``. Only the
    relative order of the calls within a list matters to the early locus, which orders each list's groups on its own.

    Example: with one list whose order is ``[u, a, vD0, vD1, vD2]`` and origins ``{u: 0, vD0: 1, vD1: 1, vD2: 1,
    a: 2}``, the medians are 0 (call 0), 3 (call 1) and 1 (call 2), so the result is ``[0, 2, 1]``.
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
            # Calls that produced no equation at all rank last with this list's calls that have none in the order.
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

    - An overwrite (`overwrite` is X'): an element at or before the cut writes X', and an element after it reads `read`,
      an earlier version of X (X, ...), which the later loop would find already overwritten. This forbids every cut.
    - A backward read (`overwrite` is None): an element at or before the cut reads `read`, which an element after it
      writes, so the earlier loop would read a value that only the later loop computes. This forbids hard cuts only: a
      soft cut is harmless, since merge_soft_splits rejoins the two loops and the rebake of the merged loop puts the
      write first, as it did before the split loci existed.
    """
    read: Symbol
    overwrite: Optional[Symbol] = field(default=None, kw_only=True)

    def forbids(self, *, soft: bool) -> bool:
        """Whether this hazard forbids a hard (`soft` False) or soft (`soft` True) cut."""
        return self.overwrite is not None or not soft


def cut_hazards(reads: Sequence[Collection[Symbol]],
                writes: Sequence[Collection[Symbol]]) -> dict[int, set[CutHazard]]:
    """
    The cut positions in a sequence of elements that a hazard (see `CutHazard`) forbids, at least for hard cuts. Element
    i reads `reads[i]` and writes `writes[i]`, and position p means "cut after element p". A cut at p is hazardous if:
    - an element at or before p writes a version of X (X', X'', ...) and an element after p reads an earlier version
      (X, X', ...), since the later loop would read the overwritten value; or
    - an element at or before p reads a symbol that an element after p writes (a backward read).
    Maps each hazardous position to its hazards. An order in which every element follows what it reads (e.g., a baked
    order) has no backward reads.
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

    # readers[base]: the (element, version, symbol) triples that read a version of base.
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
