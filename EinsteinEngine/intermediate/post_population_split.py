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
Closure assignment for auto splits at the post-population locus, i.e., after global CSE has populated the equation
lists with temporaries ("population" here means CSE populating the lists with temporaries, not the paper's populating
of the abstract program). `DslFunctionFrontend._apply_post_population_splits` queries the predicates and hands each
source list's cuts to `assign_pieces`.

Positions are 0-based over each list's baked ``order``, running across lists; position ``i`` means "split after the
i-th equation". Cutting a list whose Local CSE temporaries are already in place cannot simply slice it, because a
temporary may be read on both sides of a cut. Instead, each piece is assigned by closure:

- The roots of a piece are the equations in its slice that are neither Local CSE temporaries nor tile temporaries.
- The piece receives its roots, plus the transitive closure of the Local CSE temporaries that the roots read.

A Local temporary needed only after a cut therefore moves, one needed on both sides is duplicated (recomputed), and a
temporary that no piece needs is dropped, so dead copies never appear.

Promotion follows the paper's rule that a soft split generates no tile temporaries. A Local temporary needed by
pieces separated only by soft cuts is always duplicated: the soft cuts are merged away again, and the retainment
strategy of merge_soft_splits then decides whether the merged loop keeps one copy or forgets and recomputes it. Only
when the pieces that need it span a hard cut may the promotion predicate from global CSE promote it to a tile
temporary instead, computed in the first piece that needs it.

A tile temporary that global CSE already placed in the list (Tile kind) is never duplicated. It moves to the first
piece of its list that reads it, so no piece computes a tile temporary that nothing in it reads.

Named symbols (recipe equations and pull-out temporaries) stay in their slices and are never duplicated; one that
crosses a hard cut becomes a tile temporary through ``EqnComplex._calc_tile_temps``, as with manual splits.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Optional

from sympy import Symbol

from EinsteinEngine.common.sympywrap import free_symbols
from EinsteinEngine.common.util import OrderedSet
from EinsteinEngine.intermediate.eqnlist import EqnList, EqnListPiece, SplitBoundary
from EinsteinEngine.intermediate.temp_kind import TempKind


def local_cse_temps(eqn_list: EqnList,
                    temp_kinds: Mapping[Symbol, TempKind],
                    tile_temporaries: Collection[Symbol]) -> OrderedSet[Symbol]:
    """The LHSes of the list that are Local CSE temporaries, given the CSE tile temporaries."""
    local_temps: OrderedSet[Symbol] = OrderedSet(
        lhs for lhs in eqn_list.eqns if lhs in eqn_list.synthetic_symbols and lhs not in tile_temporaries
    )
    for t in local_temps:
        assert temp_kinds.get(t) == TempKind.Local, f"CSE temporary {t} is materialized as Local but has kind {temp_kinds.get(t)}"
    return local_temps


@dataclass(frozen=True)
class PieceAssignment:
    """The result of `assign_pieces`."""
    pieces: list[EqnListPiece]
    promoted: OrderedSet[Symbol]  # Local temporaries promoted to tile temporaries


def assign_pieces(eqn_list: EqnList,
                  cuts: Sequence[tuple[int, SplitBoundary]],
                  *,
                  local_temps: Collection[Symbol],
                  tile_temps: Collection[Symbol],
                  can_promote: Callable[[Symbol], bool]) -> PieceAssignment:
    """
    Assign the equations of the baked `eqn_list` to the pieces that `cuts` (pairs of a 0-based position in the list's
    order and the boundary after it, sorted by position) cut it into, as the module docstring describes.

    - local_temps: the list's Local CSE temporaries, which may be duplicated or dropped.
    - tile_temps: the list's LHSes that are already tile temporaries; each moves to the first piece that reads it.
    - can_promote: whether a Local temporary whose pieces span a hard cut is promoted to a tile temporary.
    """
    order = eqn_list.order
    assert set(order) == set(eqn_list.eqns.keys()), \
        f"The order of a baked list must cover exactly its equations; got {order} for {list(eqn_list.eqns.keys())}"

    position_of = {lhs: pos for pos, lhs in enumerate(order)}

    # Slice the order after each cut position. A cut after the final equation leaves an empty trailing slice.
    slice_starts = [0, *(pos + 1 for pos, _ in cuts)]
    slice_ends = [*slice_starts[1:], len(order)]
    slices = [order[start:end] for start, end in zip(slice_starts, slice_ends)]
    boundaries: list[Optional[SplitBoundary]] = [None, *(boundary for _, boundary in cuts)]
    slice_of = {lhs: piece_idx for piece_idx, sl in enumerate(slices) for lhs in sl}

    def spans_hard_cut(pieces: set[int]) -> bool:
        return any((b := boundaries[idx]) is not None and not b.soft for idx in range(min(pieces) + 1, max(pieces) + 1))

    movable = set(local_temps) | {t for t in tile_temps if t in eqn_list.eqns}
    movable_reads = {lhs: free_symbols(rhs).intersection(movable) for lhs, rhs in eqn_list.eqns.items()}

    # need[t]: the pieces that read t. placed_in[lhs]: the pieces that compute lhs.
    # Readers come after what they read in the order, so walking it backward settles every reader of t first.
    need: dict[Symbol, set[int]] = {t: set() for t in movable}
    placed_in: dict[Symbol, set[int]] = dict()
    promoted: OrderedSet[Symbol] = OrderedSet()

    for lhs in reversed(order):
        if lhs in local_temps:
            if len(need[lhs]) > 1 and spans_hard_cut(need[lhs]) and can_promote(lhs):
                promoted.add(lhs)
                placed_in[lhs] = {min(need[lhs])}
            else:
                placed_in[lhs] = need[lhs]
        elif lhs in movable:
            placed_in[lhs] = {min(need[lhs])} if len(need[lhs]) > 0 else {slice_of[lhs]}
        else:
            placed_in[lhs] = {slice_of[lhs]}

        for t in movable_reads[lhs]:
            need[t].update(placed_in[lhs])

    pieces: list[EqnListPiece] = list()
    for piece_idx, (sl, boundary) in enumerate(zip(slices, boundaries)):
        members = {lhs for lhs in order if piece_idx in placed_in[lhs]}
        for lhs in members:
            assert slice_of[lhs] <= piece_idx, \
                f"{lhs} is needed in piece {piece_idx} but sits in the later slice {slice_of[lhs]}"
        pieces.append(EqnListPiece(lhses=_piece_order(sl, members, movable_reads, position_of), boundary=boundary))

    return PieceAssignment(pieces, promoted)


def _piece_order(sl: list[Symbol],
                 members: set[Symbol],
                 movable_reads: dict[Symbol, set[Symbol]],
                 position_of: dict[Symbol, int]) -> tuple[Symbol, ...]:
    """
    The slice keeps its order. A temporary borrowed from an earlier slice is placed just before its first reader
    (after its own borrowed dependencies), so the result respects dependencies without relying on repair.
    """
    in_slice = set(sl)
    emitted: set[Symbol] = set()
    result: list[Symbol] = list()

    def emit(lhs: Symbol) -> None:
        if lhs in emitted:
            return
        emitted.add(lhs)
        for dep in sorted(movable_reads[lhs].intersection(members).difference(in_slice), key=position_of.__getitem__):
            emit(dep)
        result.append(lhs)

    for lhs in sl:
        if lhs in members:
            emit(lhs)

    assert emitted == members, f"Unreachable piece members: {members - emitted}"
    return tuple(result)
