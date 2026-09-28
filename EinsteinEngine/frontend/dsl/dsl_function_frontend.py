#  Copyright (C) 2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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

from __future__ import annotations

from collections import defaultdict, OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Unpack, Callable, Collection, Sequence

import EinsteinEngine.common.util as util
from sympy import Symbol, Expr, Idx, Indexed, IndexedBase, Matrix

from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.common.sympywrap import free_symbols
from EinsteinEngine.common.util import OrderedSet, pprint, wprint
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.intermediate.eqn_grouping import author_groups, order_groups, overwrite_hazards
from EinsteinEngine.intermediate.eqn_ordering import EqnOrderingFn, maximize_symbol_reuse, pre_cse_stand_in
from EinsteinEngine.intermediate.eqnlist import EqnComplex, EqnList, EqnListPiece, SplitBoundary
from EinsteinEngine.intermediate.post_population_split import assign_pieces, local_cse_temps
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy, retain_none
from EinsteinEngine.intermediate.split_locus import SplitLocus
from EinsteinEngine.intermediate.splitmaxxer import SplitMaxxer
from EinsteinEngine.intermediate.temp_kind import TempKind
from EinsteinEngine.intermediate.temporary_promotion_predicate import TemporaryPromotionPredicate
from EinsteinEngine.tuning import probe

if TYPE_CHECKING:
    from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend


class DslFunctionFrontendBakeOptions(TypedDict, total=False):
    do_madd: bool
    do_recycle_temporaries: bool
    splitmaxxing: bool
    # The post-population ordering function. It is also used for the pre-CSE bake unless pre_population_ordering_fn
    # is set. Before CSE, i.e., at the early locus and the pre-CSE bake, an ordering function whose name contains
    # 'bayesian' is replaced by prioritize_rare_symbols (see eqn_ordering.pre_cse_stand_in), since CSE rewrites that
    # order anyway; this applies to early_ordering_fn and pre_population_ordering_fn too.
    ordering_fn: EqnOrderingFn
    # Orders the author-level add_eqn groups of each list at the start of the bake (the early locus); None keeps the
    # recipe order.
    early_ordering_fn: Optional[EqnOrderingFn]
    # Orders the scalar equations for the pre-CSE bake (the pre-population locus), and the resulting order becomes the
    # pre-population order; None uses ordering_fn for the pre-CSE bake and leaves the pre-population order alone.
    pre_population_ordering_fn: Optional[EqnOrderingFn]
    soft_split_retainment_strategy: SoftSplitRetainmentStrategy


class SourceAnnotations:
    loops: dict[int, str]
    eqns: dict[int, dict[Symbol, str]]

    def __init__(self) -> None:
        self.loops = defaultdict(str)
        self.eqns = defaultdict(lambda: defaultdict(str))


class DslFunctionFrontend[FrontendT: "DslFrontend[Any, Any, Any]"]:
    name: str
    frontend: FrontendT
    source_annotations: SourceAnnotations
    eqn_complex: EqnComplex
    been_baked: bool
    been_late_baked: bool
    intent_override: Optional[IntentOverride]
    # The locus at which the auto split predicates are evaluated.
    auto_split_locus: SplitLocus

    # The annotation of each loop as requested when it was created: None = default by final index, '' = suppressed.
    _loop_annotations: list[Optional[str]]

    # The per-equation annotations of each list produced by the last _refine(), restored by _rebake_refined().
    _refined_eqn_annotations: dict[int, dict[Symbol, str]]

    def __init__(self,
                 name: str,
                 frontend: FrontendT,
                 intent_override: Optional[IntentOverride],
                 *,
                 owner_name: str,
                 auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                 auto_soft_split_predicate: Optional[Callable[[int], bool|SoftSplitRetainmentStrategy]] = None,
                 auto_split_locus: SplitLocus = SplitLocus.Early) -> None:
        self.name = name
        self.frontend = frontend
        self.source_annotations = SourceAnnotations()
        self.source_annotations.loops[0] = f"{self.name} loop 0"
        self._loop_annotations = [None]

        def set_eqn_annotation(loop_idx: int, key: Symbol, annotation: str) -> None:
            self.source_annotations.eqns[loop_idx][key] = annotation

        def clear_eqn_annotations(loop_idx: int) -> None:
            if loop_idx in self.source_annotations.eqns:
                del self.source_annotations.eqns[loop_idx]

        self.eqn_complex = EqnComplex(frontend.is_stencil, intent_override, set_eqn_annotation, clear_eqn_annotations)
        self.been_baked = False
        self.been_late_baked = False
        self.intent_override = intent_override
        self._refined_eqn_annotations = dict()
        from EinsteinEngine.frontend.dsl.add_eqn_manager import AddEqnManager
        self._add_eqn_manager = AddEqnManager(
            frontend,
            lambda: self._eqn_list,
            lambda: self.been_baked,
            owner_name=owner_name,
        )

        self._auto_hard_split_predicate = auto_hard_split_predicate
        self._auto_soft_split_predicate = auto_soft_split_predicate
        self.auto_split_locus = auto_split_locus

    def needs_merge(self) -> bool:
        return self.eqn_complex.needs_merge()

    def _on_soft_split_symbol_merged(self, mangled_sym: Symbol, sym: Symbol) -> None:
        pass

    def merge_soft_splits(self, soft_split_retainment_strategy: SoftSplitRetainmentStrategy) -> None:
        _, inv_subst = self.eqn_complex.merge_soft_splits(soft_split_retainment_strategy)

        for mangled_sym, sym in inv_subst.items():
            self._on_soft_split_symbol_merged(mangled_sym, sym)

        for el_idx in range(len(self.eqn_complex.eqn_lists)):
            self.source_annotations.loops[el_idx] = f"{self.name} loop {el_idx}"
        self._loop_annotations = [None] * len(self.eqn_complex.eqn_lists)

    @property
    def _eqn_list(self) -> EqnList:
        return self.eqn_complex.get_active_eqn_list()

    def _base_add_eqn(self, lhs2: Symbol, rhs2: Expr) -> None:
        self._add_eqn_manager._base_add_eqn(lhs2, rhs2)

    def get_free_indices(self, expr: Expr) -> OrderedSet[Idx]:
        return self.frontend.get_free_indices(expr)

    def split_loop(self, annotation: Optional[str] = None) -> None:
        if self.been_baked:
            raise DslException("Cannot split loop because the EqnComplex has already been baked.")

        loop_idx = len(self.eqn_complex.eqn_lists)
        self._loop_annotations.append(annotation)
        if annotation is None:
            annotation = self._default_loop_annotation(loop_idx, soft=False)

        if annotation.strip() != "":
            self.source_annotations.loops[loop_idx] = annotation

        self.eqn_complex._new_eqn_list()

    def soft_split(self, retainment_strategy: Optional[SoftSplitRetainmentStrategy] = None, annotation: Optional[str] = None) -> None:
        if self.been_baked:
            raise DslException("Cannot split loop because the EqnComplex has already been baked.")

        loop_idx = len(self.eqn_complex.eqn_lists)
        self._loop_annotations.append(annotation)
        if annotation is None:
            annotation = self._default_loop_annotation(loop_idx, soft=True)

        if annotation.strip() != "":
            self.source_annotations.loops[loop_idx] = annotation

        self.eqn_complex._new_eqn_list(True, soft_split_retainment_strategy=retainment_strategy)

    def _default_loop_annotation(self, loop_idx: int, *, soft: bool) -> str:
        return f"{self.name} loop {loop_idx}" + (" (soft split)" if soft else "")

    def _sync_loop_annotations(self) -> None:
        """Rewrite source_annotations.loops from _loop_annotations, naming default annotations by final index."""
        ec = self.eqn_complex
        assert len(self._loop_annotations) == len(ec.eqn_lists)
        self.source_annotations.loops.clear()
        for loop_idx, annotation in enumerate(self._loop_annotations):
            if annotation is None:
                annotation = self._default_loop_annotation(loop_idx, soft=loop_idx > 0 and loop_idx not in ec._hard_splits)
            if annotation.strip() != "":
                self.source_annotations.loops[loop_idx] = annotation

    def _refine(self,
                pieces_by_source: Sequence[Sequence[EqnListPiece]],
                *,
                tile_kind_temps: Collection[Symbol] = ()) -> None:
        """
        Cut the function's lists into pieces through `EqnComplex.refine` (see there), keeping the source annotations
        consistent: loop annotations are renamed by final index unless they were custom, and each list's per-equation
        annotations are those its equations had in their source list. Afterward, the refined lists are unbaked; call
        `self._rebake_refined(...)` if the lists had been baked before.
        """
        old_eqn_annotations = {idx: dict(annotations) for idx, annotations in self.source_annotations.eqns.items()}
        refined = self.eqn_complex.refine(pieces_by_source, tile_kind_temps=tile_kind_temps,
                                          inherited_annotations=self._loop_annotations)

        self._loop_annotations = [r.boundary.annotation if r.boundary is not None else None for r in refined]
        self._sync_loop_annotations()

        self._refined_eqn_annotations = {
            new_idx: {lhs: annotation for lhs, annotation in old_eqn_annotations.get(r.source_idx, dict()).items() if lhs in r.piece_order}
            for new_idx, r in enumerate(refined)
        }
        self.source_annotations.eqns.clear()
        self.infer_tile_temp_centerings()

    def _rebake_refined(self, *, force_fast: bool = False) -> None:
        """Rebake the lists produced by the last `_refine` (see `EqnComplex.rebake_refined`), restoring the per-equation
        annotations the equations had before they were cut."""
        self.eqn_complex.rebake_refined(force_fast=force_fast)
        for loop_idx, annotations in self._refined_eqn_annotations.items():
            for lhs, annotation in annotations.items():
                self.source_annotations.eqns[loop_idx][lhs] = annotation
        self._refined_eqn_annotations = dict()

    def infer_tile_temp_centerings(self) -> None:
        """
        Hook for frontends whose tile temps need a centering: give one to each temporary that is written in one list
        and read in another but has none yet (e.g., a pull-out temp that a cut turned into a tile temp).
        Called at the end of `_refine`; idempotent. The default does nothing.
        """
        pass

    def _query_auto_splits(self,
                           counts: Sequence[int],
                           uncuttable: Sequence[Collection[int]]) -> list[list[tuple[int, SplitBoundary]]]:
        """
        Evaluate the auto split predicates at the function's locus, where list i has `counts[i]` positions. Positions
        are 0-based and run across lists; position p means "split after element p". Each predicate is first reported
        to the probe with the total count. At each position, the hard predicate is queried first and the soft one only
        if the hard one is absent or declined.

        `uncuttable[i]` holds the positions within list i at which a cut would separate an overwrite from a later read
        of what it overwrites (see `_overwrite_hazards`). The predicates are still queried there, so the probe counts
        and what a tuner sees are unchanged, but a split they request there is ignored, with one warning per function.
        At the early locus without an early ordering function, this departs from the historical behavior of splitting
        in add_eqn wherever a cut would separate an overwrite X' from a later read of X. For a hard cut whose read of X
        is not promoted to a tile temporary, the historical code read the already overwritten variable; elsewhere (soft
        cuts, which the merge rejoins, or reads CSE promotes) the refusal is merely conservative.

        Returns, for each list, its cuts as (position within the list, boundary) pairs in order.
        """
        cuts_by_list: list[list[tuple[int, SplitBoundary]]] = [list() for _ in counts]
        hard, soft = self._auto_hard_split_predicate, self._auto_soft_split_predicate
        if hard is None and soft is None:
            return cuts_by_list

        for predicate in (hard, soft):
            if predicate is not None:
                probe.report_split_positions(predicate, self.name, sum(counts))

        ignored: list[int] = list()
        position = 0
        for cuts, count, forbidden in zip(cuts_by_list, counts, uncuttable):
            for local_position in range(count):
                boundary: Optional[SplitBoundary] = None
                if hard is not None and hard(position):
                    boundary = SplitBoundary(soft=False)
                elif soft is not None:
                    strategy: bool | SoftSplitRetainmentStrategy = soft(position)
                    if not isinstance(strategy, bool):
                        boundary = SplitBoundary(soft=True, retainment_strategy=strategy)
                    elif strategy:
                        boundary = SplitBoundary(soft=True)
                if boundary is not None:
                    if local_position in forbidden:
                        ignored.append(position)
                    else:
                        cuts.append((local_position, boundary))
                position += 1

        if len(ignored) > 0:
            wprint(f"{self.name}: ignoring the auto splits at positions {ignored} of the {self.auto_split_locus.name} "
                   f"locus, which would separate an overwrite X' from a later read of X.")
        return cuts_by_list

    @staticmethod
    def _overwrite_hazards(eqn_list: EqnList,
                           elements: Sequence[tuple[Symbol, ...]],
                           movable: Collection[Symbol] = ()) -> set[int]:
        """
        The positions in `elements` (the elements of `eqn_list` in order, see `_apply_auto_splits`) at which a cut
        would separate an overwrite X' from a later read of X (see eqn_grouping.overwrite_hazards). An element also
        reads, transitively, what the temporaries in `movable` that it reads read, since the closure assignment at the
        post-population locus may move or duplicate those temporaries into its piece. `elements` must respect
        dependencies on the `movable` temporaries, as a baked order does.
        """
        if not any("'" in str(lhs) for element in elements for lhs in element):
            return set()

        movable = set(movable)
        reads_through: dict[Symbol, set[Symbol]] = dict()
        element_reads: list[set[Symbol]] = list()
        for element in elements:
            reads: set[Symbol] = set()
            for lhs in element:
                lhs_reads = set(free_symbols(eqn_list.eqns[lhs]))
                for temp in lhs_reads.intersection(movable):
                    lhs_reads |= reads_through.get(temp, set())
                reads_through[lhs] = lhs_reads
                reads |= lhs_reads
            element_reads.append(reads)
        return overwrite_hazards(element_reads, elements)

    def _apply_auto_splits(self, elements_by_list: Sequence[Sequence[tuple[Symbol, ...]]]) -> bool:
        """
        Evaluate the auto split predicates over `elements_by_list` (for each list, its elements in order; an element is
        a group of LHSes that is never split) and cut the lists between elements; see `_query_auto_splits`. Used at the
        early and pre-population loci. Returns True iff any cut was made.

        Since the cut goes through `refine`, which drops empty pieces, an empty list left by manual splits is dropped
        too, but only when some auto split fires; otherwise it stays, as it always has.
        """
        cuts_by_list = self._query_auto_splits(
            [len(elements) for elements in elements_by_list],
            [self._overwrite_hazards(eqn_list, elements) for eqn_list, elements in zip(self.eqn_complex.eqn_lists, elements_by_list)]
        )
        if not any(cuts_by_list):
            return False

        pieces_by_source: list[list[EqnListPiece]] = list()
        for elements, cuts in zip(elements_by_list, cuts_by_list):
            pieces: list[EqnListPiece] = list()
            start = 0
            boundary: Optional[SplitBoundary] = None
            for position, cut in cuts:
                pieces.append(EqnListPiece(tuple(lhs for element in elements[start:position + 1] for lhs in element), boundary))
                start, boundary = position + 1, cut
            pieces.append(EqnListPiece(tuple(lhs for element in elements[start:] for lhs in element), boundary))
            pieces_by_source.append(pieces)

        pprint(f"Applying {self.auto_split_locus.name} auto splits to {self.name}...")
        self._refine(pieces_by_source)
        return True

    def _apply_post_population_splits(self,
                                      *,
                                      temp_kinds: Mapping[Symbol, TempKind],
                                      promotion_predicate: Optional[TemporaryPromotionPredicate],
                                      tile_temporaries: Collection[Symbol]) -> OrderedSet[Symbol]:
        """
        Evaluate the auto split predicates at the post-population locus, over each list's baked order, and cut the lists
        by closure assignment (see intermediate/post_population_split.py).

        - temp_kinds: the kind global CSE gave each temporary it materialized; empty if CSE did not run.
        - promotion_predicate: the predicate global CSE classified temporaries with, or None if CSE did not run.
        - tile_temporaries: the frontend's set of CSE tile temporaries.

        Returns the Local temps promoted to tile temps, which the caller adds to the frontend's CSE tile temporaries
        (empty if no cut was made).
        """
        if self._auto_hard_split_predicate is None and self._auto_soft_split_predicate is None:
            return OrderedSet()

        ec = self.eqn_complex
        # The complex's own set, not the tile_temporaries property: the property runs the @cache'd _calc_tile_temps,
        #  after which the complex can no longer be refined.
        all_tile_temps = set(tile_temporaries) | set(ec._tile_temporaries)
        local_temps_by_list = [local_cse_temps(eqn_list, temp_kinds, tile_temporaries) for eqn_list in ec.eqn_lists]
        tile_temps_by_list = [[lhs for lhs in eqn_list.eqns if lhs in eqn_list.synthetic_symbols and lhs in all_tile_temps]
                              for eqn_list in ec.eqn_lists]

        cuts_by_list = self._query_auto_splits(
            [len(el.order) for el in ec.eqn_lists],
            [self._overwrite_hazards(eqn_list, [(lhs,) for lhs in eqn_list.order], movable=[*local_temps, *tile_temps])
             for eqn_list, local_temps, tile_temps in zip(ec.eqn_lists, local_temps_by_list, tile_temps_by_list)]
        )
        if not any(cuts_by_list):
            return OrderedSet()

        pprint(f"Applying PostPopulation auto splits to {self.name}...")

        # Promoting to a tile temp is only sound for a temporary computed in a single source list; a Local temp that
        # global CSE placed in several lists of the function stays duplicated.
        writer_count: dict[Symbol, int] = dict()
        for eqn_list in ec.eqn_lists:
            for lhs in eqn_list.eqns:
                writer_count[lhs] = writer_count.get(lhs, 0) + 1

        def can_promote(temp: Symbol) -> bool:
            return (promotion_predicate is not None
                    and writer_count[temp] == 1
                    and TempKind.Tile.clamp(promotion_predicate(temp)) == TempKind.Tile)

        assignments = [
            assign_pieces(eqn_list, cuts, local_temps=local_temps, tile_temps=tile_temps, can_promote=can_promote)
            for eqn_list, cuts, local_temps, tile_temps in zip(ec.eqn_lists, cuts_by_list, local_temps_by_list, tile_temps_by_list)
        ]

        promoted: OrderedSet[Symbol] = OrderedSet()
        for assignment in assignments:
            promoted.update(assignment.promoted)

        # refine() keeps the complex's existing tile temps and adds the promoted ones to it.
        self._refine([assignment.pieces for assignment in assignments], tile_kind_temps=promoted)
        self._rebake_refined(force_fast=self.needs_merge())
        return promoted

    def _do_splitmaxxing(self) -> None:
        assert self.been_baked, "Cannot perform splitmaxxing because the EqnComplex has not been baked."
        assert not self.been_late_baked, "Cannot perform splitmaxxing because the EqnComplex has already been late-baked."

        for loop_idx, eqn_list in enumerate(self.eqn_complex.eqn_lists):
            new_eqns: OrderedDict[Symbol, Expr] = OrderedDict()
            modify_eqns: OrderedDict[Symbol, Expr] = OrderedDict()

            for lhs, rhs in eqn_list.eqns.items():
                splitmaxxer = SplitMaxxer(f'{self.name}_loop{loop_idx}_{str(lhs).replace("'", "_prime_")}')
                modify_eqns[lhs] = splitmaxxer.visit(rhs, top=True)
                new_eqns.update(splitmaxxer.new_eqns)

            for lhs, rhs in modify_eqns.items():
                eqn_list.eqns[lhs] = rhs

            for lhs, rhs in new_eqns.items():
                eqn_list.add_eqn(lhs, rhs)

            pprint(f"Rebaking {self.name} loop {loop_idx} after do_splitmaxxing...")
            eqn_list.bake(force_rebake=True)

            if util.verbose():
                eqn_list.dump()

    def add_eqn(self, lhs: Indexed | IndexedBase, rhs: Expr | Matrix | list[Expr]) -> None:
        # The auto split predicates are not evaluated here but at bake time, at the function's auto_split_locus.
        # The complex counts the author-level add_eqn calls; this call's 0-based index is its origin.
        self._add_eqn_manager.current_origin = self.eqn_complex.origin_count
        try:
            self._add_eqn_manager.add_eqn(lhs, rhs)
        finally:
            self._add_eqn_manager.current_origin = None
        self.eqn_complex.origin_count += 1

    def madd(self) -> None:
        self.eqn_complex.do_madd()

    def cse(self) -> None:
        self.eqn_complex.do_cse()

    def dump(self) -> None:
        self.eqn_complex.dump()

    def eqn_bake(self, ordering_fn: EqnOrderingFn) -> None:
        for eqn_list in self.eqn_complex.eqn_lists:
            eqn_list.ordering_fn = ordering_fn

        self.eqn_complex.bake()

    def recycle_temporaries(self) -> None:
        pprint(f"Recycling temporaries for {self.name}...")
        self.eqn_complex.recycle_temporaries()

    @staticmethod
    def _mk_default_dsl_function_frontend_bake_options() -> DslFunctionFrontendBakeOptions:
        return {
            "do_madd": False,
            "do_recycle_temporaries": True,
            "splitmaxxing": False,
            "ordering_fn": maximize_symbol_reuse,
            "early_ordering_fn": None,
            "pre_population_ordering_fn": None,
            "soft_split_retainment_strategy": retain_none(),
        }

    def _early_bake(self, **kwargs: Unpack[DslFunctionFrontendBakeOptions]) -> None:
        if self.been_baked:
            raise DslException("_early_bake should not be called more than once")
        pprint(f"Early Baking {self.name}...")

        options = self._mk_default_dsl_function_frontend_bake_options()
        options.update(kwargs)

        ordering_fn = options["ordering_fn"]
        early_ordering_fn = options["early_ordering_fn"]
        pre_population_ordering_fn = options["pre_population_ordering_fn"]

        # Early locus: order the author-level add_eqn groups of each list, then cut between groups. Without an early
        # ordering function, the groups stay in pre-population order, which is still the recipe order.
        early_elements: list[list[tuple[Symbol, ...]]] = list()
        if early_ordering_fn is not None:
            early_ordering_fn = pre_cse_stand_in(early_ordering_fn)
            for eqn_list in self.eqn_complex.eqn_lists:
                # The ordering function may need complexities.
                eqn_list._run_preliminary_complexity_analysis()
                elements = order_groups(eqn_list, early_ordering_fn)
                eqn_list.set_pre_population_order(lhs for element in elements for lhs in element)
                early_elements.append(elements)

        has_predicates = self._auto_hard_split_predicate is not None or self._auto_soft_split_predicate is not None
        if self.auto_split_locus is SplitLocus.Early and has_predicates:
            if early_ordering_fn is None:
                early_elements = [author_groups(el, el.eqn_pre_population_order) for el in self.eqn_complex.eqn_lists]
            self._apply_auto_splits(early_elements)

        self.eqn_complex.do_pull_out(self.frontend._unique_name('pull_out'))

        # Doing a first pass of complexity analysis for CSE
        for eqn_list in self.eqn_complex.eqn_lists:
            eqn_list._run_preliminary_complexity_analysis()

        if options["do_madd"]:
            self.madd()

        # Pre-population locus: the pre-CSE bake fixes the order CSE sees.
        self.eqn_bake(pre_population_ordering_fn or ordering_fn)

        for eqn_list in self.eqn_complex.eqn_lists:
            if pre_population_ordering_fn is not None:
                eqn_list.set_pre_population_order(eqn_list.order)
            # ordering_fn is sticky (every later rebake reuses it), so it must be the post-population function now.
            eqn_list.ordering_fn = ordering_fn

        if self.auto_split_locus is SplitLocus.PrePopulation:
            if self._apply_auto_splits([[(lhs,) for lhs in eqn_list.order] for eqn_list in self.eqn_complex.eqn_lists]):
                self._rebake_refined()

        self.been_baked = True

    def _late_bake(self, **kwargs: Unpack[DslFunctionFrontendBakeOptions]) -> None:
        if self.been_late_baked:
            raise DslException("_late_bake should not be called more than once")
        pprint(f"Late Baking {self.name}...")

        options = self._mk_default_dsl_function_frontend_bake_options()
        options.update(kwargs)

        if options["do_recycle_temporaries"]:
            self.recycle_temporaries()

        self.been_late_baked = True
