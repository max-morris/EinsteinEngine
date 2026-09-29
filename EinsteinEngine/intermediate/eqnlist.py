#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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

import copy
import typing
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
from functools import cache, partial
from functools import cached_property
from itertools import chain
from statistics import mean, median
from typing import cast, Dict, List, Optional, Set, Callable, Iterable, NamedTuple, Never, Generator, Sequence, Collection

from multimethod import multimethod
from termcolor import colored
from sortedcontainers import SortedDict
# noinspection PyUnusedImports
from sympy import Basic, IndexedBase, Symbol, Integer, Expr

from EinsteinEngine.intermediate.analytic_function_checker import AnalyticFunctionChecker
from EinsteinEngine.intermediate.dependencies import Dependencies
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.intermediate.eqn_ordering import maximize_symbol_reuse, EqnOrderingFn, score_memory_pressure, pre_cse_stand_in, \
    respects_dependency_order, fixed_order
from EinsteinEngine.frontend.definitions import *
from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy
from EinsteinEngine.common.stencil_idx import StencilIdxWithName, StencilIdx
from EinsteinEngine.intermediate.symbify import symbify
from EinsteinEngine.common.sympywrap import *
from EinsteinEngine.frontend.util import require_baked
from EinsteinEngine.emit.ccl.schedule.schedule_tree import IntentRegion
from EinsteinEngine.generators.sympy_complexity import SympyComplexityVisitor
from EinsteinEngine.common.util import OrderedSet, consolidate, vprint, wprint, pprint, get_or_compute
from EinsteinEngine.intermediate.intermediate_exception import IntermediateException
from EinsteinEngine.intermediate.loop_region import infer_loop_region, may_be_grid_variable
from EinsteinEngine.intermediate.eqn_grouping import overwrite_version


# These symbols represent the inverse of the
# spatial discretization.
# DXI = mk_symbol("DXI")
# DYI = mk_symbol("DYI")
# DZI = mk_symbol("DZI")
# DX = mk_symbol("DX")
# DY = mk_symbol("DY")
# DZ = mk_symbol("DZ")
#
# stencil = mk_function("stencil")

def _copy_containers[T](value: T) -> T:
    """
    A copy of `value` if it is a dict, set or list (keeping its type, e.g. OrderedDict or OrderedSet), copied
    recursively through dict values: the containers a dict holds as values are copied the same way, while a list or
    set is copied shallowly (its elements are shared). Any other value, e.g. a sympy expression or a function, is
    shared. Used by the trial copies (see `EqnComplex.trial_copy`), whose state is containers of immutable values.
    """
    if isinstance(value, dict):
        copied = copy.copy(value)
        for key, item in value.items():
            copied[key] = _copy_containers(item)
        return cast(T, copied)
    if isinstance(value, (set, list)):
        return cast(T, copy.copy(value))
    return value


class _MergeSoftSplitsResult(NamedTuple):
    subst: dict[Symbol, set[Symbol]]
    inv_subst: dict[Symbol, Symbol]

    @classmethod
    def get_unit(cls) -> '_MergeSoftSplitsResult':
        return cls(dict(), dict())

@dataclass
class TemporaryLifetime:
    symbol: Symbol
    prime: int
    read_at: OrderedSet[int]
    written_at: int
    replaces: Optional["TemporaryLifetime"]
    is_superseded: bool
    is_dead: bool

    def __str__(self) -> str:
        ticks = "'" * self.prime
        return f'{self.symbol}{ticks}'

    def __hash__(self) -> int:
        return (self.symbol, self.prime).__hash__()

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, TemporaryLifetime)
                and self.symbol.__eq__(__value.symbol)  # type: ignore[no-untyped-call]
                and self.prime.__eq__(__value.prime))

    @cached_property
    def final_read(self) -> int:
        return max(self.read_at)


@dataclass(frozen=True)
class TemporaryReplacement:
    old: Symbol
    new: Symbol
    begin_eqn: int
    end_eqn: int


@dataclass(frozen=True)
class SplitBoundary:
    """How a piece is separated from the piece before it."""
    soft: bool  # False = hard split
    retainment_strategy: Optional[SoftSplitRetainmentStrategy] = None  # Soft only; None = use the bake default
    annotation: Optional[str] = None  # None = default loop annotation by final index; '' = none


@dataclass(frozen=True)
class EqnListPiece:
    """
    One piece of a source EqnList, as passed to `EqnComplex.refine`.

    `lhses` is ordered, and `EqnComplex.rebake_refined` preserves this order (with dependency repair). A LHS may appear
    in several pieces of the same source list only if it is a synthetic symbol (a duplicated Local CSE temp); its RHS
    is taken from the source list.

    `boundary` MUST be None for the first piece of each source list (it inherits the source list's existing boundary)
    and MUST NOT be None for every other piece.
    """
    lhses: tuple[Symbol, ...]
    boundary: Optional[SplitBoundary]


@dataclass(frozen=True)
class RefinedList:
    """One EqnList produced by `EqnComplex.refine`: where it was cut from and how it is separated from the list before."""
    source_idx: int  # The index of the source list it was cut from
    # True iff its boundary comes from the boundary its source list had before refinement (a manual split), possibly
    #  made hard by folding
    inherits_boundary: bool
    boundary: Optional[SplitBoundary]  # None only for the first list of the complex
    piece_order: tuple[Symbol, ...]  # Its LHSes in the order of its piece, which rebake_refined preserves
    # The first list of the complex has no boundary, but keeps the annotation of the boundary it had before the lists
    #  in front of it were dropped (e.g., a custom annotation on a leading manual split)
    first_annotation: Optional[str] = None

    @property
    def annotation(self) -> Optional[str]:
        """The list's loop annotation: None = default annotation by final index; '' = none."""
        return self.boundary.annotation if self.boundary is not None else self.first_annotation

    def fold_into(self, kept: 'RefinedList') -> 'RefinedList':
        """
        `kept` with the boundary that results from folding in the boundary of this dropped, empty piece. A hard
        boundary wins its kind; between two soft boundaries, the one of `kept`, which is attached to actual equations,
        wins. The annotation is the first one that is not None of: the inherited (manual) boundaries' (`kept`'s first),
        the winning boundary's, and the other boundary's.
        """
        assert self.boundary is not None and kept.boundary is not None
        hard_wins = not self.boundary.soft and kept.boundary.soft
        winner, loser = (self.boundary, kept.boundary) if hard_wins else (kept.boundary, self.boundary)
        candidates = [r.boundary.annotation for r in (kept, self) if r.inherits_boundary and r.boundary is not None]
        annotation = next((a for a in (*candidates, winner.annotation, loser.annotation) if a is not None), None)
        return replace(kept,
                       inherits_boundary=kept.inherits_boundary or self.inherits_boundary,
                       boundary=replace(winner, annotation=annotation))


@dataclass(frozen=True)
class _SourceSnapshot:
    """What post-refinement validation needs to know about a source list that had already been baked."""
    params: frozenset[Symbol]
    outputs: frozenset[Symbol]
    write_decls: dict[Symbol, IntentRegion]


class EqnComplex:
    eqn_lists: list['EqnList']
    is_stencil: dict[UFunc, bool]
    intent_override: Optional[IntentOverride]
    been_baked: bool

    _tile_temporaries: set[Symbol]
    _inputs: set[Symbol]
    _outputs: set[Symbol]
    _params: set[Symbol]
    _temporaries: set[Symbol]
    _read_decls: dict[Symbol, IntentRegion]
    _write_decls: dict[Symbol, IntentRegion]
    _variables: set[Symbol]

    # Contains indices of EqnLists that are the first element in a kernel, i.e., the EqnList generated after a call to split_loop().
    # This set should NOT contain index 0; it is always the first element of the first kernel, so storing it would be redundant.
    _hard_splits: set[int]

    # Maps EqnList indices to their corresponding SoftSplitRetainmentStrategies as set by soft_split()
    _soft_split_retainment_strategies: dict[int, SoftSplitRetainmentStrategy]

    # The number of author-level add_eqn calls made on the owning function; valid eqn_origin values are 0..origin_count-1.
    origin_count: int

    # Set once any of the @cache'd _calc_* methods has run. refine() must happen before that.
    _derived_state_computed: bool

    # One entry per EqnList produced by the most recent refine(), consumed by rebake_refined().
    _refinement: Optional[list[RefinedList]]
    _refinement_snapshots: list[Optional[_SourceSnapshot]]

    def __init__(self,
                 is_stencil: Dict[UFunc, bool],
                 intent_override: Optional[IntentOverride] = None,
                 set_eqn_annotation: Optional[Callable[[int, Symbol, str], None]] = None,
                 clear_eqn_annotations: Optional[Callable[[int], None]] = None) -> None:
        self.is_stencil = is_stencil
        self.intent_override = intent_override
        self.set_eqn_annotation = set_eqn_annotation
        self.clear_eqn_annotations = clear_eqn_annotations
        self._next_recipe_position = 0
        self.origin_count = 0
        self._derived_state_computed = False
        self._refinement = None
        self._refinement_snapshots = list()
        self.eqn_lists = [self._mk_eqn_list(0)]
        self.been_baked = False
        self._tile_temporaries = OrderedSet()
        self._hard_splits = set()
        self._soft_split_retainment_strategies = dict()

    def _mk_eqn_list(self, el_idx: int) -> 'EqnList':
        new_list = EqnList(self, self.is_stencil)
        self._bind_annotation_callbacks(new_list, el_idx)
        return new_list

    def _bind_annotation_callbacks(self, eqn_list: 'EqnList', el_idx: int) -> None:
        eqn_list.set_eqn_annotation = partial(self.set_eqn_annotation, el_idx) if self.set_eqn_annotation else None
        eqn_list.clear_eqn_annotations = partial(self.clear_eqn_annotations, el_idx) if self.clear_eqn_annotations else None

    def _take_recipe_position(self) -> int:
        position = self._next_recipe_position
        self._next_recipe_position += 1
        return position

    def _new_eqn_list(self, soft_split: bool = False, soft_split_retainment_strategy: Optional[SoftSplitRetainmentStrategy] = None) -> 'EqnList':
        new_list = self._mk_eqn_list(len(self.eqn_lists))
        self.eqn_lists.append(new_list)

        if not soft_split:
            self._hard_splits.add(len(self.eqn_lists) - 1)

        if soft_split_retainment_strategy is not None:
            self._soft_split_retainment_strategies[len(self.eqn_lists) - 1] = soft_split_retainment_strategy

        return new_list

    def bake(self) -> None:
        if self.been_baked:
            raise DslException("Can't bake an EqnComplex that has already been baked.")
        self.been_baked = True

        for eqn_list in self.eqn_lists:
            eqn_list.bake()

    def get_active_eqn_list(self) -> 'EqnList':
        return self.eqn_lists[-1]

    def trial_copy(self) -> 'EqnComplex':
        """
        An independent copy of this unbaked complex, for the trial bake of ``rank_by_post_population`` (see
        ``DslFrontend._derive_early_orders``): baking the copy changes nothing in this complex. Every attribute is
        copied, containers recursively through dict values (see `_copy_containers`), except that the copy gets its own
        copies of the EqnLists (whose ``parent`` is the copy, and whose annotation callbacks are bound to the copy's
        list indices), and shares ``is_stencil`` (the frontend's) and the annotation callbacks (the owning function's,
        which write into whatever ``source_annotations`` the function holds when they are called).

        Copying attribute by attribute, rather than listing the attributes to copy, means an attribute added later is
        copied too. The complex must be unbaked, with no derived state and no pending refinement, so its state is
        just containers of symbols, expressions and ints.
        """
        if self.been_baked or self._derived_state_computed or self._refinement is not None:
            raise IntermediateException("Only an unbaked EqnComplex can be copied for a trial bake.")
        clone = copy.copy(self)
        for name, value in vars(self).items():
            if name not in ('eqn_lists', 'is_stencil'):
                setattr(clone, name, _copy_containers(value))
        clone.eqn_lists = [eqn_list._trial_copy(clone, el_idx) for el_idx, eqn_list in enumerate(self.eqn_lists)]
        return clone

    def _grid_variables(self) -> set[Symbol]:
        gv: set[Symbol] = set()
        for eqn_list in self.eqn_lists:
            gv |= eqn_list._grid_variables()
        return gv

    def do_pull_out(self, name_generator: Generator[str, Never, Never]) -> None:
        for eqn_list in self.eqn_lists:
            eqn_list.do_pull_out(name_generator)

    def do_madd(self) -> None:
        for eqn_list in self.eqn_lists:
            eqn_list.madd()

    def do_cse(self) -> None:
        old_shape: list[int] = list()
        old_lhses: list[Symbol] = list()
        old_rhses: list[Expr] = list()

        for el in self.eqn_lists:
            old_shape.append(0)
            for lhs, rhs in el.eqns.items():
                old_lhses.append(lhs)
                old_rhses.append(rhs)
                old_shape[-1] += 1

        substitutions_list: list[tuple[Symbol, Expr]]
        new_rhses: list[Expr]
        substitutions_list, new_rhses = cse(old_rhses)

        substitutions = {lhs: rhs for lhs, rhs in substitutions_list}
        substitutions_order = {lhs: idx for idx, (lhs, _) in enumerate(substitutions_list)}

        new_temp_reads: dict[Symbol, set[int]] = {sym: set() for sym in substitutions.keys()}
        new_temp_dependencies: dict[Symbol, set[Symbol]] = {sym: set() for sym in substitutions.keys()}


        # We need to figure out exactly which loops use which temporaries.
        # By doing this, we can determine which temporaries need to be promoted to tile temporaries and which loop each
        #  temporary should be computed in.
        # We will also populate the temporary-related bookkeeping fields on EqnList and EqnComplex.

        global_eqn_idx = 0
        for el_idx, el_shape in enumerate(old_shape):
            eqn_list = self.eqn_lists[el_idx]
            el_new_free_symbols: set[Symbol] = set(chain(*[free_symbols(rhs) for rhs in new_rhses[global_eqn_idx:global_eqn_idx + el_shape]]))
            new_temps = el_new_free_symbols.intersection(substitutions.keys())

            for new_temp, temp_rhs in [(new_temp, substitutions[new_temp]) for new_temp in new_temps]:
                assert new_temp not in eqn_list.inputs
                assert new_temp not in eqn_list.params
                assert new_temp not in eqn_list.outputs
                assert new_temp not in eqn_list.eqns

                new_temp_reads[new_temp].add(el_idx)

                # Temps might be substituted for expressions which contain other temps.
                # We need to recursively check the RHSes to ensure we compute the dependencies in the appropriate loops.
                def drill(lhs: Symbol, rhs: Expr) -> None:
                    temp_dependencies = free_symbols(rhs).intersection(substitutions.keys())
                    assert lhs not in temp_dependencies
                    for td in temp_dependencies:
                        new_temp_dependencies[lhs].add(td)
                        drill(td, substitutions[td])

                drill(new_temp, temp_rhs)

            for lhs in old_lhses[global_eqn_idx:global_eqn_idx + el_shape]:
                assert lhs in eqn_list.eqns
                eqn_list.eqns[lhs] = new_rhses[global_eqn_idx]
                global_eqn_idx += 1


        for new_temp, temp_dependencies in sorted(new_temp_dependencies.items(),
                                                  key=lambda kv: substitutions_order[kv[0]],
                                                  reverse=True):
            el_idx = min(new_temp_reads[new_temp])
            for td in temp_dependencies:
                new_temp_reads[td].add(el_idx)

        for new_temp, el_list in new_temp_reads.items():
            if (seen_count := len(el_list)) == 0:
                continue

            primary_el = self.eqn_lists[primary_idx := min(el_list)]
            primary_el.add_eqn(new_temp, substitutions[new_temp])

            if seen_count == 1:
                primary_el.temporaries.add(new_temp)
            else:
                self._tile_temporaries.add(new_temp)
                primary_el.uninitialized_tile_temporaries.add(new_temp)
                for eqn_list in [self.eqn_lists[el_idx] for el_idx in el_list if el_idx != primary_idx]:
                    eqn_list.preinitialized_tile_temporaries.add(new_temp)

    def dump(self) -> None:
        for idx, eqn_list in enumerate(self.eqn_lists):
            print(f'=== Loop {idx} ===')
            eqn_list.dump()
            print()

    def recycle_temporaries(self) -> None:
        for eqn_list in self.eqn_lists:
            eqn_list.recycle_temporaries()

    def needs_merge(self) -> bool:
        return len(self._hard_splits) < len(self.eqn_lists) - 1

    def merge_soft_splits(self, soft_split_retainment_strategy: SoftSplitRetainmentStrategy) -> _MergeSoftSplitsResult:
        all_eqns: dict[Symbol, Expr] = dict()

        for el in self.eqn_lists:
            all_eqns.update(el.eqns)

        dependencies = Dependencies(all_eqns)

        hard_splits = list(sorted({0, *self._hard_splits, len(self.eqn_lists)}))
        soft_ranges: list[tuple[int, int]] = list()

        all_subst: dict[Symbol, set[Symbol]] = dict()
        inv_subst: dict[Symbol, Symbol] = dict()

        name_mangle_counter = 0
        def mangle(name: str) -> str:
            nonlocal name_mangle_counter
            s = f'{name}_ss{name_mangle_counter}'
            name_mangle_counter += 1
            return s

        def mangle_sym(sym: Symbol) -> Symbol:
            return Symbol(mangle(str(sym)))  # type: ignore[no-untyped-call]

        for idx, hard_split in enumerate(hard_splits[1:], start=1):
            first, last = hard_splits[idx - 1], hard_split - 1
            if last - first > 0:
                soft_ranges.append((first, last))

        if len(soft_ranges) == 0:
            return _MergeSoftSplitsResult.get_unit()

        els_to_delete: list[int] = list()

        for first_el, last_el in soft_ranges:
            local_temp_set: set[Symbol] = set()
            local_temp_last_read: dict[Symbol, int] = dict()
            local_temp_first_write: dict[Symbol, int] = dict()
            local_temp_complexities: dict[Symbol, int] = dict()

            candidate_set_by_kernel: dict[int, set[Symbol]] = {i: set() for i in range(first_el, last_el + 1)}

            local_mangled_reads_by_kernel: dict[int, set[Symbol]] = defaultdict(set)

            for el_idx, el in enumerate(self.eqn_lists[first_el:last_el + 1], start=first_el):
                lt = el.local_temporaries

                writes = set(el.eqns.keys())
                for t in lt:
                    local_temp_last_read[t] = el_idx
                    if t in writes and t not in local_temp_first_write:
                        local_temp_first_write[t] = el_idx

                local_temp_set.update(lt)
                local_temp_complexities.update(
                    (sym, complexity) for sym, complexity in el.complexity.items()
                    if sym in lt and complexity > local_temp_complexities.get(sym, 0)
                )

            for t in local_temp_set:
                # Symbols are candidates for being "forgotten" in a certain kernel iff:
                # 1) They were written to in a previous kernel.
                # 2) Their last read is in the current or a later kernel.
                candidate_start, candidate_end = local_temp_first_write[t] + 1, local_temp_last_read[t]
                if candidate_start > candidate_end:
                    continue
                for el_idx in range(candidate_start, candidate_end + 1):
                    candidate_set_by_kernel[el_idx].add(t)

            # We can choose to forget a symbol in more than one kernel. Each time a symbol is forgotten, we mangle its
            # name and recompute it along with any forgotten dependencies. If a symbol that has previously been forgotten
            # in one kernel is retained in a future kernel, we use the most recent mangling of the name.
            forgotten_by_kernel: dict[int, set[Symbol]] = dict()
            subst_by_kernel: dict[int, dict[Symbol, Symbol]] = dict()
            most_recent_mangling: dict[Symbol, Symbol] = dict()

            for el_idx, candidate_set in candidate_set_by_kernel.items():
                candidate_complexities = {sym: c for sym, c in local_temp_complexities.items() if sym in candidate_set}
                should_retain = self._soft_split_retainment_strategies.get(el_idx, soft_split_retainment_strategy)(candidate_complexities)
                forgotten = {sym for sym in candidate_set if not should_retain(sym)}
                forgotten_by_kernel[el_idx] = forgotten

                subst_by_kernel[el_idx] = dict(most_recent_mangling)

                # When mangling, we sort the symbols by their string representation to ensure deterministic code generation.
                for sym in sorted(forgotten, key=str):
                    mangled = mangle_sym(sym)
                    subst_by_kernel[el_idx][sym] = mangled
                    most_recent_mangling[sym] = mangled

                    all_subst.setdefault(sym, set()).add(mangled)
                    inv_subst[mangled] = sym

            new_eqns: dict[Symbol, Expr] = dict()
            recipient_el = self.eqn_lists[first_el]
            # The list each absorbed equation that keeps its LHS comes from; every other new equation is a mangled copy.
            absorbed_from: dict[Symbol, EqnList] = dict()
            range_origins: dict[Symbol, int] = dict()
            for el in self.eqn_lists[first_el:last_el + 1]:
                range_origins.update(el.eqn_origin)

            completed_syms: set[Symbol] = set()
            for el_idx in range(first_el + 1, last_el + 1):
                eqn_list = self.eqn_lists[el_idx]
                subst = subst_by_kernel[el_idx]

                local_mangled_reads: set[Symbol] = set()
                local_mangled_reads_by_kernel[el_idx] = local_mangled_reads

                # Absorb the equations in their pre-population order, which the recipient's pre-population order extends.
                assert eqn_list.eqn_pre_population_order.keys() == eqn_list.eqns.keys()
                for lhs in eqn_list.eqn_pre_population_order:
                    rhs = eqn_list.eqns[lhs]
                    new_lhs = subst.get(lhs, lhs)

                    if new_lhs in new_eqns or new_lhs in recipient_el.eqns:
                        continue

                    new_rhs = rhs.xreplace(subst)  # type: ignore[no-untyped-call]

                    eqn_mangled_reads = new_rhs.free_symbols.intersection(inv_subst.keys())
                    local_mangled_reads.update(eqn_mangled_reads)

                    new_eqns[new_lhs] = new_rhs
                    if new_lhs == lhs:
                        absorbed_from[lhs] = eqn_list

                # Make sure all mangled symbols have definitions
                check = list(local_mangled_reads)
                while len(check) > 0:
                    mangled_sym = check.pop()
                    if mangled_sym in completed_syms:
                        continue
                    if mangled_sym not in new_eqns and mangled_sym in inv_subst:
                        completed_syms.add(mangled_sym)
                        sym = inv_subst[mangled_sym]
                        new_eqns[mangled_sym] = all_eqns[sym].xreplace(subst)  # type: ignore[no-untyped-call]
                        for td in dependencies.get_transitive_dependencies(sym).intersection(subst.keys()):
                            check.append(subst[td])

                els_to_delete.append(el_idx)
                recipient_el.params.update(eqn_list.params)
                recipient_el.analytic_seed.update(eqn_list.analytic_seed)

            # Drop mangled copies that nothing reads, along with the mangled copies only they read. This happens when a
            # Local temporary that CSE placed in several lists is retained, so that its copy in a later list is skipped,
            # while a temporary that copy read is forgotten there: the forgotten one is still recomputed under its
            # mangled name, but its only reader is gone. Left in, it would become an output of the merged list.
            read_count: dict[Symbol, int] = defaultdict(int)
            for rhs in chain(recipient_el.eqns.values(), new_eqns.values()):
                for sym in free_symbols(rhs):
                    read_count[sym] += 1
            dead = [lhs for lhs in new_eqns if lhs in inv_subst and read_count[lhs] == 0]
            while len(dead) > 0:
                mangled_sym = dead.pop()
                for sym in free_symbols(new_eqns.pop(mangled_sym)):
                    read_count[sym] -= 1
                    if read_count[sym] == 0 and sym in new_eqns and sym in inv_subst:
                        dead.append(sym)
                original = inv_subst.pop(mangled_sym)
                all_subst[original].discard(mangled_sym)
                if len(all_subst[original]) == 0:
                    del all_subst[original]

            # An absorbed equation keeps its recipe position (the recipe order is never rewritten), origin, and recorded
            # params. A mangled copy of a forgotten temporary is a new temporary: it takes a new recipe position and
            # inherits the origin of the temporary it copies.
            for lhs, rhs in new_eqns.items():
                if (source := absorbed_from.get(lhs)) is not None:
                    recipient_el.add_eqn(lhs, rhs,
                                         recipe_position=source.eqn_recipe_order[lhs],
                                         origin=source.eqn_origin.get(lhs, None),
                                         recorded_params=source.eqn_recorded_params.get(lhs, None))
                else:
                    recipient_el.add_eqn(lhs, rhs, origin=range_origins.get(inv_subst[lhs], None))

        for el_idx in reversed(els_to_delete):
            del self.eqn_lists[el_idx]

        for el_idx, el in enumerate(self.eqn_lists):
            self._bind_annotation_callbacks(el, el_idx)

        pprint(f'Rebaking loops after merge_soft_splits...')
        # Rebaking all loops instead of just recipients because CSE will have rebaked with force_fast=True.
        # This also realigns ordering annotations for us.
        for el in self.eqn_lists:
            el.bake(force_rebake=True)

        self._hard_splits = set(range(1, len(self.eqn_lists)))

        return _MergeSoftSplitsResult(all_subst, inv_subst)

    def refine(self,
               pieces_by_source: Sequence[Sequence[EqnListPiece]],
               *,
               tile_kind_temps: Collection[Symbol] = (),
               inherited_annotations: Sequence[Optional[str]] = ()) -> list[RefinedList]:
        """
        Split each existing EqnList into ordered pieces; every cut, at any split locus, goes through this method.
        `pieces_by_source[i]` holds the pieces of `self.eqn_lists[i]`; pieces never span two source lists.

        Each piece becomes a new, unbaked EqnList carrying its own copy of the equations, both orders, eqn_origin,
        the recorded params, `ordering_fn`, and:
        - params: {DXI, DYI, DZI}, plus the source params the piece's RHSes read, plus the params recorded for the
          piece's recipe equations;
        - synthetic_symbols intersected with the symbols the piece uses;
        - rebuilt tile sets for `tile_kind_temps` and the complex's existing tile temps: uninitialized in the piece that
          writes the temp, preinitialized in the pieces that read it.
        Hard splits and retainment strategies are re-keyed; an absent strategy stays absent, so the bake default still
        applies.

        The first piece of each source list inherits the source list's boundary, with the loop annotation
        `inherited_annotations[i]` (None, the default, names the loop by its final index).

        Empty pieces are dropped, and the boundary of a dropped piece is folded into the next list's (a hard boundary
        wins; see `RefinedList.fold_into`). An empty piece computes nothing, so this deliberately departs from the
        historical behavior of splitting in add_eqn, which kept such lists: an auto split right before a manual split
        (or after the last equation) emitted an empty loop if hard, and if soft, merge_soft_splits applied the auto
        split's retainment strategy at the empty list before applying the manual split's. If every list in front of a
        list is dropped, it becomes the first list and loses its boundary, but keeps the boundary's annotation.

        If a source list had already been baked, each of its pieces is seeded with the source's analytic symbols so
        that the inferred write regions of its outputs do not change, and `rebake_refined` validates the result.

        Returns a `RefinedList` for each resulting list.
        """
        assert not self._derived_state_computed \
               and 'stencil_limits' not in self.__dict__ \
               and 'stencil_idxes' not in self.__dict__, \
            "Cannot refine an EqnComplex after its derived state (tile temps, vars, decls, stencils) has been computed."

        self._check_pieces(pieces_by_source)
        if len(inherited_annotations) not in (0, len(self.eqn_lists)):
            raise IntermediateException(f"refine() expects an inherited annotation for each of the {len(self.eqn_lists)} EqnLists, got {len(inherited_annotations)}.")
        refined = self._flatten_pieces(pieces_by_source, inherited_annotations)

        snapshots: list[Optional[_SourceSnapshot]] = list()
        seeds: list[set[Symbol]] = list()
        for src in self.eqn_lists:
            if src.been_baked:
                snapshots.append(_SourceSnapshot(frozenset(src.params), frozenset(src.outputs), dict(src.write_decls)))
                seeds.append(src._analytic_symbols())
            else:
                snapshots.append(None)
                seeds.append(set(src.analytic_seed))

        new_lists: list[EqnList] = list()
        new_hard_splits: set[int] = set()
        new_strategies: dict[int, SoftSplitRetainmentStrategy] = dict()
        for new_idx, r in enumerate(refined):
            new_list = self._mk_eqn_list(new_idx)
            new_list._populate_from_piece(self.eqn_lists[r.source_idx], r.piece_order, seeds[r.source_idx])
            new_lists.append(new_list)

            if r.boundary is None:
                continue
            if not r.boundary.soft:
                new_hard_splits.add(new_idx)
            elif (strategy := r.boundary.retainment_strategy) is not None:
                new_strategies[new_idx] = strategy

        self._tile_temporaries = self._rebuild_tile_sets(new_lists, tile_kind_temps)
        self.eqn_lists = new_lists
        self._hard_splits = new_hard_splits
        self._soft_split_retainment_strategies = new_strategies
        self._refinement = refined
        self._refinement_snapshots = snapshots

        return refined

    def _check_pieces(self, pieces_by_source: Sequence[Sequence[EqnListPiece]]) -> None:
        """Raise unless `pieces_by_source` satisfies the contract of `refine` (see `EqnListPiece`)."""
        if len(pieces_by_source) != len(self.eqn_lists):
            raise IntermediateException(f"refine() expects pieces for each of the {len(self.eqn_lists)} EqnLists, got {len(pieces_by_source)}.")

        for src_idx, (src, pieces) in enumerate(zip(self.eqn_lists, pieces_by_source)):
            if len(pieces) == 0:
                raise IntermediateException(f"refine(): EqnList {src_idx} must have at least one piece.")
            if pieces[0].boundary is not None:
                raise IntermediateException(f"refine(): The first piece of EqnList {src_idx} inherits its boundary and must not specify one.")
            covered: set[Symbol] = set()
            for piece_idx, piece in enumerate(pieces):
                if piece_idx > 0 and piece.boundary is None:
                    raise IntermediateException(f"refine(): Piece {piece_idx} of EqnList {src_idx} needs a boundary.")
                if len(set(piece.lhses)) != len(piece.lhses):
                    raise IntermediateException(f"refine(): Piece {piece_idx} of EqnList {src_idx} lists a LHS more than once.")
                for lhs in piece.lhses:
                    if lhs not in src.eqns:
                        raise IntermediateException(f"refine(): '{lhs}' in piece {piece_idx} is not an equation of EqnList {src_idx}.")
                    if lhs in covered and lhs not in src.synthetic_symbols:
                        raise IntermediateException(f"refine(): '{lhs}' appears in several pieces of EqnList {src_idx}, but only synthetic temporaries may be duplicated.")
                    covered.add(lhs)
            if len(missing := [lhs for lhs in src.eqns if lhs not in covered and lhs not in src.synthetic_symbols]) > 0:
                raise IntermediateException(f"refine(): The pieces of EqnList {src_idx} do not cover {sorted(missing, key=str)}.")

    def _flatten_pieces(self,
                        pieces_by_source: Sequence[Sequence[EqnListPiece]],
                        inherited_annotations: Sequence[Optional[str]]) -> list[RefinedList]:
        """The lists `refine` produces: the pieces in order, with empty pieces dropped as `refine` describes."""
        candidates: list[RefinedList] = list()
        for src_idx, pieces in enumerate(pieces_by_source):
            for piece_idx, piece in enumerate(pieces):
                if piece_idx > 0:
                    candidates.append(RefinedList(src_idx, False, piece.boundary, piece.lhses))
                elif src_idx > 0:
                    inherited = SplitBoundary(soft=src_idx not in self._hard_splits,
                                              retainment_strategy=self._soft_split_retainment_strategies.get(src_idx, None),
                                              annotation=inherited_annotations[src_idx] if len(inherited_annotations) > 0 else None)
                    candidates.append(RefinedList(src_idx, True, inherited, piece.lhses))
                else:
                    candidates.append(RefinedList(src_idx, True, None, piece.lhses,
                                                  first_annotation=inherited_annotations[0] if len(inherited_annotations) > 0 else None))

        refined: list[RefinedList] = list()
        dropped: Optional[RefinedList] = None  # The last dropped empty piece, with every earlier one folded into it
        for cand in candidates:
            if dropped is not None:
                cand = dropped.fold_into(cand)
                dropped = None

            if len(cand.piece_order) > 0:
                refined.append(cand)
            elif cand.boundary is not None:
                dropped = cand

        if len(refined) == 0:
            refined.append(RefinedList(0, True, None, tuple()))

        # The first list of the complex has no boundary, even if it now starts with a later source list; it keeps that
        # boundary's annotation.
        refined[0] = replace(refined[0], boundary=None, first_annotation=refined[0].annotation)
        return refined

    def _rebuild_tile_sets(self, new_lists: list['EqnList'], tile_kind_temps: Collection[Symbol]) -> set[Symbol]:
        """
        Mark each of `tile_kind_temps` and the complex's existing tile temps as uninitialized in the new list that
        writes it and preinitialized in the new lists that read it. A temp no longer read outside its writer becomes a
        temporary of its writer. Returns the complex's new set of tile temps.
        """
        new_tile_temporaries: set[Symbol] = OrderedSet(self._tile_temporaries)
        for temp in OrderedSet(chain(tile_kind_temps, self._tile_temporaries)):
            writers = [idx for idx, el in enumerate(new_lists) if temp in el.eqns]
            readers = [idx for idx, el in enumerate(new_lists)
                       if temp not in el.eqns and any(temp in free_symbols(rhs) for rhs in el.eqns.values())]

            if len(writers) == 0:
                if len(readers) > 0:
                    raise IntermediateException(f"refine(): Tile temporary '{temp}' is read in EqnLists {readers}, but no EqnList writes it.")
                new_tile_temporaries.discard(temp)
                continue

            if len(readers) == 0:
                new_tile_temporaries.discard(temp)
                for idx in writers:
                    new_lists[idx].temporaries.add(temp)
                continue

            if len(writers) > 1:
                raise IntermediateException(f"refine(): Tile temporary '{temp}' is written in several EqnLists: {writers}.")
            [writer] = writers
            if any(reader < writer for reader in readers):
                raise IntermediateException(f"refine(): Tile temporary '{temp}' is written in EqnList {writer} after it is read in EqnLists {readers}.")

            new_tile_temporaries.add(temp)
            new_lists[writer].uninitialized_tile_temporaries.add(temp)
            for idx in readers:
                new_lists[idx].preinitialized_tile_temporaries.add(temp)

        return new_tile_temporaries

    def rebake_refined(self, *, force_fast: bool = False) -> None:
        """
        Bake every EqnList produced by the most recent `refine` with an order-preserving ordering function over its
        piece order in place of its own `ordering_fn` (order_builder still repairs dependencies). Afterward, validate
        the refinement and raise a DslException if it changed the meaning of the complex.
        """
        if self._refinement is None:
            raise IntermediateException("rebake_refined() called without a preceding refine().")
        assert len(self._refinement) == len(self.eqn_lists)

        try:
            for el_idx, (eqn_list, refined) in enumerate(zip(self.eqn_lists, self._refinement)):
                pprint(f"Rebaking loop {el_idx} after refinement...")
                eqn_list.bake(force_rebake=True, force_fast=force_fast, ordering_fn_override=fixed_order(refined.piece_order))

            self._validate_refinement()
        finally:
            # A refinement is rebaked once; a second call without a new refine() must fail loudly.
            self._refinement = None
            self._refinement_snapshots = list()

    def _validate_refinement(self) -> None:
        """
        Check that the most recent refinement did not change the meaning of the complex. Only lists cut from source
        lists that had already been baked (the pre- and post-population loci) are checked: early cuts are made before
        the first bake, like manual splits, so there is nothing to compare them with, and they get no analytic seed
        either, which keeps them identical to the splits add_eqn used to make.
        """
        assert self._refinement is not None
        refinement = self._refinement
        snapshots = self._refinement_snapshots

        # Outputs whose inferred write region changed.
        for el_idx, (eqn_list, refined) in enumerate(zip(self.eqn_lists, refinement)):
            if (snapshot := snapshots[refined.source_idx]) is None:
                continue

            where = f"loop {el_idx} (cut from loop {refined.source_idx})"
            # refine() carries over every source param a piece reads, so a missing one is a bug in refine().
            assert all(p in eqn_list.params for rhs in eqn_list.eqns.values() for p in free_symbols(rhs) if p in snapshot.params), \
                f"After splitting, {where} reads parameters which were not carried over to it."

            for lhs in eqn_list.outputs:
                if lhs not in snapshot.outputs or lhs not in snapshot.write_decls or lhs not in eqn_list.write_decls:
                    continue
                if (old_region := snapshot.write_decls[lhs]) is not (new_region := eqn_list.write_decls[lhs]):
                    raise DslException(f"After splitting, the inferred write region of '{lhs}' in {where} changed from {old_region.name} to {new_region.name}.")

        self._validate_overwrites()

        # A temporary that now crosses a new cut becomes a tile temp, which is only valid where the loop writing it
        # runs (and where the tile temps that loop read were valid). A loop that writes grid variables must not read
        # a tile temp outside the region where it is valid.
        #
        # This check is a heuristic. It predicts the loop regions the generators will infer (see infer_loop_region)
        # before the complex's variables are classified, and it relaxes the strict rule "the writer's region contains
        # the reader's" in two ways: loops that write no grid variables are not checked themselves (they only narrow
        # the region where what they compute is valid), and only crossings the refinement created are checked. The
        # strict rule rejects legitimate pull-out chains: a loop that writes only tile temps may run over a larger
        # region than the loop computing what it reads, which is harmless as long as no loop that writes grid
        # variables uses the values it computes outside the smaller region.
        writer_of: dict[Symbol, int] = dict()
        for el_idx, eqn_list in enumerate(self.eqn_lists):
            for lhs in eqn_list.eqns:
                writer_of.setdefault(lhs, el_idx)

        all_crossing: set[Symbol] = set()
        new_crossing_reads: list[set[Symbol]] = [set() for _ in self.eqn_lists]
        for el_idx, eqn_list in enumerate(self.eqn_lists):
            for rhs in eqn_list.eqns.values():
                for sym in free_symbols(rhs):
                    if sym in eqn_list.eqns or (writer := writer_of.get(sym)) is None:
                        continue
                    all_crossing.add(sym)
                    if refinement[writer].source_idx != refinement[el_idx].source_idx:
                        continue  # This crossing existed before refinement
                    if (snapshot := snapshots[refinement[writer].source_idx]) is None or sym in snapshot.outputs:
                        continue
                    new_crossing_reads[el_idx].add(sym)

        if not any(new_crossing_reads):
            return

        everywhere = frozenset({IntentRegion.Interior, IntentRegion.Boundary})

        def points(region: Optional[IntentRegion]) -> Optional[frozenset[IntentRegion]]:
            if region is None:
                return None
            return everywhere if region is IntentRegion.Everywhere else frozenset({region})

        def describe(pts: frozenset[IntentRegion]) -> str:
            return IntentRegion.Everywhere.name if pts == everywhere else ", ".join(sorted(r.name for r in pts)) or "nothing"

        valid: list[Optional[frozenset[IntentRegion]]] = list()
        for el_idx, eqn_list in enumerate(self.eqn_lists):
            def is_grid_var(sym: Symbol, el: EqnList = eqn_list) -> bool:
                return (may_be_grid_variable(sym)
                        and sym not in all_crossing
                        and sym not in el.params
                        and sym not in el.temporaries
                        and sym not in el.tile_temporaries)

            inferred = infer_loop_region(eqn_list, is_grid_var)
            loop_points = points(inferred.region)

            for temp in sorted(new_crossing_reads[el_idx], key=str):
                temp_points = valid[writer_of[temp]]
                if loop_points is None or temp_points is None:
                    continue
                if inferred.writes_grid_vars and not loop_points <= temp_points:
                    raise DslException(
                        f"After splitting, loop {el_idx}, which runs over {describe(loop_points)}, reads temporary "
                        f"'{temp}', which loop {writer_of[temp]} only computes over {describe(temp_points)}."
                    )
                loop_points = loop_points & temp_points

            valid.append(loop_points)

    def _validate_overwrites(self) -> None:
        """
        Raise a DslException if a list reads X (or an earlier version of it) after an earlier list cut from the same
        source list wrote X' (or a later version): that loop would read the overwritten value. Auto splits never cut
        there (see DslFunctionFrontend._cut_hazards), so this is a backstop. As in the tile temp check in `_validate_refinement`, only
        lists cut from source lists that had already been baked are checked, and reads across different source lists
        come from manual splits, which remain the author's responsibility.
        """
        assert self._refinement is not None
        refinement, snapshots = self._refinement, self._refinement_snapshots
        if not any("'" in str(lhs) for eqn_list in self.eqn_lists for lhs in eqn_list.eqns):
            return

        # (source list, base name) -> (the latest version written so far, the list writing it)
        overwritten: dict[tuple[int, str], tuple[int, int]] = dict()
        for el_idx, (eqn_list, refined) in enumerate(zip(self.eqn_lists, refinement)):
            if snapshots[refined.source_idx] is None:
                continue
            for sym in sorted(set(chain(*(free_symbols(rhs) for rhs in eqn_list.eqns.values()))), key=str):
                base, version = overwrite_version(sym)
                if (written := overwritten.get((refined.source_idx, base))) is not None and version < written[0]:
                    raise DslException(
                        f"After splitting, loop {el_idx} reads '{sym}', which loop {written[1]} (cut from the same loop "
                        f"{refined.source_idx}) has already overwritten."
                    )
            for lhs in eqn_list.eqns:
                base, version = overwrite_version(lhs)
                if version > overwritten.get((refined.source_idx, base), (0, -1))[0]:
                    overwritten[(refined.source_idx, base)] = (version, el_idx)

    @cache
    def _calc_tile_temps(self) -> None:
        self._derived_state_computed = True
        # Don't clear out self._tile_temporaries because it will already be populated by global_cse

        for temp in self.temporaries:
            written_el: Optional[int] = None
            read_els: set[int] = set()

            for el_idx, eqn_list in enumerate(self.eqn_lists):
                if temp in eqn_list.eqns:
                    written_el = el_idx
                    continue

                for lhs, rhs in eqn_list.eqns.items():
                    if temp in free_symbols(rhs):
                        read_els.add(el_idx)
                        break

            if written_el is not None and len(read_els) > 0:
                # Auto splits never cut there (see DslFunctionFrontend._cut_hazards); a manual split_loop() can.
                if any(read_el < written_el for read_el in read_els):
                    raise DslException(
                        f"'{temp}' is written in loop {written_el} after it is read in loop(s) {sorted(read_els)}: a "
                        f"loop cannot read a value that a later loop computes, so no split (e.g., split_loop()) may "
                        f"separate the read from the equation that writes '{temp}'."
                    )

                self._tile_temporaries.add(temp)
                self.eqn_lists[written_el].uninitialized_tile_temporaries.add(temp)
                for el_idx in read_els:
                    self.eqn_lists[el_idx].preinitialized_tile_temporaries.add(temp)

    @cache
    def _calc_vars(self) -> None:
        self._derived_state_computed = True
        self._inputs = OrderedSet()
        self._outputs = OrderedSet()
        self._params = OrderedSet()
        self._temporaries = OrderedSet()
        self._variables = OrderedSet()

        for eqn_list in self.eqn_lists:
            self._inputs |= eqn_list.inputs
            self._outputs |= eqn_list.outputs
            self._params |= eqn_list.params
            self._temporaries |= eqn_list.temporaries
            self._variables |= eqn_list.variables

        self._temporaries.update(self._inputs.intersection(self._outputs))
        self._inputs.difference_update(self._temporaries)
        self._outputs.difference_update(self._temporaries)

    @cache
    def _calc_decls(self) -> None:
        self._derived_state_computed = True
        self._read_decls = OrderedDict()
        self._write_decls = OrderedDict()

        for eqn_list in self.eqn_lists:
            consolidate(self._read_decls, eqn_list.read_decls, lambda r1, r2: r1.consolidate(r2))
            consolidate(self._write_decls, eqn_list.write_decls, lambda r1, r2: r1.consolidate(r2))

        for t in self.temporaries:
            if t in self._read_decls:
                del self._read_decls[t]
            if t in self._write_decls:
                del self._write_decls[t]

    @property
    @require_baked(msg="Can't get tile_temporaries before baking the EqnComplex.")
    def tile_temporaries(self) -> set[Symbol]:
        assert hasattr(self, '_tile_temporaries')
        self._calc_tile_temps()
        return self._tile_temporaries

    @property
    @require_baked(msg="Can't get inputs before baking the EqnComplex.")
    def inputs(self) -> set[Symbol]:
        self._calc_vars()
        return self._inputs

    @property
    @require_baked(msg="Can't get outputs before baking the EqnComplex.")
    def outputs(self) -> set[Symbol]:
        self._calc_vars()
        return self._outputs

    @property
    @require_baked(msg="Can't get params before baking the EqnComplex.")
    def params(self) -> set[Symbol]:
        self._calc_vars()
        return self._params

    @property
    @require_baked(msg="Can't get temporaries before baking the EqnComplex.")
    def temporaries(self) -> set[Symbol]:
        self._calc_vars()
        return self._temporaries

    @property
    @require_baked(msg="Can't get read_decls before baking the EqnComplex.")
    def read_decls(self) -> dict[Symbol, IntentRegion]:
        self._calc_decls()
        return self._read_decls

    @property
    @require_baked(msg="Can't get write_decls before baking the EqnComplex.")
    def write_decls(self) -> dict[Symbol, IntentRegion]:
        self._calc_decls()
        return self._write_decls

    @property
    @require_baked(msg="Can't get variables before baking the EqnComplex.")
    def variables(self) -> set[Symbol]:
        self._calc_decls()
        return self._variables

    @cached_property
    @require_baked(msg="Can't get stencil_limits before baking the EqnComplex.")
    def stencil_limits(self) -> tuple[int, int, int]:
        result = [0, 0, 0]

        for eqn_list in self.eqn_lists:
            for eqn_rhs in eqn_list.eqns.values():
                # noinspection PyProtectedMember
                eqn_list._stencil_limits(result, eqn_rhs)

        return result[0], result[1], result[2]

    @cached_property
    @require_baked(msg="Can't get stencil_idxes before baking the EqnComplex.")
    def stencil_idxes(self) -> set[StencilIdxWithName]:
        result: set[StencilIdxWithName] = set()

        for eqn_list in self.eqn_lists:
            for eqn_rhs in eqn_list.eqns.values():
                # noinspection PyProtectedMember
                eqn_list._stencil_idxes(result, eqn_rhs)

        return result


class EqnList:
    """
    This class models a generic list of equations. As such, it knows nothing about the rest of EinsteinEngine.
    Ultimately, the information in this class will be used to generate a loop to be output by EinsteinEngine.
    All it knows are the following things:
    (1) params - These are quantities that are generated outside the loop.
    (2) inputs - These are quantities which are read by equations but never written by them.
    (3) outputs - These are quantities which are written by equations but never read by them.
    (4) equations - These relate inputs to outputs. These may contain temporary variables, i.e.
                    quantities that are both read and written by equations.

    This class can remove equations and parameters that are not needed, but will complain
    about inputs that are not needed. It can also detect errors in the classification of
    symbols as inputs/outputs/params.
    """

    def __init__(self, parent: EqnComplex, is_stencil: Dict[UFunc, bool]) -> None:
        self.eqns: Dict[Symbol, Expr] = dict()
        self.params: Set[Symbol] = OrderedSet()
        self.inputs: Set[Symbol] = OrderedSet()
        self.outputs: Set[Symbol] = OrderedSet()
        self.order: List[Symbol] = list()
        self.read_decls: Dict[Symbol, IntentRegion] = OrderedDict()
        self.write_decls: Dict[Symbol, IntentRegion] = OrderedDict()
        # TODO: need a better default
        self.default_read_write_spec: IntentRegion = IntentRegion.Everywhere  # Interior
        self.is_stencil: Dict[UFunc, bool] = is_stencil
        self.temporaries: Set[Symbol] = OrderedSet()
        self.uninitialized_tile_temporaries: Set[Symbol] = OrderedSet()
        self.preinitialized_tile_temporaries: Set[Symbol] = OrderedSet()
        self.temporary_replacements: Set[TemporaryReplacement] = OrderedSet()
        self.provides: Dict[Symbol, Set[Symbol]] = dict()  # vals require key
        self.requires: Dict[Symbol, Set[Symbol]] = dict()  # key requires vals
        self.been_baked: bool = False
        self.parent = parent
        self.complexity: dict[Symbol, int] = dict()
        self.ordering_fn: EqnOrderingFn = maximize_symbol_reuse
        # Bound to the owning function's annotation callbacks by EqnComplex._bind_annotation_callbacks.
        self.set_eqn_annotation: Optional[Callable[[Symbol, str], None]] = None
        self.clear_eqn_annotations: Optional[Callable[[], None]] = None
        # The function-wide recipe position of each equation. Never rewritten; temporaries are appended.
        self.eqn_recipe_order: OrderedDict[Symbol, int] = OrderedDict()
        # Starts equal to the recipe order and is rewritten by the early and pre-population ordering functions.
        # Temporaries are appended.
        self.eqn_pre_population_order: OrderedDict[Symbol, int] = OrderedDict()
        # The 0-based index of the author-level add_eqn call that created each equation. Pull-out temps inherit the
        # origin of the equation they were pulled out of; CSE temps and equations of synthetic functions have none.
        self.eqn_origin: dict[Symbol, int] = dict()
        # The params each recipe equation registered when it was added, so params can be repartitioned by refine().
        self.eqn_recorded_params: dict[Symbol, frozenset[Symbol]] = dict()
        # Symbols read (not written) by this list which are known to be analytic because the list was cut from a
        # larger one; they seed the AnalyticFunctionChecker so that cutting does not change inferred write regions.
        self.analytic_seed: Set[Symbol] = OrderedSet()
        self.synthetic_symbols: Set[Symbol] = OrderedSet()

        # The modeling system treats these special
        # symbols as parameters.
        self.add_param(DXI)
        self.add_param(DYI)
        self.add_param(DZI)

    @property
    def tile_temporaries(self) -> set[Symbol]:
        return self.uninitialized_tile_temporaries.union(self.preinitialized_tile_temporaries)

    @property
    def local_temporaries(self) -> set[Symbol]:
        return self.temporaries - self.tile_temporaries

    #@cached_property
    @property
    @require_baked(msg="Can't get variables before baking the EqnList.")
    def variables(self) -> Set[Symbol]:
        return self.inputs | self.outputs | self.temporaries

    #@cached_property
    @property
    @require_baked(msg="Can't get sorted_eqns before baking the EqnList.")
    def sorted_eqns(self) -> list[tuple[Symbol, Expr]]:
        return sorted(self.eqns.items(), key=lambda kv: self.order.index(kv[0]))

    def _grid_variables(self) -> set[Symbol]:
        return {s for s in (self.inputs | self.outputs) if str(s) not in {'t', 'x', 'y', 'z', 'DXI', 'DYI', 'DZI'}}

    #@cached_property
    @property
    @require_baked(msg="Can't get grid_variables before baking the EqnList.")
    def grid_variables(self) -> set[Symbol]:
        return self._grid_variables()

    def add_param(self, lhs: Symbol) -> None:
        assert lhs not in self.outputs, f"The symbol '{lhs}' is already in outputs"
        assert lhs not in self.inputs, f"The symbol '{lhs}' is already in outputs"
        self.params.add(lhs)

    @multimethod
    def add_input(self, lhs: Symbol) -> None:
        # TODO: Automatically assign temps?
        return
        assert lhs not in self.outputs, f"The symbol '{lhs}' is already in outputs"
        if lhs in self.outputs:
            self.temporaries.add(lhs)
        assert lhs not in self.params, f"The symbol '{lhs}' is already in outputs"
        assert isinstance(lhs, Symbol)
        self.inputs.add(lhs)

    @add_input.register
    def _(self, lhs: IndexedBase) -> None:
        self.add_input(lhs.args[0])

    @add_input.register
    def _(self, lhs: Basic) -> None:
        raise DslException("bad input")

    def add_output(self, lhs: Symbol) -> None:
        # TODO: Automatically assign temps?
        # assert lhs not in self.inputs, f"The symbol '{lhs}' is already in outputs"
        return
        if lhs in self.inputs:
            self.temporaries.add(lhs)
        assert lhs not in self.params, f"The symbol '{lhs}' is already in outputs"
        self.outputs.add(lhs)

    def add_eqn(self,
                lhs: Symbol,
                rhs: Expr,
                *,
                recipe_position: Optional[int] = None,
                origin: Optional[int] = None,
                recorded_params: Optional[frozenset[Symbol]] = None) -> None:
        """
        Add an equation. It is appended to the pre-population order and takes the next function-wide recipe position,
        unless `recipe_position` gives the one it already has (for an equation moved here from another list of the
        complex), in which case it is inserted into eqn_recipe_order so that the order stays sorted by position.
        `origin` and `recorded_params` set its eqn_origin and eqn_recorded_params entries.
        """
        if lhs in self.eqns:
            raise IntermediateException(f"Equation for '{lhs}' is already defined")

        # Ensure we only have symbols in eqnlist
        self.eqns[lhs := symbify(lhs)] = symbify(rhs)
        if recipe_position is None:
            self.eqn_recipe_order[lhs] = self.parent._take_recipe_position()
        else:
            later = [k for k, pos in self.eqn_recipe_order.items() if pos > recipe_position]
            self.eqn_recipe_order[lhs] = recipe_position
            for k in later:
                self.eqn_recipe_order.move_to_end(k)
        self.eqn_pre_population_order[lhs] = len(self.eqn_pre_population_order)
        if origin is not None:
            self.eqn_origin[lhs] = origin
        if recorded_params is not None:
            self.eqn_recorded_params[lhs] = recorded_params

    def set_pre_population_order(self, order: Iterable[Symbol]) -> None:
        """Rewrite the pre-population order. `order` must be a permutation of this list's equations."""
        new_order: OrderedDict[Symbol, int] = OrderedDict()
        for lhs in order:
            if lhs not in self.eqns or lhs in new_order:
                raise IntermediateException(f"The new pre-population order must list each equation exactly once, but '{lhs}' is unknown or repeated.")
            new_order[lhs] = len(new_order)
        if len(new_order) != len(self.eqns):
            raise IntermediateException(f"The new pre-population order is missing {sorted(set(self.eqns) - set(new_order), key=str)}.")
        self.eqn_pre_population_order = new_order

    def do_pull_out(self, name_generator: Generator[str, Never, Never]) -> None:
        new_eqns: OrderedDict[Symbol, Expr] = OrderedDict()
        new_origins: dict[Symbol, int] = dict()
        modify_eqns: OrderedDict[Symbol, Expr] = OrderedDict()

        assert self.eqns.keys() == self.eqn_pre_population_order.keys()

        for lhs in self.eqn_pre_population_order.keys():
            rhs = self.eqns[lhs]
            for sub_expr in sorted(rhs.find(pull_out), key=str):  # type: ignore[no-untyped-call]
                if len(sub_expr_args := sub_expr.args) > 1:
                    raise IntermediateException("pull_out() should have only one argument")
                new_sym = mk_symbol(next(name_generator))
                assert new_sym not in new_eqns
                new_eqns[new_sym] = sub_expr_args[0]
                if (origin := self.eqn_origin.get(lhs, None)) is not None:
                    new_origins[new_sym] = origin
                rhs = rhs.xreplace({sub_expr: new_sym})  # type: ignore[no-untyped-call]
            assert lhs not in modify_eqns
            modify_eqns[lhs] = rhs

        for lhs, rhs in new_eqns.items():
            self.add_eqn(lhs, rhs, origin=new_origins.get(lhs, None))
            self.temporaries.add(lhs)

        for lhs, rhs in modify_eqns.items():
            self.eqns[lhs] = rhs

    def _trial_copy(self, parent: EqnComplex, el_idx: int) -> 'EqnList':
        """A copy of this unbaked list, list `el_idx` of `parent` (a trial copy of this list's parent); see
        `EqnComplex.trial_copy`."""
        if self.been_baked:
            raise IntermediateException("Only an unbaked EqnList can be copied for a trial bake.")
        clone = copy.copy(self)
        for name, value in vars(self).items():
            if name not in ('parent', 'is_stencil'):
                setattr(clone, name, _copy_containers(value))
        clone.parent = parent
        parent._bind_annotation_callbacks(clone, el_idx)
        return clone

    def _populate_from_piece(self, source: 'EqnList', lhses: Sequence[Symbol], analytic_seed: set[Symbol]) -> None:
        """Fill this fresh EqnList with the equations `lhses` of `source`. See `EqnComplex.refine`."""
        lhs_set = set(lhses)

        # Keep the source's dict orders; the piece order itself is applied by EqnComplex.rebake_refined.
        for lhs, rhs in source.eqns.items():
            if lhs in lhs_set:
                self.eqns[lhs] = rhs
        self.eqn_recipe_order = OrderedDict((lhs, pos) for lhs, pos in source.eqn_recipe_order.items() if lhs in lhs_set)
        self.eqn_pre_population_order = OrderedDict(
            (lhs, pos) for pos, lhs in enumerate(lhs for lhs in source.eqn_pre_population_order if lhs in lhs_set)
        )
        self.eqn_origin = {lhs: origin for lhs, origin in source.eqn_origin.items() if lhs in lhs_set}
        self.eqn_recorded_params = {lhs: params for lhs, params in source.eqn_recorded_params.items() if lhs in lhs_set}
        self.complexity = {lhs: c for lhs, c in source.complexity.items() if lhs in lhs_set}

        read_syms: set[Symbol] = set(chain(*(free_symbols(rhs) for rhs in self.eqns.values())))
        recorded_params: set[Symbol] = set(chain(*self.eqn_recorded_params.values()))
        for param in source.params:
            if param in read_syms or param in recorded_params:
                self.add_param(param)

        used = lhs_set | read_syms
        self.synthetic_symbols.update(sym for sym in source.synthetic_symbols if sym in used)
        self.temporaries.update(sym for sym in source.temporaries if sym in lhs_set)
        self.analytic_seed.update(sym for sym in analytic_seed if sym in read_syms and sym not in lhs_set)

        self.ordering_fn = source.ordering_fn
        self.default_read_write_spec = source.default_read_write_spec

    def _analytic_checker(self) -> AnalyticFunctionChecker:
        # Seed symbols are read but not written here, so they are passed alongside the params: known to be analytic,
        # and excluded from the result.
        known_analytic: set[Symbol] = set(self.params) | {sym for sym in self.analytic_seed if sym not in self.eqns}
        return AnalyticFunctionChecker(known_analytic, self.eqns)

    def _analytic_symbols(self) -> set[Symbol]:
        """The symbols this list writes or reads that are known to be analytic."""
        return self._analytic_checker().analytic() | {sym for sym in self.analytic_seed if sym not in self.eqns}

    def recycle_temporaries(self) -> None:
        temp_reads: Dict[Symbol, OrderedSet[int]] = OrderedDict()
        temp_writes: Dict[Symbol, OrderedSet[int]] = OrderedDict()

        local_temporaries = self.temporaries - self.parent.tile_temporaries

        for lhs, rhs in self.eqns.items():
            eqn_i = self.order.index(lhs)

            if lhs in local_temporaries:
                get_or_compute(temp_writes, lhs, lambda _: OrderedSet()).add(eqn_i)

            if len(temps_read := free_symbols(rhs).intersection(local_temporaries)) > 0:
                temp_var: Symbol
                for temp_var in temps_read:
                    get_or_compute(temp_reads, temp_var, lambda _: OrderedSet()).add(eqn_i)

        lifetimes: Set[TemporaryLifetime] = OrderedSet()

        for temp_var in local_temporaries:
            vprint(f'Temporary {temp_var}:')
            assert len(temp_writes[temp_var]) == 1

            reads_str = [str(x) for x in temp_reads[temp_var]]
            writes_str = [str(x) for x in temp_writes[temp_var]]

            vprint(f'    Read in EQNs: {", ".join(reads_str)}')
            vprint(f'    Written in EQNs: {", ".join(writes_str)}')

            lifetimes.add(TemporaryLifetime(
                symbol=temp_var,
                prime=0,
                read_at=temp_reads[temp_var],
                written_at=temp_writes[temp_var].pop(),
                replaces=None,
                is_superseded=False,
                is_dead=False
            ))

        lifetimes_assigned_at = {lt.written_at: lt for lt in lifetimes}
        lifetimes_final_read: SortedDict[int, OrderedSet[TemporaryLifetime]] = SortedDict()
        for lt in lifetimes:
            if lt.final_read in lifetimes_final_read:
                lifetimes_final_read[lt.final_read].add(lt)
            else:
                lifetimes_final_read[lt.final_read] = OrderedSet([lt])
        lifetimes_final_read_keys = list(lifetimes_final_read.keys())

        # Attempt to find a temporary lifetime that is stale (last read was before eqn_idx), not superseded, and not dead.
        def find_candidate(eqn_idx: int) -> Optional[TemporaryLifetime]:
            eqn_probe = eqn_idx
            while eqn_probe > 0:
                # In the sorted list of keys, find the index to insert `eqn_probe`. This will either give us the
                # index of `eqn_probe` itself if it's a valid key, or the index of the smallest key which is GT it.
                # If we get 0 back, we are either the first key in the list or smaller than all valid keys, so abort.
                if (key_idx := lifetimes_final_read.bisect_left(eqn_probe)) == 0:
                    return None

                # Subtract one from `key_idx` to get the next-smallest valid key.
                # Now, `eqn_probe` holds the next-smallest valid key from its previous value.
                eqn_probe = lifetimes_final_read_keys[key_idx - 1]

                assert eqn_probe < eqn_idx
                assert eqn_probe in lifetimes_final_read

                # Inspect the lifetimes which expired in eqn number `eqn_probe`. If we find a live one, return it.
                lt: TemporaryLifetime
                for lt in lifetimes_final_read[eqn_probe]:
                    if not lt.is_superseded and not lt.is_dead:
                        return lt

            return None



        for eqn_i in range(len(self.order)):
            if not (assigned_here := lifetimes_assigned_at.get(eqn_i, None)):
                continue

            if not (candidate := find_candidate(eqn_i)):
                continue

            lifetimes.add(TemporaryLifetime(
                symbol=candidate.symbol,
                prime=candidate.prime + 1,
                read_at=assigned_here.read_at,
                written_at=eqn_i,
                replaces=assigned_here,
                is_superseded=False,
                is_dead=False
            ))

            assigned_here.is_dead = True
            candidate.is_superseded = True

            self.temporary_replacements.add(TemporaryReplacement(
                old=assigned_here.symbol,
                new=candidate.symbol,
                begin_eqn=eqn_i,
                end_eqn=assigned_here.final_read
            ))

            vprint(f'Will replace the declaration of {assigned_here.symbol} with reassignment to {candidate.symbol} in equation {eqn_i}.')

        vprint("*** Dumping temporary lifetimes ***")
        for lifetime in filter(lambda lt: not lt.is_dead, sorted(lifetimes, key=lambda lt: (str(lt.symbol), lt.prime))):
            vprint(f'{lifetime} [{lifetime.written_at}, {max(lifetime.read_at)}]')

    def uses_dict(self) -> Dict[Symbol, int]:
        uses: Dict[Symbol, int] = dict()
        for k, v in self.eqns.items():
            for k2 in free_symbols(v):
                old = uses.get(k2, 0)
                uses[k2] = old + 1
        return uses

    def apply_order(self, k: Symbol, provides: Dict[Symbol, Set[Symbol]], requires: Dict[Symbol, Set[Symbol]]) -> List[Symbol]:
        result = list()
        if k not in self.params and k not in self.inputs and k not in self.preinitialized_tile_temporaries:
            self.order.append(k)
        for v in provides.get(k, set()):
            req = requires[v]
            if k in req:
                req.remove(k)
            if len(req) == 0:
                result.append(v)
        return result

    def order_builder(self,
                      complete: Dict[Symbol, int],
                      override_ordering_fn: Optional[EqnOrderingFn] = None) -> None:
        TOTAL_ORDER = True  # todo: expose this as a bake option

        for k in self.inputs:
            complete[k] = 0
        for k in self.params:
            complete[k] = 0

        ordering_fn = override_ordering_fn or self.ordering_fn
        set_eqn_annotation = self.set_eqn_annotation
        myself = self

        if respects_dependency_order(ordering_fn):
            order: list[Symbol] = list()

            for sym in ordering_fn(self.eqns, self):
                if isinstance(sym, tuple):
                    order.append(sym[0])
                    complete[sym[0]] = len(order)
                    if set_eqn_annotation:
                        set_eqn_annotation(*sym)
                else:
                    order.append(sym)
                    complete[sym] = len(order)

            self.order = order
            return


        if TOTAL_ORDER:
            total_order: list[Symbol | tuple[Symbol, str]] = list(ordering_fn(self.eqns, self))
            total_order_symbols: list[Symbol] = [(sym[0] if isinstance(sym, tuple) else sym) for sym in total_order]
            total_order_annotations: dict[Symbol, str] = {t[0]: t[1] for t in total_order if isinstance(t, tuple)}

        class Ord:
            def __init__(self, eqns: dict[Symbol, Expr]) -> None:
                self.ord: list[Symbol] = list()
                self.eqns = eqns

            def add(self, sym: Symbol) -> bool:
                if sym in complete:
                    return False

                if TOTAL_ORDER:
                    for s_dep in sorted(
                            (dep for dep in free_symbols(self.eqns[sym]) if dep in self.eqns),
                            key=lambda dep: total_order_symbols.index(dep) if dep in total_order_symbols else len(total_order_symbols)
                    ):
                        self.add(s_dep)
                        if set_eqn_annotation and s_dep in total_order_annotations:
                            set_eqn_annotation(s_dep, f'Dependency! {total_order_annotations[s_dep]}')
                else:
                    for dep in ordering_fn({dep: self.eqns[dep] for dep in free_symbols(self.eqns[sym]) if dep in self.eqns}, myself):
                        if isinstance(dep, tuple):
                            self.add(dep[0])
                            if set_eqn_annotation:
                                set_eqn_annotation(dep[0], f'Dependency! {dep[1]}')
                        else:
                            self.add(dep)
                self.ord.append(sym)
                complete[sym] = len(self.ord)

                return True

        ord = Ord(self.eqns)

        order_it: Iterable[Symbol | tuple[Symbol, str]]
        if TOTAL_ORDER:
            order_it = total_order
        else:
            order_it = ordering_fn(self.eqns, self)

        for sym in order_it:
            if isinstance(sym, tuple):
                if ord.add(sym[0]) and set_eqn_annotation:
                    set_eqn_annotation(*sym)
            else:
                ord.add(sym)

        self.order = ord.ord

    def _run_preliminary_complexity_analysis(self) -> None:
        grid_vars = self._grid_variables()
        complexity_visitor = SympyComplexityVisitor(lambda s: s in grid_vars)
        for lhs, rhs in self.eqns.items():
            self.complexity[lhs] = complexity_visitor.complexity(rhs)

    def _run_main_complexity_analysis(self) -> None:
        complexity_visitor = SympyComplexityVisitor(lambda s: s in self._grid_variables())
        for lhs, rhs in self.eqns.items():
            self.complexity[lhs] = complexity_visitor.complexity(rhs)

    def _run_complexity_analysis(self, *lhses: Symbol) -> None:
        complexity_visitor = SympyComplexityVisitor(lambda s: s in self._grid_variables())
        for lhs in lhses:
            self.complexity[lhs] = complexity_visitor.complexity(self.eqns[lhs])

    def bake(self,
             *,
             force_rebake: bool = False,
             force_fast: bool = False,
             ordering_fn_override: Optional[EqnOrderingFn] = None) -> None:
        """
        Discover inconsistencies and errors in the param/input/output/equation sets.
        `ordering_fn_override` orders the equations for this bake only, in place of `ordering_fn`.
        """
        if self.been_baked and not force_rebake:
            raise DslException("Can't bake an EqnList that has already been baked.")
        self.been_baked = True

        # Annotations are recomputed by every bake; stale ones from an earlier order must not survive.
        if self.clear_eqn_annotations:
            self.clear_eqn_annotations()

        rd_overwrites: OrderedSet[Symbol] = OrderedSet()
        wr_overwrites: OrderedSet[Symbol] = OrderedSet()
        def process_overwrite(s: Symbol) -> None:
            if "'" in (ss := str(s)):
                rd = mk_symbol(ss.replace("'", ""))
                wr = s
                rd_overwrites.add(rd)
                wr_overwrites.add(wr)

        # Bake now regenerates inputs and outputs but not parameters
        self.inputs.clear()
        self.outputs.clear()
        self.temporaries.clear()
        for lhs, rhs in self.eqns.items():
            assert lhs not in self.params, f"Symbol '{lhs}' is a parameter, but we are assigning to it."
            self.outputs.add(lhs)
            process_overwrite(lhs)
            for symb in rhs.free_symbols:
                if symb not in self.params:
                    assert isinstance(symb, Symbol), f"{symb} should be an instance of Symbol, but type={type(symb)}"
                    self.inputs.add(symb)
                    process_overwrite(symb)

        for lhs in self.outputs:
            if lhs in self.inputs:
                self.temporaries.add(lhs)
        for lhs in self.temporaries:
            self.inputs.remove(lhs)
            self.outputs.remove(lhs)

        for rd in rd_overwrites:
            if rd in self.outputs:
                raise DslException(f"Overwrite source symbol {rd} should not be in outputs")
            if rd in self.temporaries:
                raise DslException(f"Overwrite source symbol {rd} should not be in temporaries")

        for rhs in self.eqns.values():
            for call in rhs.find(lambda e: hasattr(e, "func") and self.is_stencil.get(e.func, False)):  # type: ignore[no-untyped-call]
                if len(call.args) > 0 and call.args[0] in rd_overwrites:
                    raise DslException(f"Overwrite source symbol {call.args[0]} cannot be used inside a stencil")

        for wr in wr_overwrites:
            if wr in self.inputs:
                raise DslException(f"Overwrite destination symbol {wr} should not be in inputs")
            if wr in self.temporaries:
                raise DslException(f"Overwrite destination symbol {wr} should not be in temporaries")

        needed: Set[Symbol] = OrderedSet()
        complete: Dict[Symbol, int] = OrderedDict()
        self.order = list()

        read: Set[Symbol] = OrderedSet()
        written: Set[Symbol] = OrderedSet()

        for temp in self.temporaries:
            if temp in self.outputs:
                self.outputs.remove(temp)
            if temp in self.inputs:
                self.inputs.remove(temp)

        self.read_decls.clear()
        self.write_decls.clear()

        override_e2e = self.parent.intent_override is IntentOverride.E2E
        override_2i = self.parent.intent_override is IntentOverride.WriteInterior

        # Figure out the read/writes
        for lhs in self.inputs:
            self.read_decls[lhs] = IntentRegion.Everywhere if override_e2e else IntentRegion.Interior
        for lhs in self.outputs:
            self.write_decls[lhs] = IntentRegion.Everywhere if override_e2e else IntentRegion.Interior

        for lhs, rhs in self.eqns.items():
            for sten in rhs.find(stencil):  # type: ignore[no-untyped-call]
                if sten.args[1] != 0 or sten.args[2] != 0 or sten.args[3] != 0:
                    if override_e2e:
                        raise DslException(f"Stencil '{sten}' found in the RHS for {lhs} cannot have nonzero offset in E2E mode.")
                    var = sten.args[0]
                    self.read_decls[var] = IntentRegion.Everywhere


        if not override_2i:
            checker = self._analytic_checker()
            for lhs in checker.analytic():
                if lhs in self.outputs:
                    self.write_decls[lhs] = IntentRegion.Everywhere

        vprint(colored("Inputs:", "green"), self.inputs)
        vprint(colored("Outputs:", "green"), self.outputs)
        vprint(colored("Params:", "green"), self.params)

        for k in self.eqns:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            written.add(k)
            for q in free_symbols(self.eqns[k]):
                read.add(q)

        vprint(colored("Read:", "green"), read)
        vprint(colored("Written:", "green"), written)

        for k in self.inputs:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            # With loop splitting, it can arise that an input symbol ends up in the RHS of a tile temp assigned
            #  in the previous loop, so we can just quietly fix the inconsistency.
            if k not in read:
                self.inputs.remove(k)
            assert k not in written, f"Symbol '{k}' is in inputs, but it is assigned to."

        for arg in self.inputs:
            assert isinstance(arg, Symbol), f"{arg}, type={type(arg)}"

        for k in self.outputs:
            assert isinstance(k, Symbol)
            assert k in written, f"Symbol '{k}' is in outputs, but it is never written"

        for k in written:
            assert isinstance(k, Symbol)
            if (k not in self.outputs
                    and k not in self.uninitialized_tile_temporaries
                    and k not in self.preinitialized_tile_temporaries):
                self.temporaries.add(k)

        for k in read:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            if (k not in self.inputs
                    and k not in self.params
                    and k not in self.uninitialized_tile_temporaries
                    and k not in self.preinitialized_tile_temporaries):
                self.temporaries.add(k)

        vprint(colored("Temps:", "green"), self.temporaries)
        vprint(colored("Uninitialized Tile Temps:", "green"), self.uninitialized_tile_temporaries)
        vprint(colored("Preinitialized Tile Temps:", "green"), self.preinitialized_tile_temporaries)

        class FindBad:
            def __init__(self, outer: EqnList) -> None:
                self.outer = outer
                self.msg: Optional[str] = None

            def m(self, expr: Expr) -> bool:
                if expr.is_Function:
                    if self.outer.is_stencil.get(expr.func, False):
                        for arg in expr.args:
                            if arg in self.outer.temporaries:
                                self.msg = f"Temporary passed to stencil: call='{expr}' arg='{arg}'"
                            break  # only check the first arg
                return False

            def exc(self) -> None:
                if self.msg is not None:
                    raise Exception(self.msg)

            def r(self, expr: Expr) -> Expr:
                return expr

        fb = FindBad(self)
        for eqn in self.eqns.items():
            do_replace(eqn[1], fb.m, fb.r)
            fb.exc()

        self._run_main_complexity_analysis()

        # Simple stopgap to prevent wasteful bayesian optimization calls before CSE
        if ordering_fn_override is None and (not force_rebake or force_fast):
            ordering_fn_override = pre_cse_stand_in(self.ordering_fn)

        self.order_builder(complete, override_ordering_fn=ordering_fn_override)

        vprint(colored("Order:", "green"), self.order)

        try:
            memory_pressure = score_memory_pressure(self.eqns, self.order)
            vprint(colored("Memory Pressure:", "magenta"))
            vprint(f"  Total: {sorted(memory_pressure.items(), key=lambda kv: kv[1], reverse=True)}")
            vprint(f"  Mean: {mean(memory_pressure.values())}")
            vprint(f"  Median: {median(memory_pressure.values())}")
            vprint(f"  Max: {max(memory_pressure.items(), key=lambda kv: kv[1])}")
        except:
            pass

        for k in self.temporaries:
            assert k in read, f"Temporary variable '{k}' is never read"
            assert k in written, f"Temporary variable '{k}' is never written"
            # assert k not in self.outputs, f"Temporary variable '{k}' in outputs"
            assert k not in self.inputs, f"Temporary variable '{k}' in inputs"

        for k in read:
            assert k in self.inputs or self.params or self.temporaries, f"Symbol '{k}' is read, but it is not a temp, parameter, or input."

        vprint(colored("READS:", "green"), end="")
        for var, spec in self.read_decls.items():
            if var in self.inputs:
                vprint(" ", var, "=", colored(repr(spec), "yellow"), sep="", end="")
        vprint()
        vprint(colored("WRITES:", "green"), end="")
        for var, spec in self.write_decls.items():
            if var in self.outputs:
                vprint(" ", var, "=", colored(repr(spec), "yellow"), sep="", end="")
        vprint()

        for k, v in self.eqns.items():
            assert k in complete, f"Eqn '{k} = {v}' does not contribute to the output."
            val1: int = complete[k]
            for k2 in free_symbols(v):
                val2: Optional[int] = complete.get(k2, None)
                assert val2 is not None, f"k2={k2}"
                assert val1 >= val2, f"Symbol '{k}' is part of an assignment cycle."
        for k in needed:
            if k not in complete:
                print(f"Symbol '{k}' needed but could not be evaluated. Cycle in assignment?")
        for k in self.inputs:
            assert k in complete, f"Symbol '{k}' appears in inputs but is not complete"
        for k in self.eqns:
            assert k in complete, f"Equation '{k} = {self.eqns[k]}' is never complete"

        for lhs in self.eqns:
            assert isinstance(lhs, Symbol), f"{lhs}, type={type(lhs)}"
            rhs = self.eqns[lhs]
            vprint(colored("EQN:", "cyan"), lhs, colored("=", "cyan"), rhs, " ", colored(f"[complexity = {self.complexity[lhs]}]", "magenta"))

    def trim(self) -> None:
        """ Remove temporaries of the form "a=b". They are clutter. """
        subs: Dict[Symbol, Symbol] = dict()
        for k, v in self.eqns.items():
            if v.is_symbol:
                # k is not not needed
                subs[k] = cast(Symbol, v)
                wprint(f"Equation '{k} = {v}' can be trivially eliminated")

        new_eqns: Dict[Symbol, Expr] = dict()
        for k in self.eqns:
            if k not in subs:
                v = self.eqns[k]
                v2 = do_subs(v, subs)
                new_eqns[k] = v2

        self.eqns = new_eqns

    def madd(self) -> None:
        """ Insert fused multiply add instructions """
        p0 = mk_wild("p0", exclude=[0, 1, 2, -1, -2])
        p1 = mk_wild("p1", exclude=[0, 1, 2, -1, -2])
        p2 = mk_wild("p2", exclude=[0])

        class make_madd:
            def __init__(self) -> None:
                self.value: Optional[Expr] = None

            def m(self, expr: Expr) -> bool:
                self.value = None
                g = do_match(expr, p0 * p1 + p2)
                if g:
                    q0, q1, q2 = g[p0], g[p1], g[p2]
                    self.value = muladd(self.repl(q0), self.repl(q1), self.repl(q2))
                return self.value is not None

            def r(self, expr: Expr) -> Expr:
                assert self.value is not None
                return self.value

            def repl(self, expr: Expr) -> Expr:
                for iter in range(20):
                    nexpr = do_replace(expr, self.m, self.r)
                    if nexpr == expr:
                        return nexpr
                    expr = nexpr
                return expr

        mm = make_madd()
        for k, v in self.eqns.items():
            self.eqns[k] = mm.repl(v)

    def stencil_limits(self) -> typing.Tuple[int, int, int]:
        result = [0, 0, 0]
        for eqn in self.eqns.values():
            self._stencil_limits(result, eqn)
        return result[0], result[1], result[2]

    def _stencil_limits(self, result: List[int], expr: Expr) -> None:
        def extract(arg: Basic) -> None:
            for i in range(3):
                ivar = arg.args[i + 1]
                assert isinstance(ivar, Integer), f"ivar={ivar}, type={type(ivar)}"
                result[i] = max(result[i], abs(int(ivar)))

        if str(type(expr)) == "stencil":
            extract(expr)

        for arg in expr.args:
            if str(type(arg)) == "stencil":
                extract(arg)
            else:
                if isinstance(arg, Expr):
                    self._stencil_limits(result, arg)

    def stencil_idxes(self) -> set[StencilIdxWithName]:
        result: set['StencilIdxWithName'] = set()
        for eqn in self.eqns.values():
            self._stencil_idxes(result, eqn)
        return result

    def _stencil_idxes(self, result: set[StencilIdxWithName], expr: Expr) -> None:
        grid_vars = self._grid_variables()
        stencil_calls: set[Basic] = expr.find(lambda x: hasattr(x, 'func') and self.is_stencil.get(x.func, False))  # type: ignore[no-untyped-call]
        straight_accesses: set[Basic] = expr.xreplace({call: Symbol("_stencil_call") for call in stencil_calls}).find(lambda x: x in grid_vars)  # type: ignore[no-untyped-call]

        for access in straight_accesses:
            result.add(StencilIdxWithName(StencilIdx(0, 0, 0), str(access)))

        for store in self.outputs:
            result.add(StencilIdxWithName(StencilIdx(0, 0, 0), str(store)))

        for call in stencil_calls:
            assert len(call.args) == 4, "Stencil function should have 4 arguments"
            result.add(StencilIdxWithName(tuple(int(typing.cast(Expr, a).evalf()) for a in call.args[1:]), str(call.args[0])))  # type: ignore[arg-type]

    def dump(self) -> None:
        print(colored("Dumping Equations:", "green"))
        for k in self.order:
            print(" ", colored(k, "cyan"), "=", self.eqns[k])
