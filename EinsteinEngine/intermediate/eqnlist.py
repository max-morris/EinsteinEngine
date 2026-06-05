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

import typing
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from functools import cache, partial
from functools import cached_property
from itertools import chain
from statistics import mean, median
from typing import cast, Dict, List, Optional, Set, Callable, Iterable, NamedTuple, Never, Generator

from multimethod import multimethod
from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sortedcontainers import SortedDict
# noinspection PyUnusedImports
from sympy import Basic, IndexedBase, Symbol, Integer, Expr

from EinsteinEngine.intermediate.analytic_function_checker import AnalyticFunctionChecker
from EinsteinEngine.intermediate.dependencies import Dependencies
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.intermediate.eqn_ordering import maximize_symbol_reuse, EqnOrderingFn, score_memory_pressure, \
    prioritize_rare_symbols, respects_dependency_order
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

    def __init__(self,
                 is_stencil: Dict[UFunc, bool],
                 intent_override: Optional[IntentOverride] = None,
                 set_eqn_annotation: Optional[Callable[[int, Symbol, str], None]] = None) -> None:
        self.is_stencil = is_stencil
        self.intent_override = intent_override
        self.set_eqn_annotation = set_eqn_annotation
        self.eqn_lists = [EqnList(self, is_stencil, partial(self.set_eqn_annotation, 0) if self.set_eqn_annotation else None)]
        self.been_baked = False
        self._tile_temporaries = OrderedSet()
        self._hard_splits = set()
        self._soft_split_retainment_strategies = dict()

    def _new_eqn_list(self, soft_split: bool = False, soft_split_retainment_strategy: Optional[SoftSplitRetainmentStrategy] = None) -> 'EqnList':
        new_list = EqnList(self, self.is_stencil, partial(self.set_eqn_annotation, len(self.eqn_lists)) if self.set_eqn_annotation else None)
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

            completed_syms: set[Symbol] = set()
            for el_idx in range(first_el + 1, last_el + 1):
                eqn_list = self.eqn_lists[el_idx]
                subst = subst_by_kernel[el_idx]

                local_mangled_reads: set[Symbol] = set()
                local_mangled_reads_by_kernel[el_idx] = local_mangled_reads

                for lhs, rhs in eqn_list.eqns.items():
                    new_lhs = subst.get(lhs, lhs)

                    if new_lhs in new_eqns or new_lhs in recipient_el.eqns:
                        continue

                    new_rhs = rhs.xreplace(subst)  # type: ignore[no-untyped-call]

                    eqn_mangled_reads = new_rhs.free_symbols.intersection(inv_subst.keys())
                    local_mangled_reads.update(eqn_mangled_reads)

                    new_eqns[new_lhs] = new_rhs

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

            for lhs, rhs in new_eqns.items():
                recipient_el.add_eqn(lhs, rhs)

        for el_idx in reversed(els_to_delete):
            del self.eqn_lists[el_idx]

        for el_idx, el in enumerate(self.eqn_lists):
            el.set_eqn_annotation = partial(self.set_eqn_annotation, el_idx) if self.set_eqn_annotation else None

        pprint(f'Rebaking loops after merge_soft_splits...')
        # Rebaking all loops instead of just recipients because CSE will have rebaked with force_fast=True.
        # This also realigns ordering annotations for us.
        for el in self.eqn_lists:
            el.bake(force_rebake=True)

        self._hard_splits = set(range(1, len(self.eqn_lists)))

        return _MergeSoftSplitsResult(all_subst, inv_subst)

    @cache
    def _calc_tile_temps(self) -> None:
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
                assert all(read_el > written_el for read_el in read_els), f"Determined {temp} should be a tile-temp in {self}, but it is written ({written_el}) after is is read ({read_els})"

                self._tile_temporaries.add(temp)
                self.eqn_lists[written_el].uninitialized_tile_temporaries.add(temp)
                for el_idx in read_els:
                    self.eqn_lists[el_idx].preinitialized_tile_temporaries.add(temp)

    @cache
    def _calc_vars(self) -> None:
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

    def __init__(self,
                 parent: EqnComplex,
                 is_stencil: Dict[UFunc, bool],
                 set_eqn_annotation: Optional[Callable[[Symbol, str], None]] = None) -> None:
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
        self.set_eqn_annotation = set_eqn_annotation
        self.eqn_insertion_order: OrderedDict[Symbol, int] = OrderedDict()

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

    def add_eqn(self, lhs: Symbol, rhs: Expr) -> None:
        if lhs in self.eqns:
            raise IntermediateException(f"Equation for '{lhs}' is already defined")

        # Ensure we only have symbols in eqnlist
        self.eqns[lhs := symbify(lhs)] = symbify(rhs)
        self.eqn_insertion_order[lhs] = len(self.eqns) - 1

    def do_pull_out(self, name_generator: Generator[str, Never, Never]) -> None:
        new_eqns: OrderedDict[Symbol, Expr] = OrderedDict()
        modify_eqns: OrderedDict[Symbol, Expr] = OrderedDict()

        assert self.eqns.keys() == self.eqn_insertion_order.keys()

        for lhs in self.eqn_insertion_order.keys():
            rhs = self.eqns[lhs]
            for sub_expr in sorted(rhs.find(pull_out), key=str):  # type: ignore[no-untyped-call]
                if len(sub_expr_args := sub_expr.args) > 1:
                    raise IntermediateException("pull_out() should have only one argument")
                new_sym = mk_symbol(next(name_generator))
                assert new_sym not in new_eqns
                new_eqns[new_sym] = sub_expr_args[0]
                rhs = rhs.xreplace({sub_expr: new_sym})  # type: ignore[no-untyped-call]
            assert lhs not in modify_eqns
            modify_eqns[lhs] = rhs

        for lhs, rhs in new_eqns.items():
            self.add_eqn(lhs, rhs)
            self.temporaries.add(lhs)

        for lhs, rhs in modify_eqns.items():
            self.eqns[lhs] = rhs

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

    def bake(self, *, force_rebake: bool = False, force_fast: bool = False) -> None:
        """ Discover inconsistencies and errors in the param/input/output/equation sets. """
        if self.been_baked and not force_rebake:
            raise DslException("Can't bake an EqnList that has already been baked.")
        self.been_baked = True

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
            checker = AnalyticFunctionChecker(self.params, self.eqns)
            for lhs in checker.analytic():
                if lhs in self.outputs:
                    self.write_decls[lhs] = IntentRegion.Everywhere

        vprint(colorize("Inputs:", "green"), self.inputs)
        vprint(colorize("Outputs:", "green"), self.outputs)
        vprint(colorize("Params:", "green"), self.params)

        for k in self.eqns:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            written.add(k)
            for q in free_symbols(self.eqns[k]):
                read.add(q)

        vprint(colorize("Read:", "green"), read)
        vprint(colorize("Written:", "green"), written)

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

        vprint(colorize("Temps:", "green"), self.temporaries)
        vprint(colorize("Uninitialized Tile Temps:", "green"), self.uninitialized_tile_temporaries)
        vprint(colorize("Preinitialized Tile Temps:", "green"), self.preinitialized_tile_temporaries)

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

        order_builder_kwargs = dict()

        # Simple stopgap to prevent wasteful bayesian optimization calls before CSE
        # todo: maybe make this check less hacky?
        if (
                hasattr(self.ordering_fn, 'func')
                and 'bayesian' in self.ordering_fn.func.__name__
                and (not force_rebake or force_fast)
        ) or (
                hasattr(self.ordering_fn, '__name__')
                and 'bayesian' in self.ordering_fn.__name__
                and (not force_rebake or force_fast)
        ):
            order_builder_kwargs['override_ordering_fn'] = prioritize_rare_symbols

        self.order_builder(complete, **order_builder_kwargs)

        vprint(colorize("Order:", "green"), self.order)

        try:
            memory_pressure = score_memory_pressure(self.eqns, self.order)
            vprint(colorize("Memory Pressure:", "magenta"))
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

        vprint(colorize("READS:", "green"), end="")
        for var, spec in self.read_decls.items():
            if var in self.inputs:
                vprint(" ", var, "=", colorize(repr(spec), "yellow"), sep="", end="")
        vprint()
        vprint(colorize("WRITES:", "green"), end="")
        for var, spec in self.write_decls.items():
            if var in self.outputs:
                vprint(" ", var, "=", colorize(repr(spec), "yellow"), sep="", end="")
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
            vprint(colorize("EQN:", "cyan"), lhs, colorize("=", "cyan"), rhs, " ", colorize(f"[complexity = {self.complexity[lhs]}]", "magenta"))

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
            result.add(StencilIdxWithName(tuple(int(typing.cast(Expr, a).evalf()) for a in call.args[1:]), str(call.args[0])))  # type: ignore[arg-type, no-untyped-call]

    def dump(self) -> None:
        print(colorize("Dumping Equations:", "green"))
        for k in self.order:
            print(" ", colorize(k, "cyan"), "=", self.eqns[k])
