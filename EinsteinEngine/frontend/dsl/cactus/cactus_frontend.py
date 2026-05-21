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

import re
import typing
from collections import defaultdict
from enum import auto
from itertools import chain
from typing import Collection, TypedDict, Optional, OrderedDict, cast, List, Unpack, Set, Union, Dict, Iterator, \
    NamedTuple, Iterable, Any

from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sympy import Symbol, Expr, Idx, Indexed, Basic, IndexedBase, Eq, Matrix

import EinsteinEngine.common.util as util
from EinsteinEngine.common.cse_optimization_level import CseOptimizationLevel
from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.common.sympywrap import (
    cse, free_symbols, mk_eq, mk_indexed_base, mk_symbol
)
from EinsteinEngine.emit.ccl.schedule.schedule_tree import GroupOrFunction, ScheduleBlock
from EinsteinEngine.emit.tree import Centering, Identifier
from EinsteinEngine.frontend.dsl.use_indices import subst_tensor_xyz
from EinsteinEngine.frontend.eqn_ordering import EqnOrderingFn, maximize_symbol_reuse
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy, retain_none
from EinsteinEngine.intermediate.temp_kind import TempKind
from EinsteinEngine.intermediate.temporary_promotion_predicate import (
    OnePassTemporaryPromotionStrategy, TemporaryPromotionPredicate, TemporaryPromotionStrategy,
    TwoPassTemporaryPromotionStrategy, promote_all, promote_none
)
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.emit.ccl.interface.interface_tree import TensorParity, Parity, SingleIndexParity
from EinsteinEngine.frontend.dsl.cactus.cactus_param import CactusParam, CactusParamValuesType, CactusParamDefaultType
from EinsteinEngine.frontend.dsl.use_indices import do_isub, subst_tensor
from EinsteinEngine.frontend.util import cse_isolate
from EinsteinEngine.generators.sympy_complexity import SympyComplexityVisitor
from EinsteinEngine.intermediate.eqnlist import EqnComplex, EqnList
from EinsteinEngine.intermediate.splitmaxxer import SplitMaxxer
from EinsteinEngine.common.util import ScheduleBinEnum, ScheduleFrequency, wprint, vprint, OrderedSet, pprint, get_or_compute, verbose

from EinsteinEngine.common.schedule_target import ScheduleTarget, safe_name
from EinsteinEngine.frontend.dsl.dsl_frontend import (
    DslFrontend,
    DslFrontendBakeOptions,
    SymbolDeclaration,
    SymbolDeclarationKwargs,
)
from EinsteinEngine.frontend.dsl.add_eqn_manager import AddEqnManager
from EinsteinEngine.frontend.dsl.dsl_frontend import mk_mk_subst

TfName = typing.NewType("TfName", str)
LocalElIdx = typing.NewType("LocalElIdx", int)

class CactusDeclOptionalArgs(SymbolDeclarationKwargs, total=False):
    centering: Centering
    rhs: IndexedBase
    from_thorn: str
    parity: TensorParity
    group_name: str

class ScheduleBin(ScheduleBinEnum):
    Init = auto(), 'Init', True,  ScheduleFrequency.Once, 0
    DriverInit = auto(), 'ODESolvers_Initial', False, ScheduleFrequency.Once, 1
    PostInit = auto(), 'PostInit', True,  ScheduleFrequency.Once, 2
    PostPostInit = auto(), 'PostPostInit', True,  ScheduleFrequency.Once, 3
    InitEvolve = auto(), 'InitEvolve', False,  ScheduleFrequency.Once, 3
    InitAnalysis = auto(), 'InitAnalysis', False,  ScheduleFrequency.Once, 4
    Evolve = auto(), 'Evolve', False, ScheduleFrequency.EachStep, 6
    SpecialEvolve = auto(), 'SpecialEvolve', False, ScheduleFrequency.EachStep, 7
    PostSubStep = auto(), 'PostSubStep', False, ScheduleFrequency.EachStep, 8
    PostStep = auto(), 'PostStep', True, ScheduleFrequency.EachStep, 9
    Analysis = auto(), 'Analysis', True, ScheduleFrequency.EachStep, 10
    EstimateError = auto(), 'EstimateError', False, ScheduleFrequency.Inconsistent, 11

    def is_colocated(self, other: 'ScheduleBin') -> bool:
        return self == other or (
                (s := sorted([self, other], key=lambda b: b.relative_order))[0] is ScheduleBin.Evolve and s[1] is ScheduleBin.SpecialEvolve
        )

    @staticmethod
    def _schedule_synthetic_fns(bins: Collection['ScheduleBin']) -> Collection['ScheduleBin']:
        ret: list['ScheduleBin'] = list()
        freqs: set[ScheduleFrequency] = set()
        bins = sorted(bins, key=lambda b: b.relative_order)

        for bin in bins:
            if bin.schedule_frequency == ScheduleFrequency.Inconsistent:
                freqs.add(bin.schedule_frequency)
                ret.append(bin)
                wprint(f'A global temp is accessed by a thorn function in schedule bin {bin}, which has an inconsistent schedule frequency. The temporary will be recomputed, perhaps redundantly.')
            elif bin in [ScheduleBin.PostInit, ScheduleBin.PostPostInit]:  # Never elide PostInit targets. Needed for the timestep 0 PostInit hack.
                freqs.add(bin.schedule_frequency)
                ret.append(bin)
            elif len(freqs) > 0 and bin.schedule_frequency not in freqs:
                freqs.add(bin.schedule_frequency)
                ret.append(bin)
                wprint(f'A global temp is accessed by thorn functions in schedule bins {freqs} with disparate schedule frequencies. The temporary will be recomputed, perhaps redundantly.')
            elif len(freqs) == 0:
                freqs.add(bin.schedule_frequency)
                ret.append(bin)
            else:
                assert bin.schedule_frequency in freqs

        return ret


class ThornFunctionBakeOptions(TypedDict, total=False):
    do_madd: bool
    do_recycle_temporaries: bool
    splitmaxxing: bool
    ordering_fn: EqnOrderingFn
    soft_split_retainment_strategy: SoftSplitRetainmentStrategy


class ThornDefBakeOptions(DslFrontendBakeOptions, total=False):
    # ThornDef opts
    do_cse: bool
    cse_optimization_level: CseOptimizationLevel
    temporary_promotion_strategy: TemporaryPromotionStrategy

    # ThornFunction default opts
    do_madd: bool
    do_recycle_temporaries: bool
    splitmaxxing: bool
    ordering_fn: EqnOrderingFn
    soft_split_retainment_strategy: SoftSplitRetainmentStrategy

    # Overrides for ThornFunction default opts
    functions: dict[str, ThornFunctionBakeOptions]


class ThornFunction:
    """
    Represents a function within a Cactus thorn. Important member functions include `add_eqn` for specifying
    the computations this function will perform, and `bake` for finalizing the function.
    """

    def __init__(self,
                 name: str,
                 schedule_target: ScheduleTarget,
                 thorn_def: "ThornDef",
                 schedule_before: Optional[Collection[str]],
                 schedule_after: Optional[Collection[str]],
                 intent_override: Optional[IntentOverride] = None) -> None:
        self.schedule_target = schedule_target
        self.name = name
        self.thorn_def = thorn_def
        self.source_annotations: SourceAnnotations = SourceAnnotations()
        self.source_annotations.loops[0] = f'{self.name} loop 0'

        def set_eqn_annotation(loop_idx: int, key: Symbol, annotation: str) -> None:
            self.source_annotations.eqns[loop_idx][key] = annotation

        self.eqn_complex: EqnComplex = EqnComplex(thorn_def.is_stencil, intent_override, set_eqn_annotation)
        self.been_baked: bool = False
        self.been_late_baked: bool = False
        self.schedule_before: Collection[str] = schedule_before or list()
        self.schedule_after: Collection[str] = schedule_after or list()
        self.intent_override = intent_override
        self._add_eqn_manager = AddEqnManager(
            thorn_def,
            lambda: self._eqn_list,
            lambda: self.been_baked,
            owner_name="ThornFunction"
        )

        if isinstance(schedule_target, ScheduleBlock) and schedule_target.group_or_function is GroupOrFunction.Function:
            raise DslException("Cannot schedule into this schedule block because it is not a schedule group.")

    def needs_merge(self) -> bool:
        return self.eqn_complex.needs_merge()

    def merge_soft_splits(self, soft_split_retainment_strategy: SoftSplitRetainmentStrategy) -> None:
        _, inv_subst = self.eqn_complex.merge_soft_splits(soft_split_retainment_strategy)

        for mangled_sym, sym in inv_subst.items():
            if (c := self.thorn_def.centering.get(str(sym))) is not None:
                self.thorn_def.centering[str(mangled_sym)] = c
            elif (sym_base := self.thorn_def.var2base.get(str(sym))) is not None:
                if (c := self.thorn_def.centering.get(sym_base)) is not None:
                    self.thorn_def.centering[str(mangled_sym)] = c

        for el_idx in range(len(self.eqn_complex.eqn_lists)):
            self.source_annotations.loops[el_idx] = f'{self.name} loop {el_idx}'

    @property
    def _eqn_list(self) -> EqnList:
        return self.eqn_complex.get_active_eqn_list()

    def _base_add_eqn(self, lhs2: Symbol, rhs2: Expr) -> None:
        self._add_eqn_manager._base_add_eqn(lhs2, rhs2)

    def get_free_indices(self, expr: Expr) -> OrderedSet[Idx]:
        return self.thorn_def.get_free_indices(expr)

    def split_loop(self, annotation: Optional[str] = None) -> None:
        if self.been_baked:
            raise DslException("Cannot split loop because the EqnComplex has already been baked.")

        loop_idx = len(self.eqn_complex.eqn_lists)
        if annotation is None:
            annotation = f'{self.name} loop {loop_idx}'

        if annotation.strip() != '':
            self.source_annotations.loops[loop_idx] = annotation

        self.eqn_complex._new_eqn_list()

    def soft_split(self, retainment_strategy: Optional[SoftSplitRetainmentStrategy] = None, annotation: Optional[str] = None) -> None:
        if self.been_baked:
            raise DslException("Cannot split loop because the EqnComplex has already been baked.")

        loop_idx = len(self.eqn_complex.eqn_lists)
        if annotation is None:
            annotation = f'{self.name} loop {loop_idx} (soft split)'

        if annotation.strip() != '':
            self.source_annotations.loops[loop_idx] = annotation

        self.eqn_complex._new_eqn_list(True, soft_split_retainment_strategy=retainment_strategy)

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

            pprint(f'Rebaking {self.name} loop {loop_idx} after do_splitmaxxing...')
            eqn_list.bake(force_rebake=True)

            if util.verbose():
                eqn_list.dump()

    def add_eqn(self, lhs: Indexed | IndexedBase, rhs: Expr | Matrix | list[Expr]) -> None:
        self._add_eqn_manager.add_eqn(lhs, rhs)

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
    def _mk_default_thorn_function_bake_options() -> ThornFunctionBakeOptions:
        return {
            'do_madd': False,
            'do_recycle_temporaries': True,
            'splitmaxxing': False,
            'ordering_fn': maximize_symbol_reuse,
            'soft_split_retainment_strategy': retain_none()
        }

    def _early_bake(self, **kwargs: Unpack[ThornFunctionBakeOptions]) -> None:
        if self.been_baked:
            raise DslException("_early_bake should not be called more than once")
        pprint(f"Early Baking {self.name}...")

        options = self._mk_default_thorn_function_bake_options()
        options.update(kwargs)

        # Doing a first pass of complexity analysis for CSE
        for eqn_list in self.eqn_complex.eqn_lists:
            eqn_list._run_preliminary_complexity_analysis()

        if options['do_madd']:
            self.madd()

        self.eqn_bake(options['ordering_fn'])

        self.been_baked = True

    def _late_bake(self, **kwargs: Unpack[ThornFunctionBakeOptions]) -> None:
        if self.been_late_baked:
            raise DslException("_late_bake should not be called more than once")
        pprint(f"Late Baking {self.name}...")

        options = self._mk_default_thorn_function_bake_options()
        options.update(kwargs)

        if options['do_recycle_temporaries']:
            self.recycle_temporaries()

        self.been_late_baked = True

    def show_tensor_types(self) -> None:
        keys: Set[str] = OrderedSet()
        for k1 in self.eqn_complex.inputs:
            keys.add(str(k1))
        for k2 in self.eqn_complex.outputs:
            keys.add(str(k2))
        for k in keys:
            group, indices, members = self.get_tensor_type(k)
            print(colorize(k, "green"), "is a member of", colorize(group, "green"), "with indices",
                  colorize(indices, "cyan"), "and members", colorize(members, "magenta"))

    def get_tensor_type(self, item: Union[str, Symbol]) -> tuple[str, tuple[Idx, ...], tuple[str, ...]]:
        return self.thorn_def.get_tensor_type(item)


class ThornDef(DslFrontend[CactusParam, CactusDeclOptionalArgs]):
    """
    Represents a Cactus thorn. A ThornDef object contains everything EinsteinEngine knows about a thorn over the course
    of evaluating a recipe. It is also an important interface for declaring variables, adding new thorn functions,
    and more.
    """

    # These thorns do tensor expansion with the xyz rules as opposed to our preferred nrpy rules.
    # noinspection SpellCheckingInspection
    _xyz_subst_thorns: list[str] = ["ADMBaseX", "TmunuBaseX", "HydroBaseX"]

    # Hardcoding some known nonsensical mappings from other thorns.
    # noinspection SpellCheckingInspection
    _special_group_mappings: dict[str, dict[str, str]] = {
        # https://github.com/EinsteinToolkit/CarpetX/blob/main/ADMBaseX/interface.ccl
        'ADMBaseX': {
            'g': 'metric',
            'k': 'curv',
            'alp': 'lapse',
            'beta': 'shift',
            'dtalp': 'dtlapse',
            'dtbeta': 'dtshift'
        },
        # https://github.com/EinsteinToolkit/CarpetX/blob/main/TmunuBaseX/interface.ccl
        'TmunuBaseX': {
            'eTt': 'eTti',
            'eT': 'eTij'
        }
    }

    def __init__(
            self,
            arr: str,
            name: str,
            *,
            dimensionality: int = 3,
            derivative_stencil_order: int = 5
    ) -> None:
        super().__init__(
            dimensionality=dimensionality,
            derivative_stencil_order=derivative_stencil_order
        )

        if not _is_valid_c_identifier(name):
            raise DslException(f"Thorn name '{name}' is not a valid C identifier")

        self.arrangement = arr
        self.name = name
        self.base2group: Dict[str, str] = dict()
        self.groups: Dict[str, List[str]] = dict()
        self.centering: Dict[str, Optional[Centering]] = dict()
        self.thorn_functions: Dict[str, ThornFunction] = dict()
        self.rhs: Dict[str, Symbol] = dict()
        self.base2thorn: Dict[str, str] = dict()
        self.base2parity: Dict[str, TensorParity] = dict()
        self.tile_temporaries: OrderedSet[Symbol] = OrderedSet()
        self.global_temporaries: OrderedSet[Symbol] = OrderedSet()
        self.synthetic_fns: dict[ScheduleTarget, set[ThornFunction]] = defaultdict(set)

    def _grid_variables(self) -> set[Symbol]:
        gv: set[Symbol] = set()
        for tf in self.thorn_functions.values():
            gv |= tf.eqn_complex._grid_variables()
        return gv

    def get_centering_from_var_name(self, var_name: str) -> Optional[Centering]:
        var_centering: Optional[Centering]

        # Try looking up the var's centering directly...
        if (var_centering := self.centering.get(var_name, None)) is not None:
            pass
        # Otherwise, try looking it up by the var's base...
        elif (var_base := self.var2base.get(var_name, None)) is not None:
            var_centering = self.centering.get(var_base, None)

        return var_centering

    def _flatten_indexed(self, sym: Indexed) -> Iterator[Symbol]:
        count = 0
        for sym_x, idxes, _ in self.einstein_notation.expand_free_indices(sym, self.symmetries):
            count += 1
            sym2: Basic = do_isub(sym_x, self.subs)
            if not isinstance(sym2, Symbol):
                mms = mk_mk_subst(repr(sym2))
                raise Exception(f"'{sym2}' does not evaluate a Symbol. Did you forget to call mk_subst({mms},...)?")
            yield sym2
        if count == 0:
            for ind in sym.args[1:]:
                assert isinstance(ind, Idx)
                assert self.einstein_notation.is_numeric_index(ind)
            yield cast(Symbol, self._do_subs(sym))

    @staticmethod
    def _mk_default_thorn_def_bake_options() -> ThornDefBakeOptions:
        opts: ThornDefBakeOptions = {
            'do_cse': True,
            'cse_optimization_level': CseOptimizationLevel.Optimal,
            'temporary_promotion_strategy': promote_none(),
            'ordering_fn': maximize_symbol_reuse,
            'functions': dict()
        }

        opts.update(ThornFunction._mk_default_thorn_function_bake_options())  # type: ignore[typeddict-item]

        return opts


    def bake(self, **opts: Unpack[ThornDefBakeOptions]) -> None:
        my_opts = self._mk_default_thorn_def_bake_options()
        my_opts.update(opts)
        my_tf_opts: dict[str, ThornFunctionBakeOptions] = dict()

        for tf in self.thorn_functions.values():
            tf_opts = typing.cast(ThornFunctionBakeOptions, my_opts.copy())  # ThornFunctionBakeOptions is a strict subset of ThornDefBakeOptions
            if 'functions' in my_opts and tf.name in my_opts['functions']:
                tf_opts.update(my_opts['functions'][tf.name])
            my_tf_opts[tf.name] = tf_opts

        for tf in self.thorn_functions.values():
            assert tf.name in my_tf_opts, f"Thorn function '{tf.name}' not found in my_tf_opts"
            tf._early_bake(**my_tf_opts[tf.name])

        if my_opts['do_cse']:
            pprint("Performing CSE...")
            self._do_global_cse(my_opts['temporary_promotion_strategy'], my_opts['cse_optimization_level'])

        for tf in self.thorn_functions.values():
            if tf.needs_merge():
                pprint(f"Merging soft splits in {tf.name}...")
                tf.merge_soft_splits(my_tf_opts[tf.name]['soft_split_retainment_strategy'])

        for tf in self.thorn_functions.values():
            if tf.name not in my_tf_opts:  # Must be a synthetic function
                tf._late_bake()
            else:
                if my_tf_opts[tf.name]['splitmaxxing']:
                    tf._do_splitmaxxing()
                tf._late_bake(**my_tf_opts[tf.name])


    def _do_global_cse(
            self,
            promotion_strategy: TemporaryPromotionStrategy = promote_all(),
            optimization_level: CseOptimizationLevel = CseOptimizationLevel.Optimal
    ) -> None:
        for tf in self.thorn_functions.values():
            if tf.been_late_baked:
                raise DslException(f"Cannot do_global_cse on ThornFunction {self} because it has already undergone late baking.")

        grid_vars = self._grid_variables()

        tf_names: list[TfName] = sorted([TfName(name) for name in self.thorn_functions.keys()])
        old_tf_shapes: OrderedDict[TfName, list[int]] = OrderedDict()
        old_tf_lhses: OrderedDict[TfName, list[list[Symbol]]] = OrderedDict()
        old_tf_rhses: OrderedDict[TfName, list[list[Expr]]] = OrderedDict()

        for tf_name in tf_names:
            old_tf_shapes[tf_name] = list()
            old_tf_lhses[tf_name] = list()
            old_tf_rhses[tf_name] = list()

        for tf_name, tf in sorted([(TfName(name), tf) for name, tf in self.thorn_functions.items()], key=lambda kv: tf_names.index(kv[0])):
            for eqn_list in tf.eqn_complex.eqn_lists:
                old_tf_shapes[tf_name].append(len(eqn_list.eqns))
                old_tf_lhses[tf_name].append(list())
                old_tf_rhses[tf_name].append(list())
                for lhs, rhs in sorted(eqn_list.eqns.items(), key=lambda kv: eqn_list.order.index(kv[0])):
                    old_tf_lhses[tf_name][-1].append(lhs)
                    old_tf_rhses[tf_name][-1].append(rhs)

        substitutions_list: list[tuple[Symbol, Expr]]
        new_rhses: list[Expr]

        if optimization_level is CseOptimizationLevel.Optimal:
            substitutions_list, new_rhses = cse_isolate(
                list(chain(*chain(*old_tf_rhses.values()))),
                symbols_to_isolate=grid_vars
            )
        elif optimization_level is CseOptimizationLevel.Fast:
            substitutions_list, new_rhses = cse(list(chain(*chain(*old_tf_rhses.values()))))
        else:
            raise DslException(f"Unrecognized CSE optimization level: {optimization_level}")

        substitutions = {lhs: rhs for lhs, rhs in substitutions_list}
        substitutions_order = {lhs: idx for idx, (lhs, _) in enumerate(substitutions_list)}

        new_temp_direct_reads: dict[Symbol, dict[TfName, set[LocalElIdx]]] = {sym: dict() for sym in substitutions.keys()}
        new_temp_dependencies: dict[Symbol, set[Symbol]] = {sym: set() for sym in substitutions.keys()}
        new_temp_dependents: dict[Symbol, set[Symbol]] = {sym: set() for sym in substitutions.keys()}

        temp_rhs_occurrences: dict[Symbol, int] = defaultdict(int)
        for rhs in new_rhses:
            for temp in free_symbols(rhs).intersection(substitutions.keys()):
                temp_rhs_occurrences[temp] += 1

        for rhs in substitutions.values():
            for temp in free_symbols(rhs).intersection(substitutions.keys()):
                temp_rhs_occurrences[temp] += 1

        global_eqn_idx = 0
        for tf_index, tf in enumerate(sorted(self.thorn_functions.values(), key=lambda tf: tf_names.index(TfName(tf.name)))):
            tf_name = TfName(tf.name)

            for el_idx, el_shape in enumerate(old_tf_shapes[tf_name]):
                eqn_list = tf.eqn_complex.eqn_lists[el_idx]
                el_new_free_symbols: set[Symbol] = set(chain(*[free_symbols(rhs) for rhs in new_rhses[global_eqn_idx:global_eqn_idx + el_shape]]))
                new_temps = el_new_free_symbols.intersection(substitutions.keys())

                for new_temp, temp_rhs in [(new_temp, substitutions[new_temp]) for new_temp in new_temps]:
                    assert new_temp not in eqn_list.inputs
                    assert new_temp not in eqn_list.params
                    assert new_temp not in eqn_list.outputs
                    assert new_temp not in eqn_list.eqns

                    get_or_compute(new_temp_direct_reads[new_temp], tf_name, lambda _: set()).add(LocalElIdx(el_idx))

                    # Temps might be substituted for expressions which contain other temps.
                    # We need to recursively check the RHSes to ensure we compute the dependencies in the appropriate loops.
                    def drill(lhs: Symbol, rhs: Expr) -> None:
                        temp_dependencies = free_symbols(rhs).intersection(substitutions.keys())
                        assert lhs not in temp_dependencies
                        for td in temp_dependencies:
                            new_temp_dependencies[lhs].add(td)
                            new_temp_dependents[td].add(lhs)
                            drill(td, substitutions[td])

                    drill(new_temp, temp_rhs)

                global_eqn_idx += el_shape

        tfs_reading_direct: dict[Symbol, dict[ThornFunction, set[LocalElIdx]]] = defaultdict(lambda: dict())

        for new_temp, new_rhs in substitutions.items():
            tf_names_reading_direct = set(new_temp_direct_reads[new_temp].keys())

            for tf_name in tf_names_reading_direct:
                els_reading_direct = new_temp_direct_reads[new_temp].get(tf_name, set())
                tfs_reading_direct[new_temp][self.thorn_functions[tf_name]] = els_reading_direct

        complexities: dict[Symbol, int] = dict()
        for tf in self.thorn_functions.values():
            for eqn_list in tf.eqn_complex.eqn_lists:
                complexities.update(eqn_list.complexity)

        complexity_visitor = SympyComplexityVisitor(
            lambda s: s in grid_vars
        )
        for new_temp, new_rhs in substitutions.items():
            complexities[new_temp] = complexity_visitor.complexity(new_rhs)

        promotion_predicate: TemporaryPromotionPredicate
        two_pass: bool
        if isinstance(promotion_strategy, OnePassTemporaryPromotionStrategy):
            promotion_predicate = promotion_strategy(complexities)
            two_pass = False
        elif isinstance(promotion_strategy, TwoPassTemporaryPromotionStrategy):
            # noinspection PyUnnecessaryCast
            promotion_predicate = cast(OnePassTemporaryPromotionStrategy, promote_all())(complexities)
            two_pass = True
        else:
            raise DslException(f"Not a valid promotion strategy: {promotion_strategy}")

        temp_kinds, tfs_active_reads, synthetic_global_dependents = self._classify_temps(
            new_temp_dependents,
            promotion_predicate,
            substitutions,
            tfs_reading_direct,
            substitutions_order
        )

        if two_pass:
            assert isinstance(promotion_strategy, TwoPassTemporaryPromotionStrategy)
            promotion_predicate = promotion_strategy(complexities, temp_kinds)

            temp_kinds, tfs_active_reads, synthetic_global_dependents = self._classify_temps(
                new_temp_dependents,
                promotion_predicate,
                substitutions,
                tfs_reading_direct,
                substitutions_order
            )

        checked_deps: set[Symbol] = set()
        def compute_centerings(temp: Symbol) -> None:
            if temp in checked_deps:
                return

            checked_deps.add(temp)

            for td in new_temp_dependencies[temp]:
                compute_centerings(td)

            centerings = {
                c for c in {
                    self.centering.get(self.var2base.get(str(sym)) or str(sym)) for sym in free_symbols(substitutions[temp])
                } if c is not None
            }

            if len(centerings) == 0:
                #raise DslException(f"Could not infer a centering for temp {temp} -> {substitutions[temp]}; none of its dependencies have centerings")
                #todo: Cases where a temp has no grid functions in its RHS might require us to check its dependents
                wprint(f"Could not infer a centering for temp {temp} -> {substitutions[temp]}; none of its dependencies have centerings. Defaulting to VVV.")
                centerings = {Centering.VVV}
            elif len(centerings) > 1:
                raise DslException(f"Could not infer a centering for temp {temp} -> {substitutions[temp]}; its dependencies have conflicting centerings {centerings}")

            assert len(centerings) == 1
            self.centering[str(temp)] = centerings.pop()

        for new_temp in substitutions.keys():
            compute_centerings(new_temp)

        schedule_blocks: dict[Identifier, ScheduleBlock] = dict()
        schedule_bin_targets: dict[Symbol, dict[ScheduleBin, set[ThornFunction]]] = defaultdict(lambda: defaultdict(set))
        schedule_block_targets: dict[Symbol, dict[Identifier, set[ThornFunction]]] = defaultdict(lambda: defaultdict(set))

        for new_temp in substitutions.keys():
            vprint(colorize("Temporary:", "cyan"), new_temp, colorize(f"[kind = {temp_kinds.get(new_temp, TempKind.Inline)}]", "magenta"))

        inline_temps: list[tuple[Symbol, Expr]] = list()
        for new_temp, new_rhs in sorted(substitutions.items(),
                                        key=lambda kv: substitutions_order[kv[0]],
                                        reverse=True):
            if temp_kinds.get(new_temp, None) == TempKind.Inline:
                inline_temps.append((new_temp, new_rhs))

        new_rhses = [rhs.subs(inline_temps) for rhs in new_rhses]  # type: ignore[no-untyped-call]

        for lhs in substitutions.keys():
            substitutions[lhs] = substitutions[lhs].subs(inline_temps)  # type: ignore[no-untyped-call]

        for temp, _ in inline_temps:
            del substitutions[temp]

        global_eqn_idx = 0
        for tf_index, tf in enumerate(sorted(self.thorn_functions.values(), key=lambda tf: tf_names.index(TfName(tf.name)))):
            tf_name = TfName(tf.name)
            for el_idx, el_shape in enumerate(old_tf_shapes[tf_name]):
                eqn_list = tf.eqn_complex.eqn_lists[el_idx]

                for lhs in old_tf_lhses[tf_name][el_idx]:
                    assert lhs in eqn_list.eqns
                    eqn_list.eqns[lhs] = new_rhses[global_eqn_idx]
                    global_eqn_idx += 1


        for new_temp, new_rhs in substitutions.items():
            if new_temp not in temp_kinds or temp_kinds[new_temp] == TempKind.Inline:
                continue
            elif temp_kinds[new_temp] == TempKind.Local:
                for tf, els_reading in tfs_active_reads[new_temp].items():
                    ec = tf.eqn_complex

                    for el in (ec.eqn_lists[el_idx] for el_idx in els_reading):
                        el.add_eqn(new_temp, substitutions[new_temp])
                        el.temporaries.add(new_temp)
            elif temp_kinds[new_temp] == TempKind.Tile:
                self.tile_temporaries.add(new_temp)

                for tf, els_reading in tfs_active_reads[new_temp].items():
                    ec = tf.eqn_complex
                    primary_el = ec.eqn_lists[primary_idx := min(els_reading)]

                    if len(els_reading) == 1:
                        primary_el.add_eqn(new_temp, substitutions[new_temp])
                        primary_el.temporaries.add(new_temp)
                    else:
                        primary_el.add_eqn(new_temp, substitutions[new_temp])
                        ec._tile_temporaries.add(new_temp)
                        primary_el.uninitialized_tile_temporaries.add(new_temp)
                        for eqn_list in [ec.eqn_lists[el_idx] for el_idx in els_reading if el_idx != primary_idx]:
                            eqn_list.preinitialized_tile_temporaries.add(new_temp)
            else:  # TempKind.Global
                self._add_symbol(new_temp, centering=self.centering[str(new_temp)])
                self.global_temporaries.add(new_temp)

                for tf in tfs_active_reads[new_temp]:
                    if isinstance(tf.schedule_target, ScheduleBlock):
                        name = tf.schedule_target.name
                        if name in schedule_blocks:
                            assert schedule_blocks[name] == tf.schedule_target
                        schedule_blocks[name] = tf.schedule_target
                        schedule_block_targets[new_temp][name].add(tf)
                    else:
                        schedule_bin_targets[new_temp][tf.schedule_target].add(tf)

        # Rancid hack: In CarpetX, Evolve DOES NOT run on step 0, while Analysis DOES. This breaks global temps
        #  if they happen to be initialized in Evolve then read in Analysis. To get around this, we will use
        #  PostInit to initialize any synthetic temps that are read in Analysis, plus their (global) dependencies.
        for new_temp in substitutions.keys():
            if temp_kinds.get(new_temp, None) != TempKind.Global:
                continue

            def post_init_hack(tmp: Symbol) -> None:
                return
                if temp_kinds.get(tmp, None) == TempKind.Global:
                    schedule_bin_targets[tmp][ScheduleBin.PostInit].update(set())  # Just touch the set so defaultdict initializes it
                for td in new_temp_dependencies[tmp]:
                    post_init_hack(td)

            if ScheduleBin.Analysis in schedule_bin_targets[new_temp]:
                post_init_hack(new_temp)

        for new_temp in substitutions.keys():
            if temp_kinds.get(new_temp, None) != TempKind.Global:
                continue

            def mk_synthetic_fn(schedule_target: ScheduleTarget,
                                schedule_before: Collection[str],
                                schedule_after: Collection[str]) -> ThornFunction:
                synthetic_fn = self.create_function(
                    f'synthetic_compute_{new_temp}_{safe_name(schedule_target)}',
                    schedule_target,
                    schedule_before=schedule_before,
                    schedule_after=schedule_after
                )
                synthetic_fn._base_add_eqn(new_temp, substitutions[new_temp])

                def add_deps(temp: Symbol) -> None:
                    for td in new_temp_dependencies[temp]:
                        if temp_kinds.get(td, None) in [TempKind.Local, TempKind.Tile]:
                            if td not in synthetic_fn._eqn_list.eqns:
                                synthetic_fn._base_add_eqn(td, substitutions[td])
                            add_deps(td)

                add_deps(new_temp)

                synthetic_fn._early_bake(do_madd=False, do_recycle_temporaries=False)
                self.synthetic_fns[schedule_target].add(synthetic_fn)
                return synthetic_fn

            def find_all_global_deps(temp: Symbol) -> set[Symbol]:
                deps: set[Symbol] = set()
                for td in new_temp_dependencies[temp]:
                    if temp_kinds.get(td, None) == TempKind.Global:
                        deps.add(td)
                    deps.update(find_all_global_deps(td))
                return deps

            for bin in ScheduleBin._schedule_synthetic_fns(schedule_bin_targets[new_temp].keys()):
                schedule_before_tfs = set(chain(*[schedule_bin_targets[new_temp][key] for key in schedule_bin_targets[new_temp].keys() if key.is_colocated(bin)]))
                schedule_after = sorted(list(chain(*[[f'synthetic_compute_{td}_{safe_name(bin)}_group' for dep_bin in schedule_bin_targets[td].keys() if bin.is_colocated(dep_bin)] for td in find_all_global_deps(new_temp)])))
                if bin is ScheduleBin.PostInit:
                    schedule_after.append('ODESolvers_PostStep')  # Hack to ensure AMR and synchronization happen first
                mk_synthetic_fn(bin, sorted([f'{tf.name}_group' for tf in schedule_before_tfs]), schedule_after)

            if len(schedule_block_targets) > 0:
                wprint(f'Global temporary {new_temp} is accessed in at least one custom schedule block,'
                       f' on which EinsteinEngine cannot perform schedule analysis. The temporary will be recomputed for each'
                       f' custom block, perhaps redundantly.')

            for block, schedule_before_tfs in [(schedule_blocks[id], tfs) for id, tfs in schedule_block_targets[new_temp].items()]:
                schedule_after = sorted(list(chain(*[[f'synthetic_compute_{td}_{safe_name(block)}_group' for dep_block_name in schedule_block_targets[new_temp].keys() if block.name == dep_block_name] for td in new_temp_dependencies[new_temp] if temp_kinds.get(td, None) == TempKind.Global])))
                mk_synthetic_fn(block, sorted([f'{tf.name}_group' for tf in schedule_before_tfs]), schedule_after)

        for tf in self.thorn_functions.values():
            for idx, eqn_list in enumerate(tf.eqn_complex.eqn_lists):
                pprint(f'Rebaking {tf.name} loop {idx} after CSE...')

                # If the tf needs a merge, set force_fast because another (slow) rebake will succeed CSE.
                eqn_list.bake(force_rebake=True, force_fast=tf.needs_merge())

                if verbose():
                    eqn_list.dump()

    class _ClassifyTempsResult(NamedTuple):
        temp_kinds: dict[Symbol, TempKind]
        tfs_active_reads: dict[Symbol, dict[ThornFunction, set[LocalElIdx]]]
        synthetic_global_dependents: dict[Symbol, set[Symbol]]


    @staticmethod
    def _classify_temps(
            new_temp_dependents: dict[Symbol, set[Symbol]],
            promotion_predicate: TemporaryPromotionPredicate,
            substitutions: dict[Symbol, Expr],
            tfs_reading_direct: dict[Symbol, dict[ThornFunction, set[LocalElIdx]]],
            substitutions_order: dict[Symbol, int]
    ) -> _ClassifyTempsResult:
        temp_kinds: dict[Symbol, TempKind] = dict()
        tfs_active_reads: dict[Symbol, dict[ThornFunction, set[LocalElIdx]]] = defaultdict(lambda: dict())
        synthetic_global_dependents: dict[Symbol, set[Symbol]] = defaultdict(set)

        for new_temp, new_rhs in sorted(substitutions.items(),
                                        key=lambda kv: substitutions_order[kv[0]],
                                        reverse=True):
            tfs_active_reads[new_temp].update(tfs_reading_direct[new_temp])

            for td in new_temp_dependents[new_temp]:
                if temp_kinds.get(td, None) == TempKind.Global:
                    synthetic_global_dependents[new_temp].add(td)
                else:
                    if len(synthetic_global_dependents[td]) > 0:
                        synthetic_global_dependents[new_temp].update(synthetic_global_dependents[td])
                    for transitive_read_tf, transitive_read_els in tfs_active_reads.get(td, dict()).items():
                        get_or_compute(tfs_active_reads[new_temp], transitive_read_tf, lambda _: set()).update(transitive_read_els)

            assert len(synthetic_global_dependents[new_temp]) + len(tfs_active_reads[new_temp]) > 0, f"Temporary {new_temp} has 0 active reads"

            if len(synthetic_global_dependents[new_temp]) + len(tfs_active_reads[new_temp]) > 1:
                temp_kinds[new_temp] = TempKind.Global
            elif len(synthetic_global_dependents[new_temp]) == 1:
                temp_kinds[new_temp] = TempKind.Local
            else:
                assert len(tfs_active_reads[new_temp]) == 1
                if len(list(tfs_active_reads[new_temp].values())[0]) > 1:
                    temp_kinds[new_temp] = TempKind.Tile
                else:
                    temp_kinds[new_temp] = TempKind.Local

            assert new_temp in temp_kinds
            temp_kinds[new_temp] = temp_kinds[new_temp].clamp(promotion_predicate(new_temp))

        return ThornDef._ClassifyTempsResult(temp_kinds, tfs_active_reads, synthetic_global_dependents)

    def get_tensor_type(self, item: str | Symbol) -> tuple[str, tuple[Idx, ...], tuple[str, ...]]:
        var_name = str(item)
        assert var_name in self.declarations.keys(), f"Not a defined symbol {item}"
        base_name = self.var2base.get(var_name, None)
        if base_name is None:
            return "none", tuple(), tuple()  # scalar
        return base_name, self.declarations[base_name].indices, tuple(self.groups[base_name])

    def create_function(self,
                        name: str,
                        schedule_target: ScheduleTarget,
                        *,
                        schedule_before: Optional[Collection[str]] = None,
                        schedule_after: Optional[Collection[str]] = None,
                        intent_override: Optional[IntentOverride] = None) -> ThornFunction:
        tf = ThornFunction(name, schedule_target, self, schedule_before, schedule_after, intent_override)
        self.thorn_functions[name] = tf
        return tf

    def add_param(self, name: str, default: CactusParamDefaultType, desc: str, values: CactusParamValuesType = None) -> Symbol:
        self.params[name] = CactusParam(name, default, desc, values)
        return mk_symbol(name)

    def get_state(self) -> OrderedSet[IndexedBase]:
        return OrderedSet(self.declarations[k.replace("'", "")].base for k in self.rhs)

    # noinspection PyIncorrectDocstring
    def decl(self, basename: str, indices: Iterable[Idx], **kwargs: Unpack[CactusDeclOptionalArgs]) -> IndexedBase:
        """
        Declares a new scalar or tensor variable.

        :param basename: The symbolic name of the variable.
        :param indices: The indices of the variable. If the variable is a scalar, this should be an empty list.
        :param rhs: Specifies the right-hand side of an implied PDE with d(the_var)/dt on the left.
                    Setting this argument implies that the variable to be declared is a state variable.
        :param centering: The centering of the variable. Defaults to VVV.
        :param group_name: Override the Cactus group name this variable (or its components) will be declared under.
        :param from_thorn: Specifies the thorn wherein this variable is declared. If this argument is present,
                           EinsteinEngine will not produce any declarations for the variable in the current thorn.
        :param parity: Specifies the variable's reflectional symmetries.
        :param symmetries: Specifies the permutations of the variable's indices which are symmetric with the
                           canonical ordering given in the `indices` argument.
        :param anti_symmetries: Specifies the permutations of the variable's indices which are anti-symmetric
                                with the canonical ordering given in the `indices` argument.
        :param substitution_rule: Specifies the base substitution rule for the variable. If this argument is absent,
                                  a default substitution rule is applied. Pass `None` to suppress the default rule.
                                  The default substitution rule is determined as follows:
                                  1) If the variable is a scalar, the substitution rule is the identity function.
                                  2) If the variable is a tensor with `from_thorn` set to one of the thorns in
                                     `_xyz_subst_thorns`, then the substitution rule is `subst_tensor_xyz`.
                                  3) Otherwise, the substitution rule is `subst_tensor`.

        :return: A symbolic `IndexedBase` object which represents the declared variable.
        :raises DslException: If symmetries or anti-symmetries are applied to a scalar variable.
        """
        if basename in self.declarations:
            raise DslException(f"Symbol {basename} already declared.")

        indices_tup: tuple[Idx, ...] = tuple(indices)

        if (rhs := kwargs.get('rhs', None)) is not None:
            base_sym = rhs.args[0]
            assert isinstance(base_sym, Symbol)
            self.rhs[basename] = base_sym

        if (centering := kwargs.get('centering', None)) is None:
            centering = Centering.VVV

        self.centering[basename] = centering
        self.base2group[basename] = kwargs.get('group_name', basename)

        if (from_thorn := kwargs.get('from_thorn', None)) is not None:
            self.base2thorn[basename] = from_thorn

            if ((special_mappings := self._special_group_mappings.get(from_thorn, None)) is not None
                    and (special_group := special_mappings.get(basename, None)) is not None):
                self.base2group[basename] = special_group

        if (parity := kwargs.get('parity', None)) is not None:
            self.base2parity[basename] = parity

        if len(indices_tup) != 0:
            default_subst = subst_tensor_xyz if from_thorn in self._xyz_subst_thorns else subst_tensor
            kwargs['substitution_rule'] = kwargs.get('substitution_rule', default_subst)

        # Forward the full kwarg set intentionally: base decl only consumes a subset,
        # but declarations must retain the complete kwargs payload for downstream behavior.
        # MyPy unfortunately does not let us express this in the type system.
        the_symbol = super().decl(basename, indices_tup, **cast(Any, kwargs))
        return the_symbol

    def _add_symbol(self, the_symbol: Symbol, centering: Optional[Centering]) -> None:
        basename = str(the_symbol)

        assert basename not in self.declarations
        base = mk_indexed_base(basename, shape=())
        self.declarations[basename] = SymbolDeclaration(basename=basename, base=base, indices=tuple(), kwargs=cast(CactusDeclOptionalArgs, dict()))
        self.centering[basename] = centering
        self.base2group[basename] = basename

    def _on_substitution_symbol_created(self, indexed: Indexed, sub_symbol: Symbol) -> None:
        sub_name = str(sub_symbol)
        base_name = str(indexed.base)
        self.centering[sub_name] = self.centering[base_name]
        self.var2base[sub_name] = base_name
        if base_name not in self.groups:
            self.groups[base_name] = list()
        self.groups[base_name].append(sub_name)

    def expand_eqn(self, eqn: Eq) -> List[Eq]:
        result: List[Eq] = list()
        for tup in self.einstein_notation.expand_free_indices(eqn.lhs, self.symmetries):
            lhs, idxs, _ = tup
            result += [mk_eq(self._do_subs(lhs), self._do_subs(eqn.rhs, idxs))]
        return result


def _parity_of(p: int | Parity) -> Parity:
    if isinstance(p, Parity):
        return p
    elif p == -1:
        return Parity.Negative
    elif p == 1:
        return Parity.Positive
    else:
        raise DslException(f"Parity must be -1 or +1")


def parities(*args: Parity | int) -> TensorParity:
    if len(args) == 0:
        raise DslException("Parities must not be empty")

    if len(args) % 3 != 0:
        raise DslException('Parities must come in groups of 3')

    parities: list[SingleIndexParity] = list()
    for i in range(0, len(args), 3):
        pars = [_parity_of(p) for p in args[i:i + 3]]
        parities.append(SingleIndexParity(*pars))

    return TensorParity(parities)


class SourceAnnotations:
    loops: dict[int, str]
    eqns: dict[int, dict[Symbol, str]]

    def __init__(self) -> None:
        self.loops = defaultdict(str)
        self.eqns = defaultdict(lambda: defaultdict(str))


def _is_valid_c_identifier(s: str) -> bool:
    """Check if a string is a valid C identifier."""
    if not s:
        return False
    # C identifiers must start with a letter or underscore, followed by letters, digits, or underscores
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', s))
