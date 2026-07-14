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
from typing import Callable, Collection, Optional, cast, List, Unpack, Set, Union, Dict, Iterator, Iterable, Any, Sequence

from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy
from termcolor import colored
from sympy import Symbol, Expr, Idx, Indexed, Basic, IndexedBase, Eq  # type: ignore[import-untyped]
from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.common.sympywrap import (
    free_symbols, mk_eq, mk_indexed_base, mk_symbol
)
from EinsteinEngine.emit.ccl.schedule.schedule_tree import GroupOrFunction, ScheduleBlock
from EinsteinEngine.emit.tree import Centering, Identifier
from EinsteinEngine.frontend.dsl.use_indices import subst_tensor_xyz
from EinsteinEngine.intermediate.temp_kind import TempKind
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.emit.ccl.interface.interface_tree import TensorParity, Parity, SingleIndexParity
from EinsteinEngine.frontend.dsl.cactus.cactus_param import CactusParam, CactusParamValuesType, CactusParamDefaultType
from EinsteinEngine.frontend.dsl.use_indices import do_isub, subst_tensor
from EinsteinEngine.common.util import ScheduleBinEnum, ScheduleFrequency, wprint, OrderedSet

from EinsteinEngine.common.schedule_target import ScheduleTarget, safe_name
from EinsteinEngine.frontend.dsl.dsl_frontend import (
    DslFrontend,
    DslFrontendBakeOptions,
    SymbolDeclaration,
    SymbolDeclarationKwargs,
)
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend, DslFunctionFrontendBakeOptions
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


class ThornFunctionBakeOptions(DslFunctionFrontendBakeOptions, total=False):
    pass


class ThornDefBakeOptions(DslFrontendBakeOptions[ThornFunctionBakeOptions], total=False):
    pass


class ThornFunction(DslFunctionFrontend["ThornDef"]):
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
                 intent_override: Optional[IntentOverride] = None,
                 *,
                 auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                 auto_soft_split_predicate: Optional[Callable[[int], bool|SoftSplitRetainmentStrategy]] = None) -> None:
        self.thorn_def = thorn_def
        self.schedule_target = schedule_target
        self.schedule_before: Collection[str] = schedule_before or list()
        self.schedule_after: Collection[str] = schedule_after or list()
        super().__init__(name, thorn_def, intent_override, owner_name="ThornFunction",
                         auto_hard_split_predicate=auto_hard_split_predicate,
                         auto_soft_split_predicate=auto_soft_split_predicate)

        if isinstance(schedule_target, ScheduleBlock) and schedule_target.group_or_function is GroupOrFunction.Function:
            raise DslException("Cannot schedule into this schedule block because it is not a schedule group.")

    def _on_soft_split_symbol_merged(self, mangled_sym: Symbol, sym: Symbol) -> None:
        if (c := self.thorn_def.centering.get(str(sym))) is not None:
            self.thorn_def.centering[str(mangled_sym)] = c
        elif (sym_base := self.thorn_def.var2base.get(str(sym))) is not None:
            if (c := self.thorn_def.centering.get(sym_base)) is not None:
                self.thorn_def.centering[str(mangled_sym)] = c

    def show_tensor_types(self) -> None:
        keys: Set[str] = OrderedSet()
        for k1 in self.eqn_complex.inputs:
            keys.add(str(k1))
        for k2 in self.eqn_complex.outputs:
            keys.add(str(k2))
        for k in keys:
            group, indices, members = self.get_tensor_type(k)
            print(colored(k, "green"), "is a member of", colored(group, "green"), "with indices",
                  colored(indices, "cyan"), "and members", colored(members, "magenta"))

    def get_tensor_type(self, item: Union[str, Symbol]) -> tuple[str, tuple[Idx, ...], tuple[str, ...]]:
        return self.thorn_def.get_tensor_type(item)


class ThornDef(DslFrontend[CactusParam, CactusDeclOptionalArgs, ThornFunction]):
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
            coords: Optional[Sequence[str]] = None,
            derivative_stencil_width: int = 5
    ) -> None:
        super().__init__(
            dimensionality=dimensionality,
            coords=coords,
            derivative_stencil_width=derivative_stencil_width
        )

        if not _is_valid_c_identifier(name):
            raise DslException(f"Thorn name '{name}' is not a valid C identifier")

        self.arrangement = arr
        self.name = name
        self.base2group: Dict[str, str] = dict()
        self.groups: Dict[str, List[str]] = dict()
        self.centering: Dict[str, Optional[Centering]] = dict()
        self.rhs: Dict[str, Symbol] = dict()
        self.base2thorn: Dict[str, str] = dict()
        self.base2parity: Dict[str, TensorParity] = dict()
        self.synthetic_fns: dict[ScheduleTarget, set[ThornFunction]] = defaultdict(set)

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

    def _mk_default_bake_options(self) -> ThornDefBakeOptions:
        opts: ThornDefBakeOptions = DslFrontend._mk_default_dsl_frontend_bake_options()
        opts.update(self._mk_default_function_bake_options())  # type: ignore[typeddict-item]
        return opts

    def _mk_default_function_bake_options(self) -> ThornFunctionBakeOptions:
        return DslFunctionFrontend._mk_default_dsl_function_frontend_bake_options()

    def _global_cse_pre_materialization(
            self,
            substitutions: dict[Symbol, Expr],
            new_temp_dependencies: dict[Symbol, set[Symbol]],
            temp_kinds: dict[Symbol, TempKind]
    ) -> None:
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
                wprint(f"Could not infer a centering for temp {temp} -> {substitutions[temp]}; none of its dependencies have centerings. Defaulting to VVV.")
                centerings = {Centering.VVV}
            elif len(centerings) > 1:
                raise DslException(f"Could not infer a centering for temp {temp} -> {substitutions[temp]}; its dependencies have conflicting centerings {centerings}")

            assert len(centerings) == 1
            self.centering[str(temp)] = centerings.pop()

        for new_temp in substitutions.keys():
            compute_centerings(new_temp)

    def _global_cse_handle_global_temps(
            self,
            substitutions: dict[Symbol, Expr],
            temp_kinds: dict[Symbol, TempKind],
            tfs_active_reads: dict[Symbol, dict[ThornFunction, set[int]]],
            new_temp_dependencies: dict[Symbol, set[Symbol]]
    ) -> None:
        schedule_blocks: dict[Identifier, ScheduleBlock] = dict()
        schedule_bin_targets: dict[Symbol, dict[ScheduleBin, set[ThornFunction]]] = defaultdict(lambda: defaultdict(set))
        schedule_block_targets: dict[Symbol, dict[Identifier, set[ThornFunction]]] = defaultdict(lambda: defaultdict(set))

        for new_temp in substitutions.keys():
            if temp_kinds.get(new_temp, None) != TempKind.Global:
                continue

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
                # todo: I don't remember why this code is commented out. Need to figure out whether to restore or remove this hack.
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
                    f"synthetic_compute_{new_temp}_{safe_name(schedule_target)}",
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
                schedule_after = sorted(list(chain(*[[f"synthetic_compute_{td}_{safe_name(bin)}_group" for dep_bin in schedule_bin_targets[td].keys() if bin.is_colocated(dep_bin)] for td in find_all_global_deps(new_temp)])))
                if bin is ScheduleBin.PostInit:
                    schedule_after.append("ODESolvers_PostStep")  # Hack to ensure AMR and synchronization happen first
                mk_synthetic_fn(bin, sorted([f"{tf.name}_group" for tf in schedule_before_tfs]), schedule_after)

            if len(schedule_block_targets) > 0:
                wprint(f"Global temporary {new_temp} is accessed in at least one custom schedule block,"
                       f" on which EinsteinEngine cannot perform schedule analysis. The temporary will be recomputed for each"
                       f" custom block, perhaps redundantly.")

            for block, schedule_before_tfs in [(schedule_blocks[id], tfs) for id, tfs in schedule_block_targets[new_temp].items()]:
                schedule_after = sorted(list(chain(*[[f"synthetic_compute_{td}_{safe_name(block)}_group" for dep_block_name in schedule_block_targets[new_temp].keys() if block.name == dep_block_name] for td in new_temp_dependencies[new_temp] if temp_kinds.get(td, None) == TempKind.Global])))
                mk_synthetic_fn(block, sorted([f"{tf.name}_group" for tf in schedule_before_tfs]), schedule_after)

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
                        intent_override: Optional[IntentOverride] = None,
                        auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                        auto_soft_split_predicate: Optional[Callable[[int], bool|SoftSplitRetainmentStrategy]] = None) -> ThornFunction:
        tf = ThornFunction(name, schedule_target, self, schedule_before, schedule_after, intent_override,
                           auto_hard_split_predicate=auto_hard_split_predicate,
                           auto_soft_split_predicate=auto_soft_split_predicate)
        self.functions[name] = tf
        return tf

    def add_param(self, name: str, default: CactusParamDefaultType, desc: str, values: CactusParamValuesType = None) -> Symbol:
        self.params[name] = CactusParam(name, default, desc, values)
        return mk_symbol(name)

    def get_state(self) -> OrderedSet[IndexedBase]:
        return OrderedSet(self.declarations[k.replace("'", "")].indexed_base for k in self.rhs)

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
        self.declarations[basename] = SymbolDeclaration(symbol_name=basename, indexed_base=base, indices=tuple(), kwargs=cast(CactusDeclOptionalArgs, dict()))
        self.centering[basename] = centering
        self.base2group[basename] = basename

    def _on_substitution_symbol_created(self, indexed: Indexed, sub_symbol: Symbol) -> None:
        sub_name = str(sub_symbol)
        base_name = str(indexed.base)
        self.centering[sub_name] = self.centering[base_name]
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


def _is_valid_c_identifier(s: str) -> bool:
    """Check if a string is a valid C identifier."""
    if not s:
        return False
    # C identifiers must start with a letter or underscore, followed by letters, digits, or underscores
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', s))
