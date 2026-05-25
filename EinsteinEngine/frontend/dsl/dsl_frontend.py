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

import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Set, NamedTuple, Iterable, TypedDict, Optional, cast, Unpack, Callable, Any, Generator, Never

# mypy: disable-error-code=no-redef
# The above line suppresses an unfortunate interaction between MyPy and multimethod.

import sympy as sy
from multimethod import multimethod
from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sympy import Symbol, Idx, Expr, IndexedBase, Indexed, Function, Basic, Matrix, ImmutableDenseMatrix, MatrixBase

from EinsteinEngine.common.sympywrap import (
    mk_symbol, Applier, UFunc, mk_function, mk_idxes, Pow, mk_idx, mk_zeros, do_subs, mk_indexed_base, mk_indexed,
    simplify
)
from EinsteinEngine.common.util import checked_cast, vprint, OrderedSet
from EinsteinEngine.frontend.frontend import Frontend
from EinsteinEngine.frontend.dsl.finite_difference import DivMakerVisitor, ApplyDivN
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.frontend.util import cse_isolate
from EinsteinEngine.frontend.dsl.use_indices import (
    is_lower_idx,
    idx_to_int,
    EinsteinNotationManager,
    IndexSubsVisitor,
    BaseIndexedSubstFnType,
    MkSubstType,
    subst_tensor,
)
from EinsteinEngine.frontend.dsl.symmetries import Sym
from EinsteinEngine.frontend.definitions import (
    D,
    div,
    no_idx,
    stencil,
    dummy,
    DD,
    DDI,
    DX,
    DY,
    DZ,
    DXI,
    DYI,
    DZI,
    noop,
    zero,
    one,
)
from EinsteinEngine.common.cse_optimization_level import CseOptimizationLevel
from EinsteinEngine.common.sympywrap import cse, free_symbols
from EinsteinEngine.intermediate.coef import coef
from EinsteinEngine.generators.sympy_complexity import SympyComplexityVisitor
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend, DslFunctionFrontendBakeOptions
from EinsteinEngine.intermediate.temp_kind import TempKind
from EinsteinEngine.intermediate.temporary_promotion_predicate import (
    OnePassTemporaryPromotionStrategy, TemporaryPromotionPredicate, TemporaryPromotionStrategy,
    TwoPassTemporaryPromotionStrategy, promote_all, promote_none
)
from EinsteinEngine.intermediate.eqn_ordering import EqnOrderingFn, maximize_symbol_reuse
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy
from EinsteinEngine.common.util import get_or_compute, pprint, verbose


class OverwriteSymbolRecord(NamedTuple):
    symbol: IndexedBase
    resolves_to: IndexedBase


class SymbolDeclarationKwargs(TypedDict, total=False):
    symmetries: list[tuple[Idx, Idx]]
    anti_symmetries: list[tuple[Idx, Idx]]
    substitution_rule: MkSubstType | None

class DslFrontendBakeOptions[FunctionFrontendBakeOptionsT: DslFunctionFrontendBakeOptions](TypedDict, total=False):
    # Frontend opts
    do_cse: bool
    cse_optimization_level: CseOptimizationLevel
    temporary_promotion_strategy: TemporaryPromotionStrategy

    # Function frontend default opts
    do_madd: bool
    do_recycle_temporaries: bool
    splitmaxxing: bool
    ordering_fn: EqnOrderingFn
    soft_split_retainment_strategy: SoftSplitRetainmentStrategy

    # Overrides for function frontend default opts
    functions: dict[str, FunctionFrontendBakeOptionsT]


@dataclass
class SymbolDeclaration[KwargsType: SymbolDeclarationKwargs]:
    symbol_name: str
    indexed_base: IndexedBase
    indices: tuple[Idx, ...]
    kwargs: KwargsType


class ClassifyTempsResult[FunctionFrontendT](NamedTuple):
    temp_kinds: dict[Symbol, TempKind]
    tfs_active_reads: dict[Symbol, dict[FunctionFrontendT, set[int]]]
    synthetic_global_dependents: dict[Symbol, set[Symbol]]


class DslFrontend[ParamDataT, SymbolDeclarationKwargsT: SymbolDeclarationKwargs, FunctionFrontendT: DslFunctionFrontend[Any]](Frontend):
    dimensionality: int
    declarations: dict[str, SymbolDeclaration[SymbolDeclarationKwargsT]]
    coords: list[Symbol]
    params: dict[str, ParamDataT]
    var2base: dict[str, str]
    functions: dict[str, FunctionFrontendT]
    tile_temporaries: OrderedSet[Symbol]
    global_temporaries: OrderedSet[Symbol]

    overwrite_symbols: dict[str, OverwriteSymbolRecord]

    is_stencil: dict[UFunc, bool]  # Obsolesce sometime?

    div_makers: dict[str, DivMakerVisitor]
    apply_div: Applier
    unary_custom_stencils: dict[tuple[UFunc, Idx], Expr]
    binary_custom_stencils: dict[tuple[UFunc, Idx, Idx], Expr]
    ufunc_arities: dict[str, int]

    einstein_notation: EinsteinNotationManager
    subs: dict[Indexed | IndexedBase, Expr]
    symmetries: Sym

    _unique_name_counter: int

    def __init__(self, *, dimensionality: int = 3, derivative_stencil_order: int = 5):
        super().__init__()
        self.dimensionality = dimensionality
        self.declarations = dict()
        self.coords = list()
        self.params = dict()
        self.var2base = dict()
        self.functions = dict()
        self.tile_temporaries = OrderedSet()
        self.global_temporaries = OrderedSet()

        self.overwrite_symbols = dict()

        self.is_stencil = {
            mk_function("stencil"): True
        }

        self.unary_custom_stencils = dict()
        self.binary_custom_stencils = dict()
        self.ufunc_arities = dict()
        self._set_derivative_stencil(derivative_stencil_order)

        self.div_makers = dict()
        self.div_makers["div"] = DivMakerVisitor(div)
        self.div_makers["D"] = DivMakerVisitor(D)

        for dmv in self.div_makers.values():
            dmv.params = self._mk_param_set()

        self.einstein_notation = EinsteinNotationManager(dimensionality=dimensionality)
        self.symmetries = Sym()
        self.subs = dict()
        self._populate_globals()

        self._unique_name_counter = 0

    # Wart: Proper signature is **kwargs: Unpack[SymbolDeclarationKwargsT] but MyPy does not support this.
    def decl(self, basename: str, indices: Iterable[Idx], **kwargs: Unpack[SymbolDeclarationKwargs]) -> IndexedBase:
        indices_tup: tuple[Idx, ...] = tuple(indices)
        the_symbol = mk_indexed_base(basename, shape=tuple([self.dimensionality] * len(indices_tup)))
        indexed_symbol = mk_indexed(the_symbol, *indices_tup) if len(indices_tup) != 0 else None

        if basename in self.declarations:
            raise DslException(f"Symbol {basename} already declared.")

        self.declarations[basename] = SymbolDeclaration(
            symbol_name=basename,
            indexed_base=the_symbol,
            indices=indices_tup,
            kwargs=cast(SymbolDeclarationKwargsT, kwargs),
        )

        if (symmetries := kwargs.get('symmetries', None)) is not None:
            if indexed_symbol is None:
                raise DslException('Symmetries cannot be applied to a scalar variable')
            for sym in symmetries:
                self._add_sym(indexed_symbol, *sym, sgn=1)

        if (anti_symmetries := kwargs.get('anti_symmetries', None)) is not None:
            if indexed_symbol is None:
                raise DslException('Anti-symmetries cannot be applied to a scalar variable')
            for a_sym in anti_symmetries:
                self._add_sym(indexed_symbol, *a_sym, sgn=-1)

        if indexed_symbol is not None and (substitution_rule := kwargs.get('substitution_rule', subst_tensor)) is not None:
            self.add_substitution_rule(indexed_symbol, substitution_rule)

        return the_symbol

    def _unique_name(self, prefix: str = 'tmp') -> Generator[str, Never, Never]:
        while True:
            r = f'{prefix}_{self._unique_name_counter}'
            self._unique_name_counter += 1
            yield r

    def _decl_scalar(self, basename: str) -> Symbol:
        ret = mk_indexed_base(basename, tuple())
        self.declarations[basename] = SymbolDeclaration(
            symbol_name=basename,
            indexed_base=ret,
            indices=tuple(),
            kwargs=cast(SymbolDeclarationKwargsT, dict()),
        )

        base = ret.args[0]
        assert isinstance(base, Symbol)
        return base

    def decl_fun(self, fn_name: str, args: int = 1, is_stencil: bool = False) -> UFunc:
        fun = mk_function(fn_name)
        self.ufunc_arities[fn_name] = args
        self.is_stencil[fun] = is_stencil
        return fun

    def get_free_indices(self, expr: Expr) -> OrderedSet[Idx]:
        it = self.einstein_notation.check_indices(expr, self.declarations)
        return it.free

    @staticmethod
    def find_indices(foo: Basic) -> list[Idx]:
        ret: list[Idx] = list()
        if type(foo) in [div, D]:
            ret = DslFrontend.find_indices(foo.args[0])
        for arg in foo.args[1:]:
            assert isinstance(arg, Idx)
            ret.append(arg)
        return ret

    @staticmethod
    def get_indices(expr: Expr) -> list[Idx]:
        out: list[Idx] = list()
        if type(expr) in [div, D]:
            for arg in expr.args[0].args[1:]:
                assert isinstance(arg, Idx)
                out.append(arg)
        assert isinstance(expr, Indexed)
        for arg in expr.args[1:]:
            assert isinstance(arg, Idx)
            out.append(arg)
        return out

    def get_coords(self) -> list[Symbol]:
        return self.coords

    def get_params(self) -> Set[str]:
        return OrderedSet(self.params)

    def grid_variables(self) -> set[Symbol]:
        gv: set[Symbol] = set()
        for function_frontend in self.functions.values():
            gv |= function_frontend.eqn_complex._grid_variables()
        return gv

    @staticmethod
    def _mk_default_dsl_frontend_bake_options() -> DslFrontendBakeOptions[DslFunctionFrontendBakeOptions]:
        opts: DslFrontendBakeOptions[DslFunctionFrontendBakeOptions] = {
            "do_cse": True,
            "cse_optimization_level": CseOptimizationLevel.Optimal,
            "temporary_promotion_strategy": promote_none(),
            "ordering_fn": maximize_symbol_reuse,
            "functions": dict(),
        }
        opts.update(DslFunctionFrontend._mk_default_dsl_function_frontend_bake_options())  # type: ignore[typeddict-item]
        return opts

    def _mk_default_bake_options(self) -> DslFrontendBakeOptions[DslFunctionFrontendBakeOptions]:
        return self._mk_default_dsl_frontend_bake_options()

    def _mk_default_function_bake_options(self) -> DslFunctionFrontendBakeOptions:
        return DslFunctionFrontend._mk_default_dsl_function_frontend_bake_options()

    def _mk_function_bake_options(
            self, opts: DslFrontendBakeOptions[DslFunctionFrontendBakeOptions]
    ) -> dict[str, DslFunctionFrontendBakeOptions]:
        my_tf_opts: dict[str, DslFunctionFrontendBakeOptions] = dict()

        for tf in self.functions.values():
            tf_opts = self._mk_default_function_bake_options()
            tf_opts.update(cast(DslFunctionFrontendBakeOptions, opts))
            if "functions" in opts and tf.name in opts["functions"]:
                tf_opts.update(opts["functions"][tf.name])
            my_tf_opts[tf.name] = tf_opts

        return my_tf_opts

    def bake(self, **opts: Unpack[DslFrontendBakeOptions[DslFunctionFrontendBakeOptions]]) -> None:
        my_opts = self._mk_default_bake_options()
        my_opts.update(opts)  # type: ignore[typeddict-item]
        my_tf_opts = self._mk_function_bake_options(my_opts)

        for tf in self.functions.values():
            assert tf.name in my_tf_opts, f"Function '{tf.name}' not found in my_tf_opts"
            tf._early_bake(**my_tf_opts[tf.name])

        if my_opts["do_cse"]:
            pprint("Performing CSE...")
            self._do_global_cse(my_opts["temporary_promotion_strategy"], my_opts["cse_optimization_level"])

        for tf in self.functions.values():
            if tf.needs_merge():
                pprint(f"Merging soft splits in {tf.name}...")
                tf.merge_soft_splits(my_tf_opts[tf.name]["soft_split_retainment_strategy"])

        for tf in self.functions.values():
            if tf.name not in my_tf_opts:  # Must be a synthetic function
                tf._late_bake()
            else:
                if my_tf_opts[tf.name]["splitmaxxing"]:
                    tf._do_splitmaxxing()
                tf._late_bake(**my_tf_opts[tf.name])

    @staticmethod
    def _classify_temps(
            new_temp_dependents: dict[Symbol, set[Symbol]],
            promotion_predicate: TemporaryPromotionPredicate,
            substitutions: dict[Symbol, Expr],
            tfs_reading_direct: dict[Symbol, dict[FunctionFrontendT, set[int]]],
            substitutions_order: dict[Symbol, int]
    ) -> ClassifyTempsResult[FunctionFrontendT]:
        temp_kinds: dict[Symbol, TempKind] = dict()
        tfs_active_reads: dict[Symbol, dict[FunctionFrontendT, set[int]]] = defaultdict(lambda: dict())
        synthetic_global_dependents: dict[Symbol, set[Symbol]] = defaultdict(set)

        for new_temp, _ in sorted(substitutions.items(),
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

        return ClassifyTempsResult(temp_kinds, tfs_active_reads, synthetic_global_dependents)

    def _global_cse_pre_materialization(
            self,
            substitutions: dict[Symbol, Expr],
            new_temp_dependencies: dict[Symbol, set[Symbol]],
            temp_kinds: dict[Symbol, TempKind]
    ) -> None:
        pass

    def _global_cse_handle_global_temps(
            self,
            substitutions: dict[Symbol, Expr],
            temp_kinds: dict[Symbol, TempKind],
            tfs_active_reads: dict[Symbol, dict[FunctionFrontendT, set[int]]],
            new_temp_dependencies: dict[Symbol, set[Symbol]]
    ) -> None:
        for new_temp in substitutions.keys():
            if temp_kinds.get(new_temp, None) == TempKind.Global:
                self.global_temporaries.add(new_temp)

    def _do_global_cse(
            self,
            promotion_strategy: TemporaryPromotionStrategy = promote_all(),
            optimization_level: CseOptimizationLevel = CseOptimizationLevel.Optimal
    ) -> None:
        for tf in self.functions.values():
            if tf.been_late_baked:
                raise DslException(f"Cannot do_global_cse on function {tf.name} because it has already undergone late baking.")

        grid_vars = self.grid_variables()

        tf_names = sorted(self.functions.keys())
        old_tf_shapes: dict[str, list[int]] = dict()
        old_tf_lhses: dict[str, list[list[Symbol]]] = dict()
        old_tf_rhses: dict[str, list[list[Expr]]] = dict()

        for tf_name in tf_names:
            old_tf_shapes[tf_name] = list()
            old_tf_lhses[tf_name] = list()
            old_tf_rhses[tf_name] = list()

        for tf_name, tf in sorted(self.functions.items(), key=lambda kv: tf_names.index(kv[0])):
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

        new_temp_direct_reads: dict[Symbol, dict[str, set[int]]] = {sym: dict() for sym in substitutions.keys()}
        new_temp_dependencies: dict[Symbol, set[Symbol]] = {sym: set() for sym in substitutions.keys()}
        new_temp_dependents: dict[Symbol, set[Symbol]] = {sym: set() for sym in substitutions.keys()}

        global_eqn_idx = 0
        for _, tf in enumerate(sorted(self.functions.values(), key=lambda tf: tf_names.index(tf.name))):
            tf_name = tf.name

            for el_idx, el_shape in enumerate(old_tf_shapes[tf_name]):
                eqn_list = tf.eqn_complex.eqn_lists[el_idx]
                el_new_free_symbols: set[Symbol] = set(chain(*[free_symbols(rhs) for rhs in new_rhses[global_eqn_idx:global_eqn_idx + el_shape]]))
                new_temps = el_new_free_symbols.intersection(substitutions.keys())

                for new_temp, temp_rhs in [(new_temp, substitutions[new_temp]) for new_temp in new_temps]:
                    assert new_temp not in eqn_list.inputs
                    assert new_temp not in eqn_list.params
                    assert new_temp not in eqn_list.outputs
                    assert new_temp not in eqn_list.eqns

                    get_or_compute(new_temp_direct_reads[new_temp], tf_name, lambda _: set()).add(el_idx)

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

        tfs_reading_direct: dict[Symbol, dict[FunctionFrontendT, set[int]]] = defaultdict(lambda: dict())

        for new_temp in substitutions.keys():
            tf_names_reading_direct = set(new_temp_direct_reads[new_temp].keys())

            for tf_name in tf_names_reading_direct:
                els_reading_direct = new_temp_direct_reads[new_temp].get(tf_name, set())
                tfs_reading_direct[new_temp][self.functions[tf_name]] = els_reading_direct

        complexities: dict[Symbol, int] = dict()
        for tf in self.functions.values():
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

        temp_kinds, tfs_active_reads, _ = self._classify_temps(
            new_temp_dependents,
            promotion_predicate,
            substitutions,
            tfs_reading_direct,
            substitutions_order
        )

        if two_pass:
            assert isinstance(promotion_strategy, TwoPassTemporaryPromotionStrategy)
            promotion_predicate = promotion_strategy(complexities, temp_kinds)

            temp_kinds, tfs_active_reads, _ = self._classify_temps(
                new_temp_dependents,
                promotion_predicate,
                substitutions,
                tfs_reading_direct,
                substitutions_order
            )

        self._global_cse_pre_materialization(substitutions=substitutions, new_temp_dependencies=new_temp_dependencies,
                                             temp_kinds=temp_kinds)

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
        for tf in sorted(self.functions.values(), key=lambda tf: tf_names.index(tf.name)):
            tf_name = tf.name
            for el_idx, el_shape in enumerate(old_tf_shapes[tf_name]):
                eqn_list = tf.eqn_complex.eqn_lists[el_idx]

                for lhs in old_tf_lhses[tf_name][el_idx]:
                    assert lhs in eqn_list.eqns
                    eqn_list.eqns[lhs] = new_rhses[global_eqn_idx]
                    global_eqn_idx += 1

        for new_temp in substitutions.keys():
            if new_temp not in temp_kinds or temp_kinds[new_temp] in [TempKind.Inline, TempKind.Global]:
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

        self._global_cse_handle_global_temps(
            substitutions=substitutions,
            temp_kinds=temp_kinds,
            tfs_active_reads=tfs_active_reads,
            new_temp_dependencies=new_temp_dependencies
        )

        for tf in self.functions.values():
            for idx, eqn_list in enumerate(tf.eqn_complex.eqn_lists):
                pprint(f"Rebaking {tf.name} loop {idx} after CSE...")

                # If the tf needs a merge, set force_fast because another (slow) rebake will succeed CSE.
                eqn_list.bake(force_rebake=True, force_fast=tf.needs_merge())

                if verbose():
                    eqn_list.dump()

    def overwrite(self, sym: IndexedBase) -> IndexedBase:
        if str(sym) not in self.declarations or self.declarations[str(sym)].indexed_base != sym:
            raise DslException(f"Cannot overwrite symbol {sym} which is not declared")

        orig_sym = sym if str(sym) not in self.overwrite_symbols else self.overwrite_symbols[str(sym)].resolves_to
        decl = self.declarations[str(orig_sym)]
        sym_prime = self.decl(f"{sym}'", decl.indices, **cast(Any, decl.kwargs))

        self.overwrite_symbols[str(sym_prime)] = OverwriteSymbolRecord(sym_prime, orig_sym)

        return sym_prime

    @multimethod
    def add_substitution_rule(self, indexed: Indexed, f: Callable[[Indexed, int, int], Expr]) -> None:
        def f2(ix: Indexed, *n: int) -> Expr:
            return f(ix, n[0], n[1])

        self.add_substitution_rule(indexed, f2)

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: Callable[[Indexed, int], Expr]) -> None:
        def f2(ix: Indexed, *n: int) -> Expr:
            return f(ix, n[0])

        self.add_substitution_rule(indexed, f2)

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: Callable[[Indexed, int, int, int], Expr]) -> None:
        def f2(ix: Indexed, *n: int) -> Expr:
            return f(ix, n[0], n[1], n[2])

        self.add_substitution_rule(indexed, f2)

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: BaseIndexedSubstFnType = subst_tensor) -> None:
        iter_var = indexed

        for tup in self.einstein_notation.expand_free_indices(iter_var, self.symmetries):
            indexed_sym, _, _ = tup
            assert isinstance(indexed_sym, Indexed)

            idxs = indexed_sym.indices
            sub_val_ = f(indexed_sym, *idxs)

            if sub_val_.is_Number or sub_val_.is_Function:
                pass
            else:
                assert isinstance(sub_val_, Symbol)
                base_name = str(indexed.base)
                sub_val_name = str(sub_val_)
                self.declarations[sub_val_name] = SymbolDeclaration(
                    symbol_name=sub_val_name,
                    indexed_base=mk_indexed_base(sub_val_name, tuple()),
                    indices=tuple(),
                    kwargs=cast(SymbolDeclarationKwargsT, dict())
                )
                self.var2base[sub_val_name] = base_name
                self.subs[indexed_sym] = sub_val_
                self._on_substitution_symbol_created(indexed_sym, sub_val_)
                self._on_substitution_mapping(indexed_sym, sub_val_)

    @add_substitution_rule.register
    def _(self, indexed_base: IndexedBase, f: Expr) -> None:
        rhs = simplify(self._do_subs(f, idx_subs={}))
        self.subs[indexed_base] = rhs
        self._on_substitution_mapping(indexed_base, rhs)

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: Expr) -> None:
        iter_var = indexed

        if self.get_free_indices(iter_var) != self.get_free_indices(f):
            raise Exception(f"Free indices of '{indexed}' and '{f}' do not match.")
        for tup in self.einstein_notation.expand_free_indices(iter_var, self.symmetries):
            indexed_sym, ind_rep, _ = tup
            assert isinstance(indexed_sym, Indexed)
            self.subs[indexed_sym] = simplify(self._do_subs(f, idx_subs=ind_rep))
            self._on_substitution_mapping(indexed_sym, self.subs[indexed_sym])

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: ImmutableDenseMatrix) -> None:
        self._mk_subst_matrix(indexed, f)

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: MatrixBase) -> None:
        self._mk_subst_matrix(indexed, f)

    def _mk_subst_matrix(self, indexed: Indexed, f: MatrixBase) -> None:
        iter_var = indexed
        set_matrix = f
        for tup in self.einstein_notation.expand_free_indices(iter_var, self.symmetries):
            out, idx_rep, _ = tup
            assert isinstance(out, Indexed)
            arr_idxs = tuple([idx_to_int(x) for x in out.indices])
            n_array = len(arr_idxs)
            res = simplify(set_matrix[arr_idxs[0:2]])
            if n_array >= 3:
                res = self._do_subs(res, idx_subs=idx_rep)
            self.subs[out] = res
            self._on_substitution_mapping(out, self.subs[out])

    def _on_substitution_symbol_created(self, indexed: Indexed, sub_symbol: Symbol) -> None:
        pass

    def _on_substitution_mapping(self, source: Indexed | IndexedBase, target: Expr) -> None:
        if isinstance(source, IndexedBase):
            vprint(colorize(source, "cyan"), colorize("->", "green"), colorize(target, "yellow"))
        else:
            vprint(colorize(source, "red"), colorize("->", "magenta"), colorize(target, "cyan"))

    def mk_coords(self, with_time: bool = False) -> list[Symbol]:
        # Note that x, y, and z are special symbols
        if self.dimensionality == 3:
            if with_time:
                self.coords = [self._decl_scalar("t"), self._decl_scalar("x"), self._decl_scalar("y"),
                               self._decl_scalar("z")]
            else:
                self.coords = [self._decl_scalar("x"), self._decl_scalar("y"), self._decl_scalar("z")]
        elif self.dimensionality == 4:
            self.coords = [self._decl_scalar("t"), self._decl_scalar("x"), self._decl_scalar("y"),
                           self._decl_scalar("z")]
        else:
            raise DslException(f"Unsupported dimensionality {self.dimensionality}")
        return self.coords

    def _set_derivative_stencil(self, n: int) -> None:
        assert n % 2 == 1, "n must be odd"
        assert n > 1, "n must be > 1"
        self.apply_div = ApplyDivN(
            n, self.unary_custom_stencils, self.binary_custom_stencils, self.ufunc_arities, self.dimensionality
        )

    def _mk_param_set(self) -> Set[Symbol]:
        ret: Set[Symbol] = set()
        for k in self.params:
            ret.add(mk_symbol(k))
        return ret

    def get_matrix(self, ind: Indexed) -> Matrix:
        values: dict[Idx, Idx] = dict()
        result = mk_zeros(*tuple([self.dimensionality] * (len(ind.args) - 1)))
        ind_args: list[Idx] = [checked_cast(x, Idx) for x in ind.args[1:]]
        while self.einstein_notation.incr(ind_args, values):
            arr_idxs = tuple([idx_to_int(checked_cast(do_subs(x, values), Idx)) for x in ind_args])
            r = self._do_subs(ind, idx_subs=values)
            result[arr_idxs] = r
        return result

    def find_symmetries(self, foo: Basic) -> list[tuple[int, int, int]]:
        m_sym_list: list[tuple[int, int, int]] = list()
        if foo.is_Function and hasattr(foo, "name") and foo.name in ["div", "D"]:
            if len(foo.args) == 3:
                foo_arg1 = len(foo.args[0].args) - 1
                foo_arg2 = foo_arg1 + 1
                m_sym: tuple[int, int, int] = (foo_arg1, foo_arg2, 1)
                m_sym_list += [m_sym]
                m_sym_list += self.find_symmetries(foo.args[0])
            elif len(foo.args) == 2:
                m_sym_list += self.find_symmetries(foo.args[0])
            else:
                assert False, "Only handle 1st and 2nd derivatives"
        elif isinstance(foo, Indexed):
            k = foo.base
            return self.symmetries.sd.get(k, list())
        return m_sym_list

    def _populate_globals(self) -> None:
        ui, li = self.einstein_notation.mk_pair('i')
        uj, lj = self.einstein_notation.mk_pair('j')
        uk, lk = self.einstein_notation.mk_pair('k')
        ua, la = self.einstein_notation.mk_pair('a')
        ub, lb = self.einstein_notation.mk_pair('b')
        uc, lc = self.einstein_notation.mk_pair('c')
        ud, ld = self.einstein_notation.mk_pair('d')
        u0, l0 = self.einstein_notation.mk_pair('0')
        u1, l1 = self.einstein_notation.mk_pair('1')
        u2, l2 = self.einstein_notation.mk_pair('2')
        u3, l3 = self.einstein_notation.mk_pair('3')
        u4, l4 = self.einstein_notation.mk_pair('4')
        u5, l5 = self.einstein_notation.mk_pair('5')
        up_indices = u0, u1, u2, u3, u4, u5
        down_indices = l0, l1, l2, l3, l4, l5

        globals().update(locals())

    def mk_pair(self, s: str | None = None) -> tuple[Idx, Idx]:
        return self.einstein_notation.mk_pair(s)

    def _add_sym(self, tens: Indexed, ix1: Idx, ix2: Idx, sgn: int = 1) -> None:
        i1 = -1
        i2 = -1
        for i in range(1, len(tens.args)):
            if tens.args[i] == ix1:
                i1 = i - 1
            if tens.args[i] == ix2:
                i2 = i - 1
        assert i1 != -1, f"Index {ix1} not in {tens}"
        assert i2 != -2, f"Index {ix2} not in {tens}"
        assert i1 != i2, f"Index {ix1} cannot be symmetric with itself in {tens}"
        if i1 > i2:
            i1, i2 = i2, i1
        self.symmetries.add(tens.base, i1, i2, sgn)

    def _do_div(self, expr: Expr) -> Expr:
        params = self._mk_param_set()
        r = expr
        for v in self.div_makers.values():
            v.params = params
            r = v.visit(r, no_idx)
        return r

    def _do_subs(self, arg: Expr, idx_subs: Optional[dict[Idx, Idx]] = None) -> Expr:
        isub = IndexSubsVisitor(self.subs, idx_subs)
        arg1 = arg

        max_iter = 100
        for _ in range(max_iter):
            new_arg = arg1
            new_arg = self.einstein_notation.expand_contracted_indices(new_arg, self.symmetries)
            new_arg = cast(Expr, self.symmetries.apply(new_arg))

            new_arg = isub.visit(new_arg)
            new_arg = self._do_div(new_arg)
            if new_arg == arg1:
                return new_arg
            arg1 = new_arg

        raise DslException(f"Failed to exhaustively substitute {arg} after {max_iter} iterations.")

    @multimethod
    def mk_stencil(self, func_name: str, idx: Idx, expr: Expr) -> UFunc:
        result = self.mk_stencil(func_name, expr, [idx])
        assert isinstance(result, UFunc)
        self.div_makers[func_name] = DivMakerVisitor(result)
        return result

    @mk_stencil.register
    def _mk_stencil(self, func_name: str, idx_a1: Idx, idx_a2: Idx, expr_a: Expr,
                    idx_b1: Idx, idx_b2: Idx, expr_b: Expr) -> UFunc:
        self.mk_stencil(func_name, idx_a1, idx_a2, expr_a)
        result = self.mk_stencil(func_name, idx_b1, idx_b2, expr_b)
        assert isinstance(result, UFunc)
        return result

    @mk_stencil.register
    def _mk_stencil(self, func_name: str, idx1: Idx, idx2: Idx, expr: Expr) -> UFunc:
        result = self.mk_stencil(func_name, expr, [idx1, idx2])
        assert isinstance(result, UFunc)
        self.div_makers[func_name] = DivMakerVisitor(result)
        return result

    @mk_stencil.register
    def _mk_stencil(self, func_name: str, expr: Expr, idx_list: list[Idx]) -> UFunc:
        @multimethod
        def mk_sten(idx_map: dict[Idx, Idx], expr_: Function) -> Expr:
            assert self.dimensionality == 3
            l0, l1, l2 = mk_idxes('l0 l1 l2')

            if expr_.func == stencil:
                if len(expr_.args) != 1:
                    raise DslException(expr_)
                arg = mk_sten(idx_map, expr_.args[0])
                c0 = coef(l0, arg)
                c1 = coef(l1, arg)
                c2 = coef(l2, arg)
                ret = stencil(dummy, c0, c1, c2)
                assert isinstance(ret, Expr)
                return ret
            elif expr_.func == DD:
                if len(expr_.args) != 1:
                    raise DslException(expr_)
                arg = mk_sten(idx_map, expr_.args[0])
                if arg == l0:
                    return DX
                elif arg == l1:
                    return DY
                elif arg == l2:
                    return DZ
                assert False
            elif expr_.func == DDI:
                if len(expr_.args) != 1:
                    raise DslException(expr_)
                arg = mk_sten(idx_map, expr_.args[0])
                if arg == l0:
                    return DXI
                elif arg == l1:
                    return DYI
                elif arg == l2:
                    return DZI
                assert False
            elif expr_.func == noop:
                arg = mk_sten(idx_map, expr_.args[0])
                retv: Expr = noop(arg)
                return retv
            else:
                raise DslException("Bad Func")

        @mk_sten.register
        def _mk_sten(_idx_map: dict[Idx, Idx], expr_: sy.Float) -> Expr:
            return expr_

        @mk_sten.register
        def _mk_sten(_idx_map: dict[Idx, Idx], expr_: sy.Integer) -> Expr:
            return expr_

        @mk_sten.register
        def _mk_sten(_idx_map: dict[Idx, Idx], expr_: sy.Rational) -> Expr:
            return expr_

        @mk_sten.register
        def _mk_sten(idx_map: dict[Idx, Idx], expr_: sy.Pow) -> Expr:
            result: Expr = Pow(mk_sten(idx_map, expr_.args[0]), expr_.args[1])
            return result

        @mk_sten.register
        def _mk_sten(idx_map: dict[Idx, Idx], expr_: Idx) -> Expr:
            retval = idx_map.get(expr_, expr_)
            return retval

        @mk_sten.register
        def _mk_sten(idx_map: dict[Idx, Idx], expr_: sy.Add) -> Expr:
            ret = zero
            for a in expr_.args:
                term = mk_sten(idx_map, a)
                ret += term
            return ret

        @mk_sten.register
        def _mk_sten(idx_map: dict[Idx, Idx], expr_: sy.Mul) -> Expr:
            ret = one
            for a in expr_.args:
                term = mk_sten(idx_map, a)
                ret *= term
            return ret

        func = mk_function(func_name)

        if len(idx_list) == 1 or (len(idx_list) == 2 and idx_list[0] == idx_list[1]):
            idx = idx_list[0]
            is_down_idx = is_lower_idx(idx)
            for i in range(self.dimensionality):
                idx0 = mk_idx(f'l{i}') if is_down_idx else mk_idx(f'u{i}')
                result = mk_sten({idx: idx0}, expr)
                self.unary_custom_stencils[(func, idx0)] = result
        elif len(idx_list) == 2:
            idx1 = idx_list[0]
            idx2 = idx_list[1]
            is_down_idx1 = is_lower_idx(idx1)
            is_down_idx2 = is_lower_idx(idx2)
            for i in range(self.dimensionality):
                idx10 = mk_idx(f'l{i}') if is_down_idx1 else mk_idx(f'u{i}')
                for j in range(self.dimensionality):
                    if i == j:
                        continue
                    idx20 = mk_idx(f'l{j}') if is_down_idx2 else mk_idx(f'u{j}')
                    result = mk_sten({idx1: idx10, idx2: idx20}, expr)
                    self.binary_custom_stencils[(func, idx10, idx20)] = result

        return func

# STEVE: What does this do and what does its name mean?
def mk_mk_subst(s: str) -> str:
    next_sub = 'a'
    pos = 0
    new_s = ""
    for g in re.finditer(r'\b([ul])([0-9])\b', s):
        new_s += s[pos:g.start()]
        pos = g.end()
        up_down = g.group(1)
        _index = g.group(2)
        new_s += up_down
        new_s += next_sub
        next_sub = chr(ord(next_sub) + 1)
    new_s += s[pos:]
    return new_s
