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
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Unpack

import EinsteinEngine.common.util as util
from sympy import Symbol, Expr, Idx, Indexed, IndexedBase, Matrix

from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.common.util import OrderedSet, pprint
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.intermediate.eqn_ordering import EqnOrderingFn, maximize_symbol_reuse
from EinsteinEngine.intermediate.eqnlist import EqnComplex, EqnList
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy, retain_none
from EinsteinEngine.intermediate.splitmaxxer import SplitMaxxer

if TYPE_CHECKING:
    from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend


class DslFunctionFrontendBakeOptions(TypedDict, total=False):
    do_madd: bool
    do_recycle_temporaries: bool
    splitmaxxing: bool
    ordering_fn: EqnOrderingFn
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

    def __init__(self, name: str, frontend: FrontendT, intent_override: Optional[IntentOverride], *, owner_name: str) -> None:
        self.name = name
        self.frontend = frontend
        self.source_annotations = SourceAnnotations()
        self.source_annotations.loops[0] = f"{self.name} loop 0"

        def set_eqn_annotation(loop_idx: int, key: Symbol, annotation: str) -> None:
            self.source_annotations.eqns[loop_idx][key] = annotation

        self.eqn_complex = EqnComplex(frontend.is_stencil, intent_override, set_eqn_annotation)
        self.been_baked = False
        self.been_late_baked = False
        self.intent_override = intent_override
        from EinsteinEngine.frontend.dsl.add_eqn_manager import AddEqnManager
        self._add_eqn_manager = AddEqnManager(
            frontend,
            lambda: self._eqn_list,
            lambda: self.been_baked,
            owner_name=owner_name,
        )

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
        if annotation is None:
            annotation = f"{self.name} loop {loop_idx}"

        if annotation.strip() != "":
            self.source_annotations.loops[loop_idx] = annotation

        self.eqn_complex._new_eqn_list()

    def soft_split(self, retainment_strategy: Optional[SoftSplitRetainmentStrategy] = None, annotation: Optional[str] = None) -> None:
        if self.been_baked:
            raise DslException("Cannot split loop because the EqnComplex has already been baked.")

        loop_idx = len(self.eqn_complex.eqn_lists)
        if annotation is None:
            annotation = f"{self.name} loop {loop_idx} (soft split)"

        if annotation.strip() != "":
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

            pprint(f"Rebaking {self.name} loop {loop_idx} after do_splitmaxxing...")
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
    def _mk_default_dsl_function_frontend_bake_options() -> DslFunctionFrontendBakeOptions:
        return {
            "do_madd": False,
            "do_recycle_temporaries": True,
            "splitmaxxing": False,
            "ordering_fn": maximize_symbol_reuse,
            "soft_split_retainment_strategy": retain_none(),
        }

    def _early_bake(self, **kwargs: Unpack[DslFunctionFrontendBakeOptions]) -> None:
        if self.been_baked:
            raise DslException("_early_bake should not be called more than once")
        pprint(f"Early Baking {self.name}...")

        options = self._mk_default_dsl_function_frontend_bake_options()
        options.update(kwargs)

        # Doing a first pass of complexity analysis for CSE
        for eqn_list in self.eqn_complex.eqn_lists:
            eqn_list._run_preliminary_complexity_analysis()

        if options["do_madd"]:
            self.madd()

        self.eqn_bake(options["ordering_fn"])

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
