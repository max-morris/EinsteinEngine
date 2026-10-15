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
A frontend for GRChombo compute classes: a C++ class that GRChombo's BoxLoops::loop calls once per cell (or per SIMD
vector of cells) through `template <class data_t> void compute(Cell<data_t>) const`.

The generated class is meant to replace an existing GRChombo compute class (e.g. CCZ4RHS) without editing the
generated code, so the frontend describes the class's interface in GRChombo's terms:

* the constructor signature (`ctor_args`), which should match the class being replaced;
* how each DSL parameter is initialized from the constructor arguments (`add_param(..., init=...)`);
* the grid spacing expression (`dx`), from which the DXI/DYI/DZI members are formed;
* constructor checks for configurations the generated code does not implement (`ctor_checks`).

Grid variables are GRChombo state variables: a variable `X` (or a component `X11` of a tensor) is read from the
input FArrayBox component with enum `c_X` (`enum_prefix`). A variable declared with `rhs=R` is a state variable whose
time derivative is the variable R; a component of R is written to the output FArrayBox component of the matching
state component, as GRChombo's `Cell::store_vars` does.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Iterable, Unpack, Any

from sympy import Idx, IndexedBase, Indexed, Symbol

from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.common.sympywrap import mk_symbol
from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend, SymbolDeclarationKwargs
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend
from EinsteinEngine.frontend.dsl.use_indices import idx_to_int
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy
from EinsteinEngine.intermediate.split_locus import SplitLocus
from EinsteinEngine.frontend.dsl.dsl_exception import DslException


def subst_tensor_grchombo(sym: Indexed, *_idxs: int) -> Symbol:
    """
    Names a tensor component the way GRChombo's CCZ4 variables are named: the base name followed by the 1-based
    component indices, without distinguishing up and down indices (h11, h12, Gamma1, shift3, ...).
    """
    name = str(sym.base)
    for ind in sym.args[1:]:
        assert isinstance(ind, Idx)
        name += str(idx_to_int(ind) + 1)
    return mk_symbol(name)


class GRChomboDeclKwargs(SymbolDeclarationKwargs, total=False):
    # The time derivative of this state variable. Components of `rhs` are written to the output components of the
    # matching components of this variable.
    rhs: IndexedBase


@dataclass
class GRChomboParam:
    name: str
    cpp_type: str  # 'double', 'int' or 'bool'
    init: str  # C++ expression in terms of the constructor arguments


@dataclass
class GRChomboCtorArg:
    cpp_type: str
    name: str
    default: Optional[str] = None


@dataclass
class GRChomboCtorCheck:
    failure_condition: str  # C++ expression in terms of the constructor arguments
    message: str


class GRChomboFunction(DslFunctionFrontend["GRChomboComputeClass"]):
    def __init__(self,
                 name: str,
                 frontend: "GRChomboComputeClass",
                 intent_override: Optional[IntentOverride] = None,
                 *,
                 auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                 auto_soft_split_predicate: Optional[Callable[[int], bool | SoftSplitRetainmentStrategy]] = None,
                 auto_split_locus: SplitLocus = SplitLocus.Early) -> None:
        super().__init__(
            name,
            frontend,
            intent_override,
            owner_name="GRChomboFunction",
            auto_hard_split_predicate=auto_hard_split_predicate,
            auto_soft_split_predicate=auto_soft_split_predicate,
            auto_split_locus=auto_split_locus
        )


class GRChomboComputeClass(DslFrontend[GRChomboParam, GRChomboDeclKwargs, GRChomboFunction]):
    name: str
    namespace: str
    ctor_args: list[GRChomboCtorArg]
    dx: str
    ctor_checks: list[GRChomboCtorCheck]
    includes: list[str]
    enum_prefix: str
    rhs: dict[str, str]  # state variable base name -> rhs variable base name

    def __init__(
            self,
            name: str,
            *,
            ctor_args: Sequence[GRChomboCtorArg],
            dx: str,
            namespace: str = "EE",
            ctor_checks: Sequence[GRChomboCtorCheck] = (),
            includes: Sequence[str] = (),
            enum_prefix: str = "c_",
            derivative_stencil_width: int = 5,
    ) -> None:
        super().__init__(dimensionality=3, derivative_stencil_width=derivative_stencil_width)
        self.name = name
        self.namespace = namespace
        self.ctor_args = list(ctor_args)
        self.dx = dx
        self.ctor_checks = list(ctor_checks)
        self.includes = list(includes)
        self.enum_prefix = enum_prefix
        self.rhs = dict()

    def decl(self, basename: str, indices: Iterable[Idx], **kwargs: Unpack[GRChomboDeclKwargs]) -> IndexedBase:
        """
        Declares a variable, named GRChombo-style by default (see `subst_tensor_grchombo`). `rhs=R` makes it a state
        variable whose time derivative is the (previously declared) variable R.
        """
        if (rhs := kwargs.pop('rhs', None)) is not None:
            rhs_name = str(rhs.args[0])
            if rhs_name not in self.declarations:
                raise DslException(f"rhs variable {rhs_name} of {basename} must be declared first.")
            self.rhs[basename] = rhs_name
        kwargs.setdefault('substitution_rule', subst_tensor_grchombo)
        return super().decl(basename, indices, **kwargs)

    def state_of_rhs_component(self, component: str) -> Optional[str]:
        """The state-variable component written by an rhs component (h_rhs12 -> h12), or None."""
        for state, rhs in self.rhs.items():
            suffix = component[len(rhs):]
            if component.startswith(rhs) and (suffix == '' or suffix.isdigit()):
                return state + suffix
        return None

    def create_function(self,
                        name: str,
                        *,
                        intent_override: Optional[IntentOverride] = None,
                        auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                        auto_soft_split_predicate: Optional[Callable[[int], bool | SoftSplitRetainmentStrategy]] = None,
                        auto_split_locus: SplitLocus = SplitLocus.Early) -> GRChomboFunction:
        """
        Creates the class's compute function. GRChombo calls one compute() per cell, so a class has exactly one
        function, and it cannot be hard split (a hard split needs a second pass over the box).
        """
        if len(self.functions) > 0:
            raise DslException("A GRChombo compute class has exactly one function.")
        fn = GRChomboFunction(
            name,
            self,
            intent_override,
            auto_hard_split_predicate=auto_hard_split_predicate,
            auto_soft_split_predicate=auto_soft_split_predicate,
            auto_split_locus=auto_split_locus
        )
        self.functions[name] = fn
        return fn

    def add_param(self, name: str, *, cpp_type: str = 'double', init: str) -> Symbol:
        """
        Declares a parameter: a const member of the generated class, of type `cpp_type`, initialized in the
        constructor's member initializer list with the C++ expression `init` (in terms of the constructor arguments).
        """
        if cpp_type not in ('double', 'int', 'bool'):
            raise DslException(f"Unsupported parameter type {cpp_type} for {name}")
        if name in self.params:
            raise DslException(f"Parameter {name} already declared.")
        self.params[name] = GRChomboParam(name, cpp_type, init)
        return mk_symbol(name)
