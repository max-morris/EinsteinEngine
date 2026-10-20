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
A frontend for AMReX kernel classes: a C++ class whose `AMREX_GPU_DEVICE` member functions compute a right-hand side
at one cell `(i, j, k)` of two `amrex::Array4`s, the state (read only) and the right-hand side (written, and read where
a later function updates what an earlier one wrote), and whose host function `compute(MultiFab &rhs,
const MultiFab &state)` launches one `amrex::ParallelFor` per device function, in the order the functions were created.
This is the shape of GRTeclyn's CCZ4RHS and MovingPunctureGauge (Examples/BinaryBH/BinaryBHLevel.cpp,
`specific_eval_rhs`), and of other AMReX codes that keep their evolved variables as components of one MultiFab.

The generated class replaces existing host-code classes without edits to the generated code, so the frontend
describes the interface in the host code's terms:

* the constructor signature (`ctor_args`), C++ statements run first in the constructor (`ctor_prelude`, e.g. reading
  parameter structs), and how each DSL parameter is initialized from them (`add_param(..., init=...)`);
* the grid spacing expression (`dx`), from which the DX/DXI/DYI/DZI members are formed;
* constructor checks for configurations the generated code does not implement (`ctor_checks`).

Grid variables are components of the MultiFabs: a variable `X` (or a component `X11` of a tensor) is component
`c_X` (`enum_prefix`) of the state. A variable declared with `rhs=R` is a state variable whose time derivative is R;
a component of R is component `c_X` of the right-hand side MultiFab. A function may read an rhs component written by an
earlier function, and may update it through an overwrite (`overwrite(R)`, then `add_eqn(R', R + ...)`).

The coordinates x, y, z are the cell centers `(i + 1/2) dx` (and likewise in y and z), measured from the index
origin, as GRTeclyn's Coordinates class computes them before subtracting its center.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Iterable, Unpack

from sympy import Idx, IndexedBase, Symbol

from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.common.sympywrap import mk_symbol
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend, SymbolDeclarationKwargs
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend
from EinsteinEngine.frontend.dsl.grchombo.grchombo_frontend import subst_tensor_grchombo
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy
from EinsteinEngine.intermediate.split_locus import SplitLocus


class AMReXDeclKwargs(SymbolDeclarationKwargs, total=False):
    # The time derivative of this state variable: component c_X of the right-hand side MultiFab.
    rhs: IndexedBase


@dataclass
class AMReXParam:
    name: str
    cpp_type: str  # 'amrex::Real', 'int' or 'bool'
    init: str  # C++ expression, valid after the constructor prelude


@dataclass
class AMReXCtorArg:
    cpp_type: str
    name: str
    default: Optional[str] = None


@dataclass
class AMReXCtorCheck:
    failure_condition: str  # C++ expression, valid after the constructor prelude
    message: str


class AMReXFunction(DslFunctionFrontend["AMReXKernelClass"]):
    def __init__(self,
                 name: str,
                 frontend: "AMReXKernelClass",
                 intent_override: Optional[IntentOverride] = None,
                 *,
                 auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                 auto_soft_split_predicate: Optional[Callable[[int], bool | SoftSplitRetainmentStrategy]] = None,
                 auto_split_locus: SplitLocus = SplitLocus.Early) -> None:
        super().__init__(
            name,
            frontend,
            intent_override,
            owner_name="AMReXFunction",
            auto_hard_split_predicate=auto_hard_split_predicate,
            auto_soft_split_predicate=auto_soft_split_predicate,
            auto_split_locus=auto_split_locus
        )


class AMReXKernelClass(DslFrontend[AMReXParam, AMReXDeclKwargs, AMReXFunction]):
    name: str
    namespace: str
    ctor_args: list[AMReXCtorArg]
    ctor_prelude: list[str]
    dx: str
    ctor_checks: list[AMReXCtorCheck]
    includes: list[str]
    enum_prefix: str
    rhs: dict[str, str]  # state variable base name -> rhs variable base name
    function_order: list[str]  # launch order: creation order

    def __init__(
            self,
            name: str,
            *,
            ctor_args: Sequence[AMReXCtorArg],
            dx: str,
            namespace: str = "EE",
            ctor_prelude: Sequence[str] = (),
            ctor_checks: Sequence[AMReXCtorCheck] = (),
            includes: Sequence[str] = (),
            enum_prefix: str = "c_",
            derivative_stencil_width: int = 5,
    ) -> None:
        super().__init__(dimensionality=3, derivative_stencil_width=derivative_stencil_width)
        self.name = name
        self.namespace = namespace
        self.ctor_args = list(ctor_args)
        self.ctor_prelude = list(ctor_prelude)
        self.dx = dx
        self.ctor_checks = list(ctor_checks)
        self.includes = list(includes)
        self.enum_prefix = enum_prefix
        self.rhs = dict()
        self.function_order = list()

    def decl(self, basename: str, indices: Iterable[Idx], **kwargs: Unpack[AMReXDeclKwargs]) -> IndexedBase:
        """
        Declares a variable, named GRChombo/GRTeclyn-style by default (h11, Gamma1, ...; see `subst_tensor_grchombo`).
        `rhs=R` makes it a state variable whose time derivative is the (previously declared) variable R.
        """
        if (rhs := kwargs.pop('rhs', None)) is not None:
            rhs_name = str(rhs.args[0])
            if rhs_name not in self.declarations:
                raise DslException(f"rhs variable {rhs_name} of {basename} must be declared first.")
            self.rhs[basename] = rhs_name
        kwargs.setdefault('substitution_rule', subst_tensor_grchombo)
        return super().decl(basename, indices, **kwargs)

    @staticmethod
    def _component_of(component: str, base: str) -> Optional[str]:
        """The component suffix if `component` is `base` or one of its components (h12 of h -> '12'), else None."""
        suffix = component[len(base):]
        if component.startswith(base) and (suffix == '' or suffix.isdigit()):
            return suffix
        return None

    def state_component(self, component: str) -> bool:
        """True iff `component` is a component of a state variable (chi, h12, ...)."""
        return any(self._component_of(component, state) is not None for state in self.rhs)

    def state_of_rhs_component(self, component: str) -> Optional[str]:
        """The state-variable component whose rhs is `component` (h_rhs12 -> h12), or None."""
        for state, rhs in self.rhs.items():
            if (suffix := self._component_of(component, rhs)) is not None:
                return state + suffix
        return None

    def create_function(self,
                        name: str,
                        *,
                        intent_override: Optional[IntentOverride] = None,
                        auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                        auto_soft_split_predicate: Optional[Callable[[int], bool | SoftSplitRetainmentStrategy]] = None,
                        auto_split_locus: SplitLocus = SplitLocus.Early) -> AMReXFunction:
        """
        Creates a device function of the class. `compute()` launches one ParallelFor per function, in creation order,
        so a function sees every rhs component the earlier ones wrote.
        """
        if name in self.functions:
            raise DslException(f"Function {name} already exists.")
        if name == "compute":
            raise DslException("'compute' is the name of the generated host launcher.")
        fn = AMReXFunction(
            name,
            self,
            intent_override,
            auto_hard_split_predicate=auto_hard_split_predicate,
            auto_soft_split_predicate=auto_soft_split_predicate,
            auto_split_locus=auto_split_locus
        )
        self.functions[name] = fn
        self.function_order.append(name)
        return fn

    def add_param(self, name: str, *, cpp_type: str = 'amrex::Real', init: str) -> Symbol:
        """
        Declares a parameter: a member of the generated class, of type `cpp_type`, assigned the C++ expression `init`
        in the constructor body, after the prelude.
        """
        if cpp_type not in ('amrex::Real', 'int', 'bool'):
            raise DslException(f"Unsupported parameter type {cpp_type} for {name}")
        if name in self.params:
            raise DslException(f"Parameter {name} already declared.")
        self.params[name] = AMReXParam(name, cpp_type, init)
        return mk_symbol(name)
