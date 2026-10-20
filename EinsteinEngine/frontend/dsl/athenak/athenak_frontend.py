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
A frontend for AthenaK kernels: a C++ host function that runs one or more Kokkos `par_for` loops over the
(m, k, j, i) cells of a MeshBlockPack, AthenaK's own loop helper (athena.hpp). Each DSL function becomes one
`par_for`, run in creation order, so a later function may read what an earlier one wrote, at any stencil offset.

The generated function is meant to replace a loop in AthenaK (e.g. the Z4c right-hand side) without editing the
generated code, so the frontend describes the host side in AthenaK's terms:

* the C++ function's parameters (`args`) and the host code that runs before the loops (`preamble`), which binds
  the views, the loop bounds and anything the parameter initializers use;
* the `par_for` bounds (`loop_bounds`, a C++ argument list "ml, mu, kl, ku, jl, ju, il, iu");
* the inverse grid spacings (`inv_spacing`: three device expressions in the MeshBlock index m), from which
  DXI/DYI/DZI are formed in each loop;
* how each DSL parameter is initialized on the host (`add_param(..., init=...)`); parameters are captured by value;
* host checks for configurations the generated code does not implement (`checks`);
* for hard-split functions, the extents (m, k, j, i) of the scratch view that carries values between their loops
  (`scratch_extents`, host expressions; the view is allocated once, grown when needed, freed at Kokkos::finalize).

Grid variables live in AthenaK's DvceArray5D views, indexed (m, n, k, j, i). A variable is declared with `view`
(the C++ name of its view) and `enum` (the C++ index n of a scalar, or the prefix of its components' indices: a
tensor component's index is the prefix followed by its index letters in upper case, e.g. I_Z4C_G + XY). Components
are named with `subst_tensor_xyz` (gxy, betax, ...). Variables declared without `view` are temporaries.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Callable, Iterable, Unpack, cast

from sympy import Idx, IndexedBase, Symbol

from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.common.sympywrap import mk_symbol
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend, SymbolDeclarationKwargs
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend
from EinsteinEngine.frontend.dsl.use_indices import subst_tensor_xyz
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy
from EinsteinEngine.intermediate.split_locus import SplitLocus

_XYZ = 'xyz'


class AthenaKDeclKwargs(SymbolDeclarationKwargs, total=False):
    # The C++ name of the DvceArray5D that holds this grid variable.
    view: str
    # The C++ index (second view index) of a scalar, or the prefix of a tensor's component indices.
    enum: str


@dataclass(frozen=True)
class AthenaKGridVar:
    base: str
    rank: int
    view: str
    enum: str

    def component_enum(self, suffix: str) -> str:
        return self.enum + suffix.upper()


@dataclass
class AthenaKParam:
    name: str
    cpp_type: str  # 'Real', 'int' or 'bool'
    init: str  # host C++ expression


@dataclass
class AthenaKCheck:
    failure_condition: str  # host C++ expression
    message: str


class AthenaKFunction(DslFunctionFrontend["AthenaKKernel"]):
    label: str
    launch_bounds: Optional[tuple[int, int]]

    def __init__(self,
                 name: str,
                 frontend: "AthenaKKernel",
                 intent_override: Optional[IntentOverride] = None,
                 *,
                 label: Optional[str] = None,
                 launch_bounds: Optional[tuple[int, int]] = None,
                 auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                 auto_soft_split_predicate: Optional[Callable[[int], bool | SoftSplitRetainmentStrategy]] = None,
                 auto_split_locus: SplitLocus = SplitLocus.Early) -> None:
        super().__init__(
            name,
            frontend,
            intent_override,
            owner_name="AthenaKFunction",
            auto_hard_split_predicate=auto_hard_split_predicate,
            auto_soft_split_predicate=auto_soft_split_predicate,
            auto_split_locus=auto_split_locus
        )
        self.label = label if label is not None else f"EE {frontend.name} {name}"
        self.launch_bounds = launch_bounds


class AthenaKKernel(DslFrontend[AthenaKParam, AthenaKDeclKwargs, AthenaKFunction]):
    name: str
    namespace: str
    args: list[tuple[str, str]]
    preamble: list[str]
    loop_bounds: str
    inv_spacing: tuple[str, str, str]
    checks: list[AthenaKCheck]
    includes: list[str]
    scratch_extents: Optional[tuple[str, str, str, str]]
    grid: dict[str, AthenaKGridVar]

    def __init__(
            self,
            name: str,
            *,
            args: Sequence[tuple[str, str]],
            preamble: Sequence[str],
            loop_bounds: str,
            inv_spacing: tuple[str, str, str],
            namespace: str = "ee",
            checks: Sequence[AthenaKCheck] = (),
            includes: Sequence[str] = ("athena.hpp",),
            scratch_extents: Optional[tuple[str, str, str, str]] = None,
            derivative_stencil_width: int = 5,
    ) -> None:
        super().__init__(dimensionality=3, derivative_stencil_width=derivative_stencil_width)
        self.name = name
        self.namespace = namespace
        self.args = list(args)
        self.preamble = list(preamble)
        self.loop_bounds = loop_bounds
        self.inv_spacing = inv_spacing
        self.checks = list(checks)
        self.includes = list(includes)
        self.scratch_extents = scratch_extents
        self.grid = dict()

    def decl(self, basename: str, indices: Iterable[Idx], **kwargs: Unpack[AthenaKDeclKwargs]) -> IndexedBase:
        """
        Declares a variable. With `view` and `enum` it is a grid variable stored in that view; otherwise it is a
        temporary. Tensor components are named with subst_tensor_xyz unless a substitution_rule is given.
        """
        indices = tuple(indices)
        view = kwargs.pop('view', None)
        enum = kwargs.pop('enum', None)
        if (view is None) != (enum is None):
            raise DslException(f"{basename}: a grid variable needs both view and enum.")
        if "'" in basename:
            # An overwrite() of a grid variable: the same storage as the original.
            original = basename.replace("'", "")
            if original in self.grid:
                view, enum = self.grid[original].view, self.grid[original].enum
        if view is not None and enum is not None:
            if basename.replace("'", "") not in self.grid:
                self.grid[basename.replace("'", "")] = AthenaKGridVar(basename.replace("'", ""), len(indices),
                                                                       view, enum)
        kwargs.setdefault('substitution_rule', subst_tensor_xyz)
        return super().decl(basename, indices, **cast(Any, kwargs))

    def grid_location(self, component: str) -> Optional[tuple[str, str]]:
        """The (view, index) of a grid-variable component symbol (gxy -> (u0, I_Z4C_GXY)), or None."""
        name = component.replace("'", "")
        for base in sorted(self.grid, key=len, reverse=True):
            gv = self.grid[base]
            suffix = name[len(base):]
            if name.startswith(base) and len(suffix) == gv.rank and all(c in _XYZ for c in suffix):
                return gv.view, gv.component_enum(suffix)
        return None

    def create_function(self,
                        name: str,
                        *,
                        label: Optional[str] = None,
                        launch_bounds: Optional[tuple[int, int]] = None,
                        intent_override: Optional[IntentOverride] = None,
                        auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                        auto_soft_split_predicate: Optional[Callable[[int], bool | SoftSplitRetainmentStrategy]] = None,
                        auto_split_locus: SplitLocus = SplitLocus.Early) -> AthenaKFunction:
        """
        Creates a loop: one par_for over the cells, run after the loops created before it (several if the function is
        hard split; their cross-loop temporaries need `scratch_extents`). `label` is the par_for's Kokkos label (what
        the Kokkos Tools and nsys report); `launch_bounds` = (max threads per block, min blocks per multiprocessor)
        launches it with Kokkos::LaunchBounds.
        """
        if name in self.functions:
            raise DslException(f"Function {name} already exists.")
        fn = AthenaKFunction(
            name,
            self,
            intent_override,
            label=label,
            launch_bounds=launch_bounds,
            auto_hard_split_predicate=auto_hard_split_predicate,
            auto_soft_split_predicate=auto_soft_split_predicate,
            auto_split_locus=auto_split_locus
        )
        self.functions[name] = fn
        return fn

    def add_param(self, name: str, *, cpp_type: str = 'Real', init: str) -> Symbol:
        """
        Declares a parameter: a host-side `const <cpp_type> <name> = <init>;` before the loops, captured by value.
        """
        if cpp_type not in ('Real', 'int', 'bool'):
            raise DslException(f"Unsupported parameter type {cpp_type} for {name}")
        if name in self.params:
            raise DslException(f"Parameter {name} already declared.")
        self.params[name] = AthenaKParam(name, cpp_type, init)
        return mk_symbol(name)
