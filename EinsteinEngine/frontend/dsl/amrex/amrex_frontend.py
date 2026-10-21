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
The frontend of EE's shared AMReX backend (docs/AMREX-BACKEND-DESIGN.md): an AMReXModule declares fields, outputs,
parameters and temporaries, and its functions; its strategies (arrays and their access, coordinates, parameters,
emission mode) are required constructor arguments, so that a code-specific subclass states every interface choice.

Classification is by declaration, never by name:
- a field (an input read from an array) is declared by `field(...)`, by `decl(..., rhs=R)` (a field whose time
  derivative is the output R; kernel mode), or by a plain `decl` that the module's `binder` binds;
- an output is declared by `output(...)` (point mode) or as the R of `decl(..., rhs=R)`;
- parameters by `add_param`; everything else declared with `decl` is a temporary.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Unpack, cast

from sympy import Idx, Indexed, IndexedBase, Symbol

from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.common.sympywrap import mk_symbol
from EinsteinEngine.frontend.dsl.amrex.layout import ArrayParam, FieldBinder, FieldBinding
from EinsteinEngine.frontend.dsl.amrex.naming import digit_component_names
from EinsteinEngine.frontend.dsl.amrex.schemes import CoordinateScheme, EmissionMode, KernelLaunchMode, \
    MemberParams, ParamScheme, PointFunctionMode, StructParams
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend, DslFrontendBakeOptions, SymbolDeclarationKwargs
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend, DslFunctionFrontendBakeOptions
from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy
from EinsteinEngine.intermediate.split_locus import SplitLocus


class AMReXDeclKwargs(SymbolDeclarationKwargs, total=False):
    # The time derivative of this field: an output, written to this field's component of the rhs array (kernel mode).
    rhs: IndexedBase


@dataclass
class AMReXParam:
    name: str
    cpp_type: str  # 'amrex::Real', 'int' or 'bool'
    init: Optional[str]  # MemberParams: the C++ expression assigned in the constructor; StructParams: unused


@dataclass(frozen=True)
class OutputInfo:
    """An output component: its base, and in kernel mode the field component whose rhs component it is."""
    base: str
    field_component: Optional[str]


class AMReXFunction(DslFunctionFrontend["AMReXModule"]):
    def __init__(self, name: str, frontend: "AMReXModule", intent_override: Optional[IntentOverride] = None, *,
                 auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                 auto_soft_split_predicate: Optional[Callable[[int], bool | SoftSplitRetainmentStrategy]] = None,
                 auto_split_locus: SplitLocus = SplitLocus.Early) -> None:
        super().__init__(name, frontend, intent_override, owner_name="AMReXFunction",
                         auto_hard_split_predicate=auto_hard_split_predicate,
                         auto_soft_split_predicate=auto_soft_split_predicate, auto_split_locus=auto_split_locus)


class AMReXModule(DslFrontend[AMReXParam, AMReXDeclKwargs, AMReXFunction]):
    name: str
    namespace: str
    arrays: list[ArrayParam]
    binder: Callable[[str], Optional[FieldBinder]]
    rhs_binder: Optional[FieldBinder]
    coordinate_scheme: CoordinateScheme
    param_scheme: ParamScheme
    mode: EmissionMode
    naming: Callable[..., Symbol]
    includes: list[str]
    max_stencil_offset: int
    function_order: list[str]
    field_binders: dict[str, FieldBinder]  # field base -> binder
    output_bases: dict[str, Optional[str]]  # output base -> field base (rhs= shorthand) or None
    component_of: dict[tuple[str, tuple[int, ...]], str]  # (base, component indices) -> component name
    indices_of: dict[str, tuple[str, tuple[int, ...]]]  # component name -> (base, component indices)

    def __init__(self, name: str, *, arrays: Sequence[ArrayParam], binder: Callable[[str], Optional[FieldBinder]],
                 coordinates: CoordinateScheme, params: ParamScheme, mode: EmissionMode, namespace: str = "EE",
                 rhs_binder: Optional[FieldBinder] = None, naming: Callable[..., Symbol] = digit_component_names,
                 includes: Sequence[str] = (), derivative_stencil_width: int = 5, max_stencil_offset: int = 3) -> None:
        super().__init__(dimensionality=3, derivative_stencil_width=derivative_stencil_width)
        supported = {KernelLaunchMode: (MemberParams,), PointFunctionMode: (StructParams,)}
        if type(params) not in supported.get(type(mode), ()):
            raise DslException(f"Unsupported combination: {type(mode).__name__} with {type(params).__name__} "
                               f"(supported: {', '.join(f'{m.__name__} with {p[0].__name__}' for m, p in supported.items())}).")
        array_names = [a.name for a in arrays]
        if len(set(array_names)) != len(array_names):
            raise DslException(f"Array names must be unique: {array_names}")
        self.name = name
        self.namespace = namespace
        self.arrays = list(arrays)
        self.binder = binder
        self.rhs_binder = rhs_binder
        self.coordinate_scheme = coordinates
        self.param_scheme = params
        self.mode = mode
        self.naming = naming
        self.includes = list(includes)
        self.max_stencil_offset = max_stencil_offset
        self.function_order = list()
        self.field_binders = dict()
        self.output_bases = dict()
        self.component_of = dict()
        self.indices_of = dict()

    # Declarations

    def _record_components(self, base: IndexedBase | Symbol, basename: str, indices: tuple[Idx, ...]) -> None:
        """Scalars are their own component; tensor components are recorded as the substitution rule names them."""
        if len(indices) == 0:
            self.component_of[(basename, ())] = basename
            self.indices_of[basename] = (basename, ())

    def _on_substitution_symbol_created(self, indexed: Indexed, sub_symbol: Symbol) -> None:
        from EinsteinEngine.frontend.dsl.use_indices import idx_to_int
        base = str(indexed.base)
        key = (base, tuple(idx_to_int(cast(Idx, ind)) for ind in indexed.indices))
        self.component_of[key] = str(sub_symbol)
        self.indices_of[str(sub_symbol)] = key

    def decl(self, basename: str, indices: Iterable[Idx], **kwargs: Unpack[AMReXDeclKwargs]) -> IndexedBase:
        """
        Declares a variable, named by the module's naming rule. With `rhs=R` (R declared first), it is a field bound by
        the module's `rhs_binder`, whose time derivative is the output R. Otherwise the module's `binder` decides: a
        FieldBinder makes it a field, None a temporary.
        """
        rhs = kwargs.pop('rhs', None)
        kwargs.setdefault('substitution_rule', self.naming)
        indices = tuple(indices)
        if rhs is not None:
            rhs_name = str(rhs.args[0])
            if rhs_name not in self.output_bases:
                raise DslException(f"rhs variable {rhs_name} of {basename} must be declared (as an output) first.")
            if self.rhs_binder is None:
                raise DslException(f"{basename}: rhs= needs the module's rhs_binder (the array the fields live in).")
            self.output_bases[rhs_name] = basename
            self.field_binders[basename] = self.rhs_binder
        elif (binder := self.binder(basename)) is not None:
            self.field_binders[basename] = binder
        result = super().decl(basename, indices, **cast(SymbolDeclarationKwargs, kwargs))
        self._record_components(result, basename, indices)
        return result

    def field(self, basename: str, indices: Iterable[Idx], *, array: str, component: Callable[[str], str],
              **kwargs: Unpack[SymbolDeclarationKwargs]) -> IndexedBase:
        """Declares a field explicitly: every component is read from `array` with component expression
        `component(name)`."""
        if self.binder(basename) is not None:
            raise DslException(f"{basename} is declared as a field both by the binder and by field().")
        if array not in {a.name for a in self.arrays}:
            raise DslException(f"{basename}: unknown array {array}")
        kwargs.setdefault('substitution_rule', self.naming)
        indices = tuple(indices)
        self.field_binders[basename] = FieldBinder(array, component)
        result = super().decl(basename, indices, **kwargs)
        self._record_components(result, basename, indices)
        return result

    def output(self, basename: str, indices: Iterable[Idx], **kwargs: Unpack[SymbolDeclarationKwargs]) -> IndexedBase:
        """Declares an output: in point mode a register output of the functions that assign it; also the R that a
        later `decl(X, rhs=R)` attaches to field X."""
        if self.binder(basename) is not None:
            raise DslException(f"{basename} is bound as a field and cannot be an output.")
        kwargs.setdefault('substitution_rule', self.naming)
        indices = tuple(indices)
        self.output_bases[basename] = None
        result = super().decl(basename, indices, **kwargs)
        self._record_components(result, basename, indices)
        return result

    def add_param(self, name: str, *, cpp_type: str = 'amrex::Real', init: Optional[str] = None) -> Symbol:
        """Declares a parameter. MemberParams needs `init`, the C++ expression assigned in the constructor."""
        if cpp_type not in ('amrex::Real', 'int', 'bool'):
            raise DslException(f"Unsupported parameter type {cpp_type} for {name}")
        if name in self.params:
            raise DslException(f"Parameter {name} already declared.")
        if isinstance(self.param_scheme, MemberParams) and init is None:
            raise DslException(f"Parameter {name}: MemberParams needs an init expression.")
        self.params[name] = AMReXParam(name, cpp_type, init)
        return mk_symbol(name)

    def create_function(self, name: str, *, intent_override: Optional[IntentOverride] = None,
                        auto_hard_split_predicate: Optional[Callable[[int], bool]] = None,
                        auto_soft_split_predicate: Optional[Callable[[int], bool | SoftSplitRetainmentStrategy]] = None,
                        auto_split_locus: SplitLocus = SplitLocus.Early) -> AMReXFunction:
        """Creates a function. Kernel mode launches them in creation order; point mode emits one inline function each."""
        if name in self.functions:
            raise DslException(f"Function {name} already exists.")
        if name == "compute":
            raise DslException("'compute' is the name of the generated host launcher.")
        fn = AMReXFunction(name, self, intent_override, auto_hard_split_predicate=auto_hard_split_predicate,
                           auto_soft_split_predicate=auto_soft_split_predicate, auto_split_locus=auto_split_locus)
        self.functions[name] = fn
        self.function_order.append(name)
        return fn

    # Classification (used by the generator)

    def field_binding(self, component: str) -> Optional[FieldBinding]:
        """The array and component expression of a field component, or None if `component` is not a field's."""
        if (key := self.indices_of.get(component)) is None or key[0] not in self.field_binders:
            return None
        return self.field_binders[key[0]].bind(component)

    def output_info(self, component: str) -> Optional[OutputInfo]:
        """An output component's base, and the field component it is the rhs of (rhs= shorthand), or None."""
        if (key := self.indices_of.get(component)) is None or key[0] not in self.output_bases:
            return None
        base, indices = key
        field_base = self.output_bases[base]
        field_component = None if field_base is None else self.component_of.get((field_base, indices))
        if field_base is not None and field_component is None:
            raise DslException(f"Output {component} has no matching component of field {field_base}.")
        return OutputInfo(base, field_component)

    # Bake: the guard fold always comes first (docs, section 7).

    def bake(self, **opts: Unpack[DslFrontendBakeOptions[DslFunctionFrontendBakeOptions]]) -> None:
        from EinsteinEngine.generators.amrex.guards import fold_guards, fold_guards_optimization
        from EinsteinEngine.common.cse_optimizations import default_cse_optimizations
        if opts.get('do_cse', True):
            optimizations = list(opts.get('cse_optimizations', default_cse_optimizations()))
            opts['cse_optimizations'] = [fold_guards_optimization(), *optimizations]
        else:
            for fn in self.functions.values():
                for eqn_list in fn.eqn_complex.eqn_lists:
                    for lhs, rhs in list(eqn_list.eqns.items()):
                        eqn_list.eqns[lhs] = fold_guards(rhs)
        super().bake(**opts)
