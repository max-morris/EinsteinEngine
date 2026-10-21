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
from typing import Any, Callable, Iterable, Optional, Sequence, Unpack, cast

from sympy import Idx, Indexed, IndexedBase, Symbol

from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.common.sympywrap import mk_symbol
from EinsteinEngine.frontend.dsl.amrex.layout import ArrayParam, FieldBinder, FieldBinding, StridedPointer
from EinsteinEngine.frontend.dsl.amrex.naming import digit_component_names
from EinsteinEngine.frontend.dsl.amrex.schemes import CellCentered, CoordinateScheme, EmissionMode, \
    KernelLaunchMode, MemberParams, NodeCentered, ParamScheme, PointFunctionMode, StructParams
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
        supported = ((KernelLaunchMode, MemberParams, CellCentered), (PointFunctionMode, StructParams, NodeCentered))
        if not any(isinstance(mode, m) and isinstance(params, p) and isinstance(coordinates, c) for m, p, c in supported):
            raise DslException(
                f"Unsupported combination: {type(mode).__name__} with {type(params).__name__} and "
                f"{type(coordinates).__name__} (supported: "
                f"{'; '.join(f'{m.__name__} with {p.__name__} and {c.__name__}' for m, p, c in supported)}). The "
                f"coordinate scheme must spell the spacing its mode declares (DX in kernel mode, dx in point mode).")
        if isinstance(mode, PointFunctionMode) and any(isinstance(a.access, StridedPointer) for a in arrays):
            raise DslException("Point mode reads arrays through Array4Call; StridedPointer's per-function preamble is "
                               "a kernel-mode access scheme.")
        array_names = [a.name for a in arrays]
        if len(set(array_names)) != len(array_names):
            raise DslException(f"Array names must be unique: {array_names}")
        # One owner of the spacing's name: the parameter scheme (DX in kernel mode, its dx argument in point mode).
        if isinstance(coordinates, CellCentered) and coordinates.spacing != 'DX':
            raise DslException(f"CellCentered(spacing={coordinates.spacing!r}): kernel mode's spacing member is DX.")
        if isinstance(coordinates, NodeCentered) and isinstance(params, StructParams) and coordinates.dx != params.dx:
            raise DslException(f"NodeCentered(dx={coordinates.dx!r}) and StructParams(dx={params.dx!r}) must name the "
                               f"same spacing argument.")
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
        if bad := sorted(set(array_names) & self.reserved_names(with_arrays=False)):
            raise DslException(f"Array names {bad} are reserved by the generated code.")
        self.field_binders = dict()
        self.output_bases = dict()
        self.component_of = dict()
        self.indices_of = dict()

    # Declarations

    def reserved_names(self, with_arrays: bool = True) -> set[str]:
        """Names the generated code uses itself, which no parameter, field, output or temporary may take: the cell
        indices, the spacing, the coordinates (from the coordinate scheme), the array arguments, and each mode's and
        access scheme's own locals."""
        names = {'i', 'j', 'k', 'DX', 'DXI', 'DYI', 'DZI'}
        if with_arrays:
            names |= {a.name for a in self.arrays}
        if isinstance(self.mode, KernelLaunchMode):
            names |= {'rhs_ptr', 'rn', 'tile_ptr', 'tn', 'tile', 'self', 'compute', 'box_no', self.mode.rhs_array}
            for a in self.arrays:
                if isinstance(a.access, StridedPointer):
                    names |= {a.access.ptr, a.access.sy, a.access.sz, a.access.sn}
        if isinstance(self.mode, PointFunctionMode):
            names.add('out')
        if isinstance(self.coordinate_scheme, NodeCentered):
            names |= {self.coordinate_scheme.prob_lo, self.coordinate_scheme.dx}
        if isinstance(self.param_scheme, StructParams):
            names |= {self.param_scheme.arg, self.param_scheme.dx}
        if isinstance(self.param_scheme, MemberParams):
            names |= {a.name for a in self.param_scheme.ctor_args}
        return names

    def _prelude_identifiers(self) -> set[str]:
        """The identifiers of MemberParams' constructor prelude (outside string literals): a parameter of the same
        name would be shadowed there, or assigned from itself."""
        import re
        if not isinstance(self.param_scheme, MemberParams):
            return set()
        text = re.sub(r'"(?:\\.|[^"\\])*"', '', '\n'.join(self.param_scheme.ctor_prelude))
        return set(re.findall(r'[A-Za-z_]\w*', text))

    def _check_name(self, name: str, what: str) -> None:
        if name in self.reserved_names():
            raise DslException(f"{what} {name}: the name is reserved by the generated code "
                               f"({', '.join(sorted(self.reserved_names()))}).")
        if name in self.params:
            raise DslException(f"{what} {name}: already declared as a parameter.")
        if isinstance(self.mode, KernelLaunchMode):
            for a in self.arrays:
                if isinstance(a.access, StridedPointer) and name.startswith(a.access.var_prefix):
                    raise DslException(f"{what} {name}: names starting with {a.access.var_prefix} are the component "
                                       f"pointers of the generated code.")

    def _record_components(self, base: IndexedBase | Symbol, basename: str, indices: tuple[Idx, ...]) -> None:
        """Scalars are their own component; tensor components are recorded as the substitution rule names them."""
        if len(indices) == 0:
            self.component_of[(basename, ())] = basename
            self.indices_of[basename] = (basename, ())

    def _on_substitution_symbol_created(self, indexed: Indexed, sub_symbol: Symbol) -> None:
        from EinsteinEngine.frontend.dsl.use_indices import idx_to_int
        base = str(indexed.base)
        self._check_name(str(sub_symbol), f"Component of {base}")
        key = (base, tuple(idx_to_int(cast(Idx, ind)) for ind in indexed.indices))
        self.component_of[key] = str(sub_symbol)
        self.indices_of[str(sub_symbol)] = key

    def decl(self, basename: str, indices: Iterable[Idx], **kwargs: Unpack[AMReXDeclKwargs]) -> IndexedBase:
        """
        Declares a variable, named by the module's naming rule. The module's `binder` is consulted first: a
        FieldBinder makes it a field, None a temporary. With `rhs=R` (R declared first as an output), it is a field
        whose time derivative is the output R; the binder binds it, or else the module's `rhs_binder`.
        """
        rhs = kwargs.pop('rhs', None)
        kwargs.setdefault('substitution_rule', self.naming)
        indices = tuple(indices)
        self._check_name(basename, "Variable")
        binder = self.binder(basename)
        if rhs is not None:
            rhs_name = str(rhs.args[0])
            if rhs_name not in self.output_bases:
                raise DslException(f"rhs variable {rhs_name} of {basename} must be declared (as an output) first.")
            if self.output_bases[rhs_name] is not None:
                raise DslException(f"{rhs_name} is already the rhs of {self.output_bases[rhs_name]}; it cannot also be "
                                   f"the rhs of {basename}.")
            if binder is None and self.rhs_binder is None:
                raise DslException(f"{basename}: rhs= needs a field binding (the module's binder or its rhs_binder).")
            if binder is not None and self.rhs_binder is not None and binder != self.rhs_binder:
                raise DslException(f"{basename}: the binder and rhs_binder bind it differently; bind rhs= fields in "
                                   f"one place.")
            self.output_bases[rhs_name] = basename
            self.field_binders[basename] = binder if binder is not None else cast(FieldBinder, self.rhs_binder)
        elif binder is not None:
            self.field_binders[basename] = binder
        result = super().decl(basename, indices, **cast(SymbolDeclarationKwargs, kwargs))
        self._record_components(result, basename, indices)
        return result

    def field(self, basename: str, indices: Iterable[Idx], *, array: str, component: Callable[[str], str],
              **kwargs: Unpack[SymbolDeclarationKwargs]) -> IndexedBase:
        """Declares a field explicitly: every component is read from `array` with component expression
        `component(name)`."""
        self._check_name(basename, "Field")
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
        self._check_name(basename, "Output")
        if isinstance(self.mode, PointFunctionMode) and basename.startswith(self.mode.prefix):
            raise DslException(f"Output {basename}: names starting with {self.mode.prefix} are the generated locals.")
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
        if name in self.reserved_names():
            raise DslException(f"Parameter {name}: the name is reserved by the generated code "
                               f"({', '.join(sorted(self.reserved_names()))}).")
        if name in self.declarations or name in self.indices_of:
            raise DslException(f"Parameter {name}: already declared as a variable (field, output or temporary).")
        if name in self._prelude_identifiers():
            raise DslException(f"Parameter {name}: the name is an identifier of the constructor prelude, which the "
                               f"member would shadow or be assigned from.")
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

    def output_bases_components(self) -> set[str]:
        """Every declared output component's name."""
        return {c for (base, _), c in self.component_of.items() if base in self.output_bases}

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

    def default_cse_optimizations(self) -> list[Any]:
        """The CSE optimizations when the bake options name none (subclass hook); the guard fold is always added
        first."""
        from EinsteinEngine.common.cse_optimizations import default_cse_optimizations
        return list(default_cse_optimizations())

    def sympy_visitor(self, **kwargs: Any) -> Any:
        """The sympy-to-code-tree visitor of the generated functions (subclass hook, e.g. for other printing of
        functions or constants)."""
        from EinsteinEngine.emit.code.amrex.amrex_sympy_visitor import AMReXSympyVisitor
        return AMReXSympyVisitor(**kwargs)

    def bake(self, **opts: Unpack[DslFrontendBakeOptions[DslFunctionFrontendBakeOptions]]) -> None:
        from EinsteinEngine.generators.amrex.guards import fold_guards, fold_guards_optimization
        if opts.get('do_cse', True):
            given = opts.get('cse_optimizations')
            optimizations = list(self.default_cse_optimizations() if given is None else given)
            opts['cse_optimizations'] = [fold_guards_optimization(), *optimizations]
        else:
            for fn in self.functions.values():
                for eqn_list in fn.eqn_complex.eqn_lists:
                    for lhs, rhs in list(eqn_list.eqns.items()):
                        eqn_list.eqns[lhs] = fold_guards(rhs)
        super().bake(**opts)
