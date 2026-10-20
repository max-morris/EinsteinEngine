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

import sympy as sy

from EinsteinEngine.common.sympywrap import free_symbols
from EinsteinEngine.emit.code.common.code_tree import Stmt
from EinsteinEngine.emit.code.amrex.amrex_sympy_visitor import AMReXSympyVisitor
from EinsteinEngine.emit.code.amrex.amrex_tree import StatePointerDecl, RhsLoad, CoordDecl, TempDecl, RhsStore, \
    KernelParam, DeviceFunctionDecl, KernelClassDecl
from EinsteinEngine.frontend.dsl.amrex.amrex_frontend import AMReXKernelClass, AMReXFunction
from EinsteinEngine.generators.dsl_generator import DslGenerator
from EinsteinEngine.generators.generator_exception import GeneratorException
from EinsteinEngine.generators.substitute_recycled_temporaries import substitute_recycled_temporaries

STENCIL_FN = 'stencil'
COORDS = {'x': 'i', 'y': 'j', 'z': 'k'}


def _name(s: sy.Basic) -> str:
    return str(s).replace("'", "")


class AMReXGenerator(DslGenerator[AMReXKernelClass]):
    """
    Generates an AMReX kernel class from an AMReXKernelClass frontend whose functions are baked. Each function becomes
    one device function, in creation order: a pointer per state component read, the rhs components it reads (written
    by an earlier function) loaded before any store, the cell coordinates it uses, the scalar temporaries in the baked
    order, and one store per rhs component written.
    """

    def _device_function(self, fn: AMReXFunction, prior_writes: set[str]) -> tuple[DeviceFunctionDecl, set[str]]:
        frontend = self.frontend
        complex_ = fn.eqn_complex
        complex_._calc_tile_temps()
        complex_._calc_vars()
        if len(complex_.eqn_lists) != 1 or len(complex_.tile_temporaries) > 0:
            raise GeneratorException(
                f"{fn.name}: the AMReX backend emits one ParallelFor per function, so a function cannot be hard split "
                f"(got {len(complex_.eqn_lists)} loops)."
            )
        eqn_list = complex_.eqn_lists[0]
        subst = substitute_recycled_temporaries(eqn_list)

        # Classify the symbols the equations read and write. An overwrite X' is written to X's component, while a
        # read of X (no prime) is the value an earlier function wrote, so temporaries are matched by their raw names.
        lhs_raw = {str(lhs) for lhs, _ in subst.eqns}
        written_raw = {str(lhs) for lhs in eqn_list.outputs}
        written: dict[str, str] = dict()  # rhs component -> state component
        for lhs in eqn_list.outputs:
            w = _name(lhs)
            if (state := frontend.state_of_rhs_component(w)) is None:
                raise GeneratorException(f"{fn.name}: {w} is written but is not the rhs of a state variable.")
            written[w] = state
        params = {str(p) for p in eqn_list.params}
        state_reads: set[str] = set()
        rhs_reads: set[str] = set()
        coords: set[str] = set()
        for _, rhs in subst.eqns:
            for call in rhs.atoms(sy.Function):
                if getattr(call.func, 'name', None) == STENCIL_FN:
                    var = _name(call.args[0])
                    if not frontend.state_component(var):
                        raise GeneratorException(f"{fn.name}: stencil read of {var}, which is not a state component.")
                    state_reads.add(var)
            for s in free_symbols(rhs):
                name = _name(s)
                if str(s) in written_raw:
                    # A store goes to the rhs array, not to a local: a later read in this function would print the
                    # value loaded before the store (or an undeclared name).
                    raise GeneratorException(f"{fn.name}: reads {s}, which this function writes to the rhs array.")
                if str(s) in params or str(s) in lhs_raw or name in self.vars_to_ignore - set(COORDS):
                    continue
                if "'" in str(s):
                    raise GeneratorException(f"{fn.name}: reads the overwrite {s} before it is written.")
                if name in COORDS:
                    coords.add(name)
                elif frontend.state_component(name):
                    state_reads.add(name)
                elif frontend.state_of_rhs_component(name) is not None:
                    if name not in prior_writes:
                        raise GeneratorException(
                            f"{fn.name}: reads {name}, which no earlier function writes; the functions are launched "
                            f"in creation order {frontend.function_order}."
                        )
                    rhs_reads.add(name)
                else:
                    raise GeneratorException(f"{fn.name}: {name} is neither a state component, an rhs component, "
                                             f"a parameter, a coordinate nor a temporary.")
        if overlap := sorted(rhs_reads & lhs_raw):
            raise GeneratorException(f"{fn.name}: {overlap} are both rhs reads and temporaries.")

        reassigned = {s.eqn_idx for s in subst.substitutions}
        recycled_targets = {_name(s.new) for s in subst.substitutions}
        annotations = {str(lhs): ann for lhs, ann in fn.source_annotations.eqns[0].items()}
        visitor = AMReXSympyVisitor(stencil_fns={STENCIL_FN}, is_state=frontend.state_component)

        body: list[Stmt] = [StatePointerDecl(v, frontend.enum_prefix + v) for v in sorted(state_reads)]
        body += [RhsLoad(v, frontend.enum_prefix + frontend.state_of_rhs_component(v)) for v in sorted(rhs_reads)]  # type: ignore[operator]
        body += [CoordDecl(c, COORDS[c]) for c in sorted(coords)]
        for eqn_idx, (lhs, rhs) in enumerate(subst.eqns):
            lhs_name = _name(lhs)
            expr = visitor.visit(rhs)
            comment = annotations.get(str(lhs))
            if lhs_name in written:
                body.append(RhsStore(frontend.enum_prefix + written[lhs_name], expr, comment))
            else:
                body.append(TempDecl(lhs_name, expr, reassign=eqn_idx in reassigned,
                                     mutable=lhs_name in recycled_targets, comment=comment))
        comment = fn.source_annotations.loops.get(0) or fn.name
        return DeviceFunctionDecl(fn.name, comment, body), set(written)

    def generate_kernel_class(self, banner: str = '') -> KernelClassDecl:
        frontend = self.frontend
        if len(frontend.global_temporaries) > 0:
            raise GeneratorException("Global temporaries are not supported by the AMReX backend.")
        functions: list[DeviceFunctionDecl] = list()
        prior_writes: set[str] = set()
        used_params: set[sy.Symbol] = set()
        for name in frontend.function_order:
            fn = frontend.functions[name]
            assert fn.been_baked
            decl, writes = self._device_function(fn, prior_writes)
            functions.append(decl)
            prior_writes |= writes
            used_params |= set(fn.eqn_complex.params)

        params = [KernelParam(p.name, p.cpp_type, p.init)
                  for p in sorted(frontend.params.values(), key=lambda p: p.name)
                  if sy.Symbol(p.name) in used_params]  # type: ignore[no-untyped-call]

        return KernelClassDecl(
            name=frontend.name,
            namespace=frontend.namespace,
            includes=frontend.includes,
            ctor_args=[(a.cpp_type, a.name, a.default) for a in frontend.ctor_args],
            ctor_prelude=frontend.ctor_prelude,
            dx=frontend.dx,
            ctor_checks=[(c.failure_condition, c.message) for c in frontend.ctor_checks],
            params=params,
            functions=functions,
            banner=banner
        )
