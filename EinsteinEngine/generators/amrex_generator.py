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
    TileLoad, TileStore, KernelParam, DeviceFunctionDecl, LaunchGroup, KernelClassDecl
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
    Generates an AMReX kernel class from an AMReXKernelClass frontend whose functions are baked. Each loop of each
    function (one loop unless the function is hard split) becomes one device function, launched in creation order:
    a pointer per state component read, the rhs components it reads (written by an earlier loop) and the tile
    temporaries it reads (written by an earlier loop of the same function, kept in a scratch MultiFab) loaded before
    any store, the cell coordinates it uses, the scalar temporaries in the baked order, one store per rhs component
    written, and one store per tile temporary it computes.
    """

    def _device_functions(self, fn: AMReXFunction, prior_writes: set[str]) -> tuple[list[DeviceFunctionDecl],
                                                                                     list[str], set[str]]:
        """
        The device functions of one DSL function, one per loop (a hard split gives several), in loop order; the
        function's tile temporaries (written in one loop, read in later ones), in their scratch-component order; and
        the rhs components the function writes.
        """
        frontend = self.frontend
        complex_ = fn.eqn_complex
        complex_._calc_tile_temps()
        complex_._calc_vars()
        tiles = sorted((str(t) for t in complex_.tile_temporaries))
        tile_index = {t: i for i, t in enumerate(tiles)}
        nloops = len(complex_.eqn_lists)
        decls: list[DeviceFunctionDecl] = list()
        writes: set[str] = set()
        prior = set(prior_writes)
        for loop_idx, eqn_list in enumerate(complex_.eqn_lists):
            where = fn.name if nloops == 1 else f"{fn.name} loop {loop_idx}"
            subst = substitute_recycled_temporaries(eqn_list)
            original_lhs = [str(lhs) for lhs, _ in eqn_list.sorted_eqns]
            stored_tiles = {str(t) for t in eqn_list.uninitialized_tile_temporaries}
            loaded_tiles = {str(t) for t in eqn_list.preinitialized_tile_temporaries}
            if bad := sorted({str(x.new) for x in subst.substitutions} & loaded_tiles):
                raise GeneratorException(f"{where}: recycling reuses the loaded tile temporaries {bad}.")

            # Classify the symbols the equations read and write. An overwrite X' is written to X's component, while a
            # read of X (no prime) is the value an earlier loop wrote, so temporaries are matched by their raw names.
            lhs_raw = {str(lhs) for lhs, _ in subst.eqns}
            # A tile temporary that its writer loop does not read itself is among that loop's outputs: it is stored to
            # the scratch MultiFab, not to the rhs.
            outputs = [lhs for lhs in eqn_list.outputs if str(lhs) not in stored_tiles]
            written_raw = {str(lhs) for lhs in outputs}
            written: dict[str, str] = dict()  # rhs component -> state component
            for lhs in outputs:
                w = _name(lhs)
                if (state := frontend.state_of_rhs_component(w)) is None:
                    raise GeneratorException(f"{where}: {w} is written but is not the rhs of a state variable.")
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
                            raise GeneratorException(
                                f"{where}: stencil read of {var}, which is not a state component.")
                        state_reads.add(var)
                for s in free_symbols(rhs):
                    name = _name(s)
                    if str(s) in written_raw:
                        # A store goes to the rhs array, not to a local: a later read in this loop would print the
                        # value loaded before the store (or an undeclared name).
                        raise GeneratorException(f"{where}: reads {s}, which this loop writes to the rhs array.")
                    if str(s) in params or str(s) in lhs_raw or str(s) in loaded_tiles \
                            or name in self.vars_to_ignore - set(COORDS):
                        continue
                    if "'" in str(s):
                        raise GeneratorException(f"{where}: reads the overwrite {s} before it is written.")
                    if name in COORDS:
                        coords.add(name)
                    elif frontend.state_component(name):
                        state_reads.add(name)
                    elif frontend.state_of_rhs_component(name) is not None:
                        if name not in prior:
                            raise GeneratorException(
                                f"{where}: reads {name}, which no earlier loop writes; the functions are launched "
                                f"in creation order {frontend.function_order}, each loop in order."
                            )
                        rhs_reads.add(name)
                    else:
                        raise GeneratorException(f"{where}: {name} is neither a state component, an rhs component, "
                                                 f"a parameter, a coordinate nor a temporary.")
            if overlap := sorted(rhs_reads & lhs_raw):
                raise GeneratorException(f"{where}: {overlap} are both rhs reads and temporaries.")

            reassigned = {x.eqn_idx for x in subst.substitutions}
            recycled_targets = {_name(x.new) for x in subst.substitutions}
            annotations = {str(lhs): ann for lhs, ann in fn.source_annotations.eqns[loop_idx].items()}
            visitor = AMReXSympyVisitor(stencil_fns={STENCIL_FN}, is_state=frontend.state_component)

            body: list[Stmt] = [StatePointerDecl(v, frontend.enum_prefix + v) for v in sorted(state_reads)]
            body += [RhsLoad(v, frontend.enum_prefix + frontend.state_of_rhs_component(v))  # type: ignore[operator]
                     for v in sorted(rhs_reads)]
            body += [TileLoad(t, tile_index[t]) for t in sorted(loaded_tiles)]
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
                    if original_lhs[eqn_idx] in stored_tiles:
                        body.append(TileStore(lhs_name, tile_index[original_lhs[eqn_idx]]))
            name = fn.name if nloops == 1 else f"{fn.name}_{loop_idx}"
            comment = fn.source_annotations.loops.get(loop_idx) or name
            decls.append(DeviceFunctionDecl(name, comment, body, uses_tile=len(tiles) > 0))
            prior |= set(written)
            writes |= set(written)
        return decls, tiles, writes

    def generate_kernel_class(self, banner: str = '') -> KernelClassDecl:
        frontend = self.frontend
        if len(frontend.global_temporaries) > 0:
            raise GeneratorException("Global temporaries are not supported by the AMReX backend.")
        launches: list[LaunchGroup] = list()
        prior_writes: set[str] = set()
        used_params: set[sy.Symbol] = set()
        for name in frontend.function_order:
            fn = frontend.functions[name]
            assert fn.been_baked
            decls, tiles, writes = self._device_functions(fn, prior_writes)
            launches.append(LaunchGroup(fn.name, tiles, decls))
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
            launches=launches,
            banner=banner
        )
