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

from EinsteinEngine.common.stencil_idx import StencilIdx
from EinsteinEngine.emit.code.athenak.athenak_sympy_visitor import AthenaKSympyVisitor
from EinsteinEngine.emit.code.athenak.athenak_tree import TempDecl, GridStore, KernelParam, KernelLoop, \
    KernelFunctionDecl
from EinsteinEngine.emit.code.common.code_tree import Stmt, IdExpr
from EinsteinEngine.emit.tree import Identifier
from EinsteinEngine.frontend.dsl.athenak.athenak_frontend import AthenaKKernel, AthenaKFunction
from EinsteinEngine.generators.dsl_generator import DslGenerator
from EinsteinEngine.generators.generator_exception import GeneratorException
from EinsteinEngine.generators.substitute_recycled_temporaries import substitute_recycled_temporaries


def _name(s: sy.Basic) -> str:
    return str(s).replace("'", "")


def _stencil_reads(expr: sy.Basic) -> dict[str, set[StencilIdx]]:
    """The offsets at which each grid variable is read through stencil(var, x, y, z) in expr."""
    reads: dict[str, set[StencilIdx]] = dict()
    for f in expr.atoms(sy.Function):
        if getattr(f.func, 'name', None) == 'stencil':
            off = StencilIdx(*(int(a) for a in f.args[1:]))
            reads.setdefault(_name(f.args[0]), set()).add(off)
    return reads


class AthenaKGenerator(DslGenerator[AthenaKKernel]):
    _scratch: dict[str, int]  # cross-loop temporary -> its index in ee_scratch

    """
    Generates an AthenaK kernel function from an AthenaKKernel frontend: host code, then one par_for per baked
    function, in creation order. Each function's equations become the straight-line body of its KOKKOS_LAMBDA: the
    scalar temporaries in the baked order, and a store per written grid-variable component.

    A loop may read a grid variable it writes only at its own cell (offset 0, before the store, as an update such
    as rhs += dissipation does); a stencil read of a variable written in the same par_for would race between
    threads, so it is rejected. A hard-split function becomes several par_fors, its cross-loop temporaries kept in
    the kernel's scratch view (see _generate_loops).
    """

    def _generate_loops(self, fn: AthenaKFunction) -> tuple[list[KernelLoop], set[sy.Symbol]]:
        """
        One par_for per loop of the function: one, or several if it is hard split. A value computed in one loop and
        used in a later one (an EE tile temporary) goes through the kernel's scratch view, ee_scratch(m, t, k, j, i),
        written where it is computed and read in place of the symbol later; it may only be read at the same cell.
        """
        frontend = self.frontend
        assert fn.been_baked
        complex_ = fn.eqn_complex
        complex_._calc_tile_temps()
        complex_._calc_vars()

        reads = {_name(s) for s in complex_.read_decls} - self.vars_to_ignore
        writes = {_name(s) for s in complex_.write_decls} - self.vars_to_ignore
        for v in sorted(reads | writes):
            if frontend.grid_location(v) is None:
                raise GeneratorException(f"{fn.name}: {v} is read or written as a grid variable but has no view.")

        tile_temps = sorted(_name(t) for t in complex_.tile_temporaries)
        for t in tile_temps:
            self._scratch.setdefault(t, len(self._scratch))

        loops: list[KernelLoop] = []
        for loop_idx, eqn_list in enumerate(complex_.eqn_lists):
            subst = substitute_recycled_temporaries(eqn_list)
            reassigned = {s.eqn_idx for s in subst.substitutions}
            recycled_targets = {_name(s.new) for s in subst.substitutions}
            annotations = {str(lhs): ann for lhs, ann in fn.source_annotations.eqns.get(loop_idx, {}).items()}
            loop_writes = {_name(lhs) for lhs, _ in subst.eqns} & writes
            stores_here = {_name(t) for t in eqn_list.uninitialized_tile_temporaries}
            loads_here = {_name(t) for t in eqn_list.preinitialized_tile_temporaries}

            for lhs, rhs in subst.eqns:
                for var, offsets in _stencil_reads(rhs).items():
                    if var in loop_writes and offsets != {StencilIdx(0, 0, 0)}:
                        raise GeneratorException(
                            f"{fn.name}@{loop_idx}: {var} is written in this loop and read at offsets "
                            f"{sorted(offsets)}: the read would race with other threads' writes. Split the loop."
                        )
                    if var in loads_here or var in tile_temps:
                        raise GeneratorException(
                            f"{fn.name}@{loop_idx}: the cross-loop temporary {var} is read at a stencil offset; only "
                            f"same-cell reads go through the scratch view."
                        )

            def locate(name: str, loads: set[str] = loads_here) -> tuple[str, str] | None:
                if name in loads:
                    return 'ee_scratch', str(self._scratch[name])
                return frontend.grid_location(name) if name in reads or name in writes else None

            visitor = AthenaKSympyVisitor(stencil_fns={'stencil'}, locate=locate)
            body: list[Stmt] = []
            for eqn_idx, (lhs, rhs) in enumerate(subst.eqns):
                lhs_name = _name(lhs)
                expr = visitor.visit(rhs)
                comment = annotations.get(lhs_name)
                if lhs_name in writes:
                    loc = frontend.grid_location(lhs_name)
                    assert loc is not None
                    body.append(GridStore(loc[0], loc[1], expr, comment))
                else:
                    body.append(TempDecl(lhs_name, expr, reassign=eqn_idx in reassigned,
                                         mutable=lhs_name in recycled_targets, comment=comment))
                    original = _name(eqn_list.sorted_eqns[eqn_idx][0])
                    if original in stores_here:
                        body.append(GridStore('ee_scratch', str(self._scratch[original]),
                                              IdExpr(Identifier(lhs_name))))
            label = fn.label if len(complex_.eqn_lists) == 1 else f"{fn.label} {loop_idx}"
            loops.append(KernelLoop(label, frontend.loop_bounds, frontend.inv_spacing, body, fn.launch_bounds))
        return loops, set(complex_.params)

    def generate_kernel_function(self, banner: str = '') -> KernelFunctionDecl:
        frontend = self.frontend
        if len(frontend.functions) == 0:
            raise GeneratorException("An AthenaK kernel needs at least one function.")
        if len(frontend.global_temporaries) > 0:
            raise GeneratorException("Global temporaries are not supported by the AthenaK backend.")

        loops: list[KernelLoop] = []
        used_params: set[sy.Symbol] = set()
        self._scratch = dict()
        for fn in frontend.functions.values():
            fn_loops, loop_params = self._generate_loops(fn)
            loops.extend(fn_loops)
            used_params |= loop_params

        params = [KernelParam(p.name, p.cpp_type, p.init)
                  for p in sorted(frontend.params.values(), key=lambda p: p.name)
                  if sy.Symbol(p.name) in used_params]  # type: ignore[no-untyped-call]

        return KernelFunctionDecl(
            name=frontend.name,
            namespace=frontend.namespace,
            includes=frontend.includes,
            args=frontend.args,
            preamble=frontend.preamble,
            checks=[(c.failure_condition, c.message) for c in frontend.checks],
            params=params,
            loops=loops,
            scratch_count=len(self._scratch),
            scratch_extents=frontend.scratch_extents,
            banner=banner
        )
