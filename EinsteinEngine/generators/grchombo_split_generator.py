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

"""PROBE (branch probe/grchombo-split; never merged): a GRChombo compute class whose function is hard split into
passes. GRChombo calls compute(Cell) once per cell, so the validated backend (grchombo_generator.py) refuses split
functions; this variant emits one `compute_pass<k>(Cell)` per loop instead, plus one adaptor struct per pass with the
`compute(Cell)` that BoxLoops::loop calls. Values that a later pass needs (EE's tile temporaries) go through a scratch
FArrayBox that the host allocates per box over the output box, so a cell's output index addresses it; the host sets
its component pointers with `set_tiles` before the passes of each box (the probe's BinaryBHLevel patch).
"""
from dataclasses import dataclass
from typing import Sequence

import sympy as sy

from EinsteinEngine.emit.code.common.code_tree import Expr, Stmt
from EinsteinEngine.emit.code.grchombo.grchombo_tree import ComputeClassDecl, ComputeClassParam, PointerDecl, \
    StoreVar, TempDecl
from EinsteinEngine.generators.generator_exception import GeneratorException
from EinsteinEngine.generators.grchombo_generator import GRChomboGenerator, _name
from EinsteinEngine.emit.code.grchombo.grchombo_sympy_visitor import GRChomboSympyVisitor
from EinsteinEngine.generators.substitute_recycled_temporaries import substitute_recycled_temporaries


@dataclass
class TileLoad(Stmt):
    """const data_t <name> = data_t(SIMDIFY<data_t>(m_tile[<index>])[oidx]);"""
    name: str
    index: int


@dataclass
class TileStore(Stmt):
    """SIMDIFY<data_t>(m_tile[<index>])[oidx] = <name>;"""
    name: str
    index: int


@dataclass
class SplitComputeClassDecl(ComputeClassDecl):
    passes: Sequence[Sequence[Stmt]] = ()
    tiles: Sequence[str] = ()


class GRChomboSplitGenerator(GRChomboGenerator):
    def generate_compute_class(self, banner: str = '') -> ComputeClassDecl:
        frontend = self.frontend
        if len(frontend.functions) != 1:
            raise GeneratorException("A GRChombo compute class needs exactly one function.")
        fn = next(iter(frontend.functions.values()))
        assert fn.been_baked
        complex_ = fn.eqn_complex
        complex_._calc_tile_temps()
        complex_._calc_vars()
        if len(frontend.global_temporaries) > 0:
            raise GeneratorException(f"{fn.name}: global temporaries are not supported by the GRChombo backend.")
        reads = sorted({_name(s) for s in complex_.read_decls} - self.vars_to_ignore)
        writes = sorted({_name(s) for s in complex_.write_decls} - self.vars_to_ignore)
        tiles = sorted(_name(t) for t in complex_.tile_temporaries)
        tile_index = {t: i for i, t in enumerate(tiles)}
        stored: dict[str, str] = dict()
        for w in writes:
            if w in tile_index:
                continue
            state = frontend.state_of_rhs_component(w)
            if state is None:
                raise GeneratorException(f"{fn.name}: {w} is written but is not the rhs of a state variable.")
            stored[w] = state
        if read_written := sorted(set(reads) & set(stored)):
            raise GeneratorException(f"{fn.name}: {read_written} are both written and read.")
        read_set = set(reads)
        visitor = GRChomboSympyVisitor(stencil_fns={'stencil'}, is_grid_var=lambda name: name in read_set)

        passes: list[list[Stmt]] = []
        stored_somewhere: set[str] = set()
        for loop_idx, eqn_list in enumerate(complex_.eqn_lists):
            where = f'{fn.name} loop {loop_idx}'
            to_store = {_name(t) for t in eqn_list.uninitialized_tile_temporaries}
            to_load = sorted(_name(t) for t in eqn_list.preinitialized_tile_temporaries)
            subst = substitute_recycled_temporaries(eqn_list)
            reassigned = {s.eqn_idx for s in subst.substitutions}
            recycled_targets = {_name(s.new) for s in subst.substitutions}
            if bad := sorted(recycled_targets & (set(to_load) | to_store)):
                raise GeneratorException(f"{where}: recycling reuses the tile temporaries {bad}.")
            if bad := sorted(set(to_load) - stored_somewhere):
                raise GeneratorException(f"{where}: loads {bad}, which no earlier pass stores.")
            body: list[Stmt] = [PointerDecl(var, frontend.enum_prefix + var) for var in reads]
            body += [TileLoad(t, tile_index[t]) for t in to_load]
            for eqn_idx, (lhs, rhs) in enumerate(subst.eqns):
                lhs_name = _name(lhs)
                expr = visitor.visit(rhs)
                if lhs_name in stored:
                    body.append(StoreVar(frontend.enum_prefix + stored[lhs_name], expr))
                    continue
                body.append(TempDecl(lhs_name, expr, reassign=eqn_idx in reassigned,
                                     mutable=lhs_name in recycled_targets))
                if lhs_name in to_store:
                    body.append(TileStore(lhs_name, tile_index[lhs_name]))
            stored_somewhere |= to_store
            passes.append(body)

        params = [ComputeClassParam(p.name, p.cpp_type, p.init)
                  for p in sorted(frontend.params.values(), key=lambda p: p.name)
                  if sy.Symbol(p.name) in complex_.params]  # type: ignore[no-untyped-call]
        return SplitComputeClassDecl(
            name=frontend.name, namespace=frontend.namespace, includes=frontend.includes,
            ctor_args=[(a.cpp_type, a.name, a.default) for a in frontend.ctor_args], dx=frontend.dx,
            ctor_checks=[(c.failure_condition, c.message) for c in frontend.ctor_checks], params=params,
            body=[], banner=banner, passes=passes, tiles=tiles)


def render_split(visit, n: SplitComputeClassDecl) -> str:  # type: ignore[no-untyped-def]
    """C++ for a SplitComputeClassDecl; `visit` renders expressions and the other statements."""
    from EinsteinEngine.common.util import indent
    guard = f'{n.namespace}_{n.name}_HPP_'.upper()
    includes = '\n'.join(f'#include "{h}"' for h in ['BoxLoops.hpp', 'Cell.hpp', 'FArrayBox.H', 'simd.hpp',
                                                       'UserVariables.hpp', *n.includes])
    ctor_args = ',\n'.join(f'{t} {name}' + (f' = {default}' if default is not None else '')
                           for t, name, default in n.ctor_args)
    members = '\n'.join(f'const {p.cpp_type} {p.name};' for p in n.params)
    inits = ',\n'.join([f'{p.name}({p.init})' for p in n.params] + [f'{d}(1.0 / ({n.dx}))' for d in ('DXI', 'DYI', 'DZI')])
    checks = '\n'.join(f'if ({cond})\n    MayDay::Error("{n.namespace}::{n.name}: {msg}");' for cond, msg in n.ctor_checks)
    tiles_doc = '\n'.join(f'//   {i}: {t}' for i, t in enumerate(n.tiles))
    methods = []
    for k, body in enumerate(n.passes):
        methods.append(
            f'template <class data_t> void compute_pass{k}(Cell<data_t> current_cell) const\n'
            '{\n' + indent(
                'const auto &box_pointers = current_cell.get_box_pointers();\n'
                'const int idx = current_cell.get_in_index();\n'
                'const auto oidx = current_cell.get_out_index();\n'
                'const int sy = box_pointers.m_in_stride[1];\n'
                'const int sz = box_pointers.m_in_stride[2];\n'
                '(void)sy;\n(void)sz;\n(void)oidx;\n' + '\n'.join(visit(s) for s in body)) + '\n}')
    cls = (
        f'class {n.name}\n{{\n  protected:\n'
        + indent(members + '\nconst double DXI, DYI, DZI;\ndouble *m_tile[' + str(max(len(n.tiles), 1)) + '] = {};') + '\n\n'
        '  public:\n'
        + indent(
            f'static constexpr int num_passes = {len(n.passes)};\n'
            f'static constexpr int num_tiles = {len(n.tiles)};\n\n'
            f'{n.name}(\n' + indent(ctor_args) + ')\n' + indent(': ' + inits.replace('\n', '\n  ')) + '\n'
            '{\n' + indent(checks) + '\n}\n\n'
            '// The scratch FArrayBox of one box: num_tiles components over the output box (the rhs FArrayBox\'s box),\n'
            '// so that a cell\'s output index addresses it. Set before the passes of each box.\n'
            '// Components:\n' + tiles_doc + '\n'
            'void set_tiles(Chombo::FArrayBox &a_scratch)\n{\n'
            + indent('CH_assert(a_scratch.nComp() >= num_tiles);\nfor (int i = 0; i < num_tiles; ++i)\n    m_tile[i] = a_scratch.dataPtr(i);')
            + '\n}\n\n' + '\n\n'.join(methods))
        + '\n};')
    adaptors = '\n\n'.join(
        f'// BoxLoops::loop calls compute(Cell); this runs pass {k}.\n'
        f'struct {n.name}Pass{k}\n{{\n    const {n.name} &m_kernel;\n'
        f'    template <class data_t> void compute(Cell<data_t> current_cell) const {{ m_kernel.compute_pass{k}(current_cell); }}\n}};'
        for k in range(len(n.passes)))
    runner = (f'// The passes over one box: the host allocates the scratch (num_tiles components over out.box()), calls\n'
              f'// set_tiles, then this. Each pass is a complete BoxLoops::loop over the box, so pass k+1 reads only values\n'
              f'// that pass k stored for the same cell.\n'
              f'inline void run_passes(const {n.name} &a_kernel, const Chombo::FArrayBox &a_in, Chombo::FArrayBox &a_out,\n'
              f'                       const Chombo::Box &a_box)\n{{\n'
              + ''.join(f'    BoxLoops::loop({n.name}Pass{k}{{a_kernel}}, a_in, a_out, a_box);\n' for k in range(len(n.passes)))
              + '}')
    helpers = ('template <class data_t> ALWAYS_INLINE data_t ee_abs(const data_t &x) { return simd_max(x, -x); }\n\n'
               'static_assert(GR_SPACEDIM == 3, "EE kernels are generated for GR_SPACEDIM == 3");')
    return (f'{n.banner}\n// PROBE (EE branch probe/grchombo-split): hard-split passes; not the validated slot-in.\n'
            f'#ifndef {guard}\n#define {guard}\n\n{includes}\n\n'
            f'namespace {n.namespace}\n{{\n\n{helpers}\n\n{cls}\n\n{adaptors}\n\n{runner}\n\n}} // namespace {n.namespace}\n\n'
            f'#endif /* {guard} */\n')
