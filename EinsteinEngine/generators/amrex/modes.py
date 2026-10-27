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

"""Assembly of the analyzed loops into each emission mode's code tree (docs, section 5.4)."""

from typing import Sequence

from EinsteinEngine.emit.code.amrex.amrex_tree import Line, RhsLoad, TileLoad, TileStore, CoordDecl, TempDecl, \
    RhsStore, OutputAssign, KernelParam, DeviceFunctionDecl, LaunchGroup, KernelClassDecl, PointArg, \
    PointFunctionDecl, PointHeaderDecl
from EinsteinEngine.emit.code.common.code_tree import IdExpr, Stmt
from EinsteinEngine.emit.tree import Identifier
from EinsteinEngine.frontend.dsl.amrex.amrex_frontend import AMReXModule
from EinsteinEngine.frontend.dsl.amrex.schemes import KernelLaunchMode, MemberParams, NodeCentered, \
    PointFunctionMode, StructParams
from EinsteinEngine.generators.amrex.analysis import COORDS, STENCIL_FN, LoopAnalysis, unprimed
from EinsteinEngine.generators.generator_exception import GeneratorException


def kernel_class(module: AMReXModule, groups: Sequence[tuple[str, list[str], list[LoopAnalysis]]],
                 used_params: set[str],
                 banner: str) -> KernelClassDecl:
    mode = module.mode
    params = module.param_scheme
    assert isinstance(mode, KernelLaunchMode) and isinstance(params, MemberParams)
    if len(module.arrays) != 1:
        raise GeneratorException("Kernel mode reads one state array (the compute() argument a_state).")
    state = module.arrays[0]
    launches = []
    for fn_name, tiles, loops in groups:
        decls = []
        for loop in loops:
            # The printer classifies symbols as the analysis did: a field is what this loop reads as one.
            visitor = module.sympy_visitor(stencil_fns={STENCIL_FN}, is_field=loop.field_reads.__contains__,
                                           max_offset=module.max_stencil_offset)
            body: list[Stmt] = [Line(d) for v in sorted(loop.field_reads)
                                if (d := state.access.component_decl(state.name, v, loop.field_reads[v].component))]
            for v in sorted(loop.output_reads):
                field_component = loop.output_reads[v].field_component
                if field_component is None:
                    raise GeneratorException(f"{loop.name}: kernel mode stores only rhs= outputs ({v}).")
                body.append(RhsLoad(v, module.field_binding(field_component).component))  # type: ignore[union-attr]
            body += [TileLoad(t, i) for t, i in loop.tile_loads.items()]
            body += [CoordDecl(c, module.coordinate_scheme.expr(COORDS.index(c), 'ijk'[COORDS.index(c)]))
                     for c in loop.coords]
            for st in loop.statements:
                expr = visitor.visit(st.rhs)
                if st.output is not None:
                    if st.output.field_component is None:
                        raise GeneratorException(f"{loop.name}: kernel mode stores only rhs= outputs ({st.lhs}).")
                    enum = module.field_binding(st.output.field_component).component  # type: ignore[union-attr]
                    if st.local:  # read again by this loop or a later one: a local, then the store
                        body.append(TempDecl(st.lhs, expr, reassign=st.reassign, mutable=st.mutable,
                                             comment=st.comment))
                        body.append(RhsStore(enum, IdExpr(Identifier(st.lhs)), None))
                        if st.tile_index is not None:
                            body.append(TileStore(st.lhs, st.tile_index))
                    else:
                        body.append(RhsStore(enum, expr, st.comment))
                else:
                    body.append(TempDecl(st.lhs, expr, reassign=st.reassign, mutable=st.mutable, comment=st.comment))
                    if st.tile_index is not None:
                        body.append(TileStore(st.lhs, st.tile_index))
            decls.append(DeviceFunctionDecl(loop.name, loop.comment, body, state.name, mode.rhs_array,
                                            state.access.preamble(state.name), state.access.unused_markers(state.name),
                                            uses_tile=len(tiles) > 0))
        launches.append(LaunchGroup(fn_name, tiles, decls))
    kparams = [KernelParam(p.name, p.cpp_type, p.init or '')
               for p in sorted(module.params.values(), key=lambda p: p.name)
               if p.name in used_params]
    return KernelClassDecl(
        name=module.name, namespace=module.namespace, includes=module.includes,
        ctor_args=[(a.cpp_type, a.name, a.default) for a in params.ctor_args], ctor_prelude=list(params.ctor_prelude),
        dx=params.dx, ctor_checks=[(c.failure_condition, c.message) for c in params.ctor_checks],
        params=kparams, launches=launches, banner=banner)


def point_header(module: AMReXModule, groups: Sequence[tuple[str, list[str], list[LoopAnalysis]]],
                 output_order: list[str], banner: str) -> PointHeaderDecl:
    mode = module.mode
    params = module.param_scheme
    assert isinstance(mode, PointFunctionMode) and isinstance(params, StructParams)
    prefix = mode.prefix
    functions = []
    for fn_name, tiles, loops in groups:
        if len(loops) != 1 or tiles:
            raise GeneratorException(f"{fn_name}: point mode emits one inline function per DSL function; hard splits "
                                     f"are not supported yet.")
        loop = loops[0]
        locals_ = ({st.lhs for st in loop.statements if st.output is None or st.local} | set(loop.coords)
                   | set(loop.spacing))

        def name_of(n: str) -> str:
            if n in locals_:
                return prefix + n
            if n in module.params:
                return f'{params.arg}.{params.member(n)}'
            return n

        visitor = module.sympy_visitor(stencil_fns={STENCIL_FN}, is_field=loop.field_reads.__contains__,
                                       name_of=name_of, max_offset=module.max_stencil_offset)
        body: list[Stmt] = []
        body += [Line(f'const amrex::Real {prefix}{d} = 1.0 / {params.dx}[{"XYZ".index(d[1])}];') for d in loop.spacing]
        body += [CoordDecl(prefix + c, module.coordinate_scheme.expr(COORDS.index(c), 'ijk'[COORDS.index(c)]))
                 for c in loop.coords]
        outs = [o for o in output_order if o in loop.writes]
        for st in loop.statements:
            expr = visitor.visit(st.rhs)
            if st.output is not None:
                target = f'out.{unprimed(st.original)}' if mode.outputs == 'struct' else unprimed(st.original)
                if st.local:  # read again by this function: a local, then the output
                    body.append(TempDecl(prefix + st.lhs, expr, reassign=st.reassign, mutable=st.mutable,
                                         comment=st.comment))
                    body.append(OutputAssign(target, IdExpr(Identifier(prefix + st.lhs)), None))
                else:
                    body.append(OutputAssign(target, expr, st.comment))
            else:
                body.append(TempDecl(prefix + st.lhs, expr, reassign=st.reassign, mutable=st.mutable,
                                     comment=st.comment))
        arrays_used = {b.array for b in loop.field_reads.values()}
        coord_args = isinstance(module.coordinate_scheme, NodeCentered)
        args = [PointArg('int', 'i', True), PointArg('int', 'j', True), PointArg('int', 'k', True)]
        args += [PointArg(a.cpp_type, a.name, a.name in arrays_used) for a in module.arrays]
        if coord_args:
            assert isinstance(module.coordinate_scheme, NodeCentered)
            args.append(PointArg('amrex::GpuArray<amrex::Real, 3> const &', module.coordinate_scheme.prob_lo,
                                 bool(loop.coords)))
        args.append(PointArg('amrex::GpuArray<amrex::Real, 3> const &', params.dx, bool(loop.coords or loop.spacing)))
        args.append(PointArg(params.cpp_type, params.arg, bool(loop.params)))
        if mode.outputs == 'refs':
            args += [PointArg('amrex::Real &', o, True) for o in outs]
        functions.append(PointFunctionDecl(fn_name, loop.comment, args, body, outs,
                                           struct_name=f'{fn_name}_outputs' if mode.outputs == 'struct' else None))
    guard = f'{module.namespace}_{module.name}_H_'.upper()
    return PointHeaderDecl(module.namespace, module.includes, functions, banner, guard)

