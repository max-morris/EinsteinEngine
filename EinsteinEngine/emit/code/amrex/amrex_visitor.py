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
Prints the shared AMReX backend's code tree (docs/AMREX-BACKEND-DESIGN.md, section 6). Kernel mode prints a class of
AMREX_GPU_DEVICE member functions on Array4 and a host compute() launching one ParallelFor per loop; point mode prints
inline AMREX_GPU_HOST_DEVICE functions with register outputs. Field reads print through each array's AccessScheme.
"""

from fractions import Fraction
from typing import Callable, Mapping

from multimethod import multimethod

from EinsteinEngine.common.util import indent
from EinsteinEngine.emit.code.common.code_tree import CodeNode, StandardizedFunctionCallType, IntLiteralExpr, \
    IdExpr, FloatLiteralExpr, VerbatimExpr, UnOpExpr, BinOpExpr, BinOp, NArityOpExpr, IfElseExpr, FunctionCall, \
    StandardizedFunctionCall, GroupedExpr
from EinsteinEngine.emit.code.common.literals import real_literal
from EinsteinEngine.emit.code.amrex.amrex_tree import FieldLoad, Line, RhsLoad, TileLoad, TileStore, CoordDecl, \
    TempDecl, RhsStore, OutputAssign, DeviceFunctionDecl, LaunchGroup, KernelClassDecl, PointFunctionDecl, \
    PointHeaderDecl
from EinsteinEngine.emit.tree import Identifier, LineComment, Verbatim
from EinsteinEngine.emit.visitor import Visitor, visit_each
from EinsteinEngine.generators.generator_exception import GeneratorException

MAX_EXPANDED_POWER = 4


class AMReXVisitor(Visitor[CodeNode]):
    standardized_function_calls: dict[StandardizedFunctionCallType, str] = {
        StandardizedFunctionCallType.Sin: 'sin',
        StandardizedFunctionCallType.Cos: 'cos',
        StandardizedFunctionCallType.Tan: 'tan',
        StandardizedFunctionCallType.Sinh: 'sinh',
        StandardizedFunctionCallType.Cosh: 'cosh',
        StandardizedFunctionCallType.Tanh: 'tanh',
        StandardizedFunctionCallType.Exp: 'exp',
        StandardizedFunctionCallType.Erf: 'erf',
        StandardizedFunctionCallType.Log: 'log'
    }

    comparisons: dict[BinOp, str] = {BinOp.Gt: '>', BinOp.Lt: '<', BinOp.Gte: '>=', BinOp.Lte: '<=',
                                     BinOp.Eq: '==', BinOp.Neq: '!=', BinOp.And: '&&', BinOp.Or: '||'}

    # var -> its load printer (set by the generator per function: the array's AccessScheme and component)
    field_load: Callable[[FieldLoad], str]

    def __init__(self) -> None:
        self.field_load = lambda n: (_ for _ in ()).throw(GeneratorException(f'no field binding for {n.var}'))

    @multimethod
    def visit(self, n: CodeNode) -> str:
        self.not_implemented(n)

    @visit.register
    def _(self, n: Identifier) -> str:
        return n.identifier

    @visit.register
    def _(self, n: IdExpr) -> str:
        return n.id.identifier

    @visit.register
    def _(self, n: IntLiteralExpr) -> str:
        return f'{n.integer}'

    @visit.register
    def _(self, n: FloatLiteralExpr) -> str:
        return self.float_literal(float(n.fl))

    def float_literal(self, fl: float) -> str:
        """The shortest decimal that reads back as exactly `fl` (subclass hook)."""
        return real_literal(fl)

    @visit.register
    def _(self, n: Verbatim) -> str:
        return n.text

    @visit.register
    def _(self, n: VerbatimExpr) -> str:
        return n.v.text

    @visit.register
    def _(self, n: LineComment) -> str:
        return f'// {n.text}'

    @visit.register
    def _(self, n: UnOpExpr) -> str:
        return f'({n.op.representation}({self.visit(n.e)}))'

    @visit.register
    def _(self, n: BinOpExpr) -> str:
        if n.op is BinOp.Pow:
            return self.pow(n)
        if n.op in self.comparisons:
            return f'({self.visit(n.lhs)} {self.comparisons[n.op]} {self.visit(n.rhs)})'
        if n.op in (BinOp.Add, BinOp.Sub, BinOp.Mul, BinOp.Div):
            return f'({self.visit(n.lhs)} {n.op.representation} {self.visit(n.rhs)})'
        raise GeneratorException(f'AMReX backend: operator {n.op} is not supported')

    def pow(self, n: BinOpExpr) -> str:
        """Powers (docs, section 6): 0 -> 1.0; +-1..4 -> products or 1.0/products; +-1/2 -> sqrt forms; any other
        exponent -> pow(x, p) with p the exact double or the symbol (subclass hook)."""
        base = self.visit(n.lhs)
        exponent = n.rhs
        if isinstance(exponent, UnOpExpr) and isinstance(exponent.e, IntLiteralExpr):
            exponent = IntLiteralExpr(-exponent.e.integer)
        if isinstance(exponent, IntLiteralExpr):
            k = exponent.integer
            if k == 0:
                return '1.0'
            if abs(k) <= MAX_EXPANDED_POWER:
                power = ' * '.join([base] * abs(k))
                return f'({power})' if k > 0 else f'(1.0 / ({power}))'
            return f'pow({base}, {self.float_literal(float(k))})'
        if (isinstance(exponent, BinOpExpr) and exponent.op is BinOp.Div
                and isinstance(exponent.lhs, FloatLiteralExpr) and isinstance(exponent.rhs, FloatLiteralExpr)):
            p, q = exponent.lhs.fl, exponent.rhs.fl
            if (p, q) == (1.0, 2.0):
                return f'sqrt({base})'
            if (p, q) == (-1.0, 2.0):
                return f'(1.0 / sqrt({base}))'
            return f'pow({base}, {self.float_literal(float(Fraction(int(p), int(q))))})'
        return f'pow({base}, {self.visit(exponent)})'

    @visit.register
    def _(self, n: NArityOpExpr) -> str:
        if n.op not in (BinOp.Add, BinOp.Mul):
            raise GeneratorException(f'AMReX backend: n-ary operator {n.op} is not supported')
        return '(' + f' {n.op.representation} '.join(visit_each(self, n.args)) + ')'

    @visit.register
    def _(self, n: IfElseExpr) -> str:
        return f'({self.visit(n.cond)} ? amrex::Real({self.visit(n.then)}) : amrex::Real({self.visit(n.else_)}))'

    @visit.register
    def _(self, n: FunctionCall) -> str:
        fn_name = self.visit(n.name)
        fn_args = ", ".join(visit_each(self, n.args))
        if fn_name == "noop":
            return f'({fn_args})'
        if fn_name == "!":
            return f'(!({fn_args}))'
        if len(n.template_args) > 0:
            raise GeneratorException(f'AMReX backend: unexpected template call {fn_name}')
        return f'{fn_name}({fn_args})'

    @visit.register
    def _(self, n: GroupedExpr) -> str:
        return f'({self.visit(n.expr)})'

    @visit.register
    def _(self, n: StandardizedFunctionCall) -> str:
        if n.type not in self.standardized_function_calls:
            raise GeneratorException(f'AMReX backend: {n.type} is not supported')
        return f'{self.standardized_function_calls[n.type]}({", ".join(visit_each(self, n.args))})'

    @visit.register
    def _(self, n: FieldLoad) -> str:
        return self.field_load(n)

    @visit.register
    def _(self, n: Line) -> str:
        return n.text

    @visit.register
    def _(self, n: RhsLoad) -> str:
        return f'const amrex::Real {n.name} = rhs_ptr[static_cast<std::ptrdiff_t>({n.enum}) * rn];'

    @visit.register
    def _(self, n: TileLoad) -> str:
        return f'const amrex::Real {n.name} = tile_ptr[{n.index} * tn];'

    @visit.register
    def _(self, n: TileStore) -> str:
        return f'tile_ptr[{n.index} * tn] = {n.name};'

    @visit.register
    def _(self, n: CoordDecl) -> str:
        return f'const amrex::Real {n.name} = {n.expr};'

    @visit.register
    def _(self, n: TempDecl) -> str:
        if n.reassign:
            line = f'{n.name} = {self.visit(n.expr)};'
        else:
            line = f'{"" if n.mutable else "const "}amrex::Real {n.name} = {self.visit(n.expr)};'
        return line + (f' // {n.comment}' if n.comment else '')

    @visit.register
    def _(self, n: RhsStore) -> str:
        line = f'rhs_ptr[static_cast<std::ptrdiff_t>({n.enum}) * rn] = {self.visit(n.expr)};'
        return line + (f' // {n.comment}' if n.comment else '')

    @visit.register
    def _(self, n: OutputAssign) -> str:
        line = f'{n.target} = {self.visit(n.expr)};'
        return line + (f' // {n.comment}' if n.comment else '')

    # Kernel mode

    @visit.register
    def _(self, n: DeviceFunctionDecl) -> str:
        head = '\n'.join([*n.preamble,
                          f'amrex::Real *const rhs_ptr = {n.rhs_array}.ptr(i, j, k);',
                          f'const std::ptrdiff_t rn = {n.rhs_array}.stride.a[2];',
                          *(f'(void){u};' for u in [*n.unused, 'rn'])]) + '\n'
        tile_arg = ''
        if n.uses_tile:
            tile_arg = ',\n    amrex::Array4<amrex::Real> const &tile'
            head += ('amrex::Real *const tile_ptr = tile.ptr(i, j, k);\n'
                     'const std::ptrdiff_t tn = tile.stride.a[2];\n(void)tile_ptr;\n(void)tn;\n')
        return (
            f'// {n.comment}\n'
            f'AMREX_GPU_DEVICE AMREX_FORCE_INLINE void {n.name}(\n'
            f'    int i, int j, int k, amrex::Array4<amrex::Real> const &{n.rhs_array},\n'
            f'    amrex::Array4<amrex::Real const> const &{n.state_array}{tile_arg}) const\n'
            '{\n'
            + indent(head + '\n'.join(visit_each(self, n.body)))
            + '\n}'
        )

    def _launches(self, g: LaunchGroup) -> str:
        lines = []
        if g.tiles:
            lines += [
                f'// Scratch for {g.name}\'s tile temporaries ({", ".join(g.tiles)}), on the valid cells; stream-ordered',
                '// allocation, so freeing it at the end of compute() needs no synchronization.',
                f'amrex::MultiFab tile_{g.name}(a_rhs.boxArray(), a_rhs.DistributionMap(), {len(g.tiles)}, 0,',
                '                             amrex::MFInfo().SetArena(amrex::The_Async_Arena()));',
                f'const auto &tile_{g.name}_arrays = tile_{g.name}.arrays();',
            ]
        for f in g.functions:
            tile = f', tile_{g.name}_arrays[box_no]' if f.uses_tile else ''
            lines.append(f'amrex::ParallelFor(a_rhs, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)\n'
                         f'                   {{ self.{f.name}(i, j, k, rhs_arrays[box_no], state_arrays[box_no]{tile}); }});')
        return '\n'.join(lines)

    @visit.register
    def _(self, n: KernelClassDecl) -> str:
        guard = f'{n.namespace}_{n.name}_HPP_'.upper()
        includes = '\n'.join(f'#include {h}' if h.startswith('<') else f'#include "{h}"'
                             for h in ['<AMReX_Array4.H>', '<AMReX_MultiFab.H>', '<AMReX_REAL.H>', '<cstddef>',
                                       '<type_traits>', *n.includes])
        ctor_args = ',\n'.join(
            f'{t} {name}' + (f' = {default}' if default is not None else '') for t, name, default in n.ctor_args
        )
        members = '\n'.join(f'{p.cpp_type} {p.name}{{}};' for p in n.params)
        ctor_body = '\n'.join(
            [*n.ctor_prelude]
            + [f'if ({cond})\n    amrex::Abort("{n.namespace}::{n.name}: {msg}");' for cond, msg in n.ctor_checks]
            + [f'{p.name} = {p.init};' for p in n.params]
            + [f'DX = {n.dx};'] + [f'{d} = 1.0 / ({n.dx});' for d in ('DXI', 'DYI', 'DZI')]
        )
        launches = '\n'.join(self._launches(g) for g in n.launches)
        compute = (
            '// Launches the device functions, one ParallelFor each, in order, over the valid cells of a_rhs.\n'
            '// The caller synchronizes the stream.\n'
            'void compute(amrex::MultiFab &a_rhs, const amrex::MultiFab &a_state) const\n'
            '{\n'
            + indent(
                'const auto &rhs_arrays = a_rhs.arrays();\n'
                'const auto &state_arrays = a_state.const_arrays();\n'
                f'const {n.name} self = *this;\n'
                + launches
            )
            + '\n}'
        )
        functions = '\n\n'.join(visit_each(self, [f for g in n.launches for f in g.functions]))
        cls = (
            f'class {n.name}\n'
            '{\n'
            '  protected:\n'
            + indent(members + '\namrex::Real DX{}, DXI{}, DYI{}, DZI{};') + '\n\n'
            '  public:\n'
            + indent(
                f'{n.name}(\n' + indent(ctor_args) + ')\n'
                '{\n' + indent(ctor_body) + '\n}\n\n'
                + compute + '\n\n'
                + functions
            )
            + '\n};'
        )
        return (
            f'{n.banner}\n'
            f'#ifndef {guard}\n#define {guard}\n\n'
            f'{includes}\n\n'
            f'namespace {n.namespace}\n{{\n\n'
            '// The recipe is written in three dimensions.\n'
            'static_assert(AMREX_SPACEDIM == 3, "EE kernels are generated for AMREX_SPACEDIM == 3");\n\n'
            f'{cls}\n\n'
            f'static_assert(std::is_trivially_copyable_v<{n.name}>, "{n.name} is captured by value in device lambdas");\n\n'
            f'}} // namespace {n.namespace}\n\n'
            f'#endif /* {guard} */\n'
        )

    # Point mode

    @visit.register
    def _(self, n: PointFunctionDecl) -> str:
        args = ',\n'.join(f'    {"" if a.used else "[[maybe_unused]] "}{a.cpp_type} {a.name}' for a in n.args)
        ret = n.struct_name if n.struct_name is not None else 'void'
        body = '\n'.join(visit_each(self, n.body))
        struct = ''
        if n.struct_name is not None:
            struct = (f'// The outputs of {n.name}, in order.\nstruct {n.struct_name}\n{{\n'
                      + indent('\n'.join(f'amrex::Real {o};' for o in n.outputs)) + '\n};\n\n')
            body = f'{n.struct_name} out{{}};\n' + body + '\nreturn out;'
        return (
            struct
            + f'// {n.comment}\n'
            f'AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE {ret} {n.name}(\n{args})\n'
            '{\n' + indent(body) + '\n}'
        )

    @visit.register
    def _(self, n: PointHeaderDecl) -> str:
        includes = '\n'.join(f'#include {h}' if h.startswith('<') else f'#include "{h}"'
                             for h in ['<AMReX_Array4.H>', '<AMReX_REAL.H>', '<AMReX_GpuQualifiers.H>',
                                       '<AMReX_Array.H>', '<AMReX_Algorithm.H>', '<AMReX_Math.H>', '<cmath>',
                                       *n.includes])
        functions = '\n\n'.join(visit_each(self, n.functions))
        return (
            f'{n.banner}\n'
            f'#ifndef {n.guard}\n#define {n.guard}\n\n'
            f'{includes}\n\n'
            f'namespace {n.namespace}\n{{\n\n'
            '// The recipe is written in three dimensions.\n'
            'static_assert(AMREX_SPACEDIM == 3, "EE kernels are generated for AMREX_SPACEDIM == 3");\n\n'
            f'{functions}\n\n'
            f'}} // namespace {n.namespace}\n\n'
            f'#endif /* {n.guard} */\n'
        )
