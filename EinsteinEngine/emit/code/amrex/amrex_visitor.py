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
Prints an AMReX kernel class (see frontend/dsl/amrex). Each device function is

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void f(int i, int j, int k, amrex::Array4<amrex::Real> const &rhs,
                                               amrex::Array4<amrex::Real const> const &state) const

and reads the state through per-component pointers with int strides, and writes the rhs through one pointer to the
cell, the way GRTeclyn's derivative classes and CellData do. The host function compute() launches one
amrex::ParallelFor per device function over the rhs MultiFab's valid boxes, in order, on the current stream; the caller
synchronizes, as GRTeclyn's specific_eval_rhs does after its launches.
"""

from multimethod import multimethod

from EinsteinEngine.common.util import indent
from EinsteinEngine.emit.code.common.code_tree import CodeNode, StandardizedFunctionCallType, IntLiteralExpr, \
    IdExpr, FloatLiteralExpr, VerbatimExpr, UnOpExpr, BinOpExpr, BinOp, NArityOpExpr, IfElseExpr, FunctionCall, \
    StandardizedFunctionCall, GroupedExpr
from EinsteinEngine.emit.code.amrex.amrex_tree import StateLoad, StatePointerDecl, RhsLoad, CoordDecl, TempDecl, \
    RhsStore, DeviceFunctionDecl, KernelClassDecl
from EinsteinEngine.emit.tree import Identifier, LineComment, Verbatim
from EinsteinEngine.emit.visitor import Visitor, visit_each
from EinsteinEngine.generators.generator_exception import GeneratorException


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
                                     BinOp.Eq: '==', BinOp.Neq: '!='}

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
        # repr() is the shortest decimal string that rounds back to the same double.
        return f'{float(n.fl)!r}'

    @visit.register
    def _(self, n: Verbatim) -> str:
        return n.text

    @visit.register
    def _(self, n: VerbatimExpr) -> str:
        return n.text

    @visit.register
    def _(self, n: LineComment) -> str:
        return f'// {n.text}'

    @visit.register
    def _(self, n: UnOpExpr) -> str:
        return f'({n.op.representation}({self.visit(n.e)}))'

    @visit.register
    def _(self, n: BinOpExpr) -> str:
        if n.op is BinOp.Pow:
            return self._pow(n)
        if n.op in self.comparisons:
            return f'({self.visit(n.lhs)} {self.comparisons[n.op]} {self.visit(n.rhs)})'
        if n.op in (BinOp.Add, BinOp.Sub, BinOp.Mul, BinOp.Div):
            return f'({self.visit(n.lhs)} {n.op.representation} {self.visit(n.rhs)})'
        raise GeneratorException(f'AMReX backend: operator {n.op} is not supported')

    def _pow(self, n: BinOpExpr) -> str:
        base = self.visit(n.lhs)
        exponent = n.rhs
        if isinstance(exponent, UnOpExpr) and isinstance(exponent.e, IntLiteralExpr):
            exponent = IntLiteralExpr(-exponent.e.integer)
        if isinstance(exponent, IntLiteralExpr):
            k = exponent.integer
            if k == 0:
                return '1.0'
            power = ' * '.join([base] * abs(k))
            return f'({power})' if k > 0 else f'(1.0 / ({power}))'
        if (isinstance(exponent, BinOpExpr) and exponent.op is BinOp.Div
                and isinstance(exponent.lhs, FloatLiteralExpr) and isinstance(exponent.rhs, FloatLiteralExpr)):
            p, q = exponent.lhs.fl, exponent.rhs.fl
            if (p, q) == (1.0, 2.0):
                return f'sqrt({base})'
            if (p, q) == (-1.0, 2.0):
                return f'(1.0 / sqrt({base}))'
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
    def _(self, n: StateLoad) -> str:
        # i is the unit-stride direction; sy and sz are the state's j and k strides.
        offset = ''
        for k, stride in zip(n.offset, (None, 'sy', 'sz')):
            if k == 0:
                continue
            magnitude = abs(k)
            term = str(magnitude) if stride is None else stride if magnitude == 1 else f'{magnitude} * {stride}'
            offset += f' {"+" if k > 0 else "-"} {term}'
        return f'in_{n.var}[{offset.lstrip(" +") or "0"}]'

    @visit.register
    def _(self, n: StatePointerDecl) -> str:
        return f'const amrex::Real *const in_{n.var} = state_ptr + static_cast<std::ptrdiff_t>({n.enum}) * sn;'

    @visit.register
    def _(self, n: RhsLoad) -> str:
        return f'const amrex::Real {n.name} = rhs_ptr[static_cast<std::ptrdiff_t>({n.enum}) * rn];'

    @visit.register
    def _(self, n: CoordDecl) -> str:
        return f'const amrex::Real {n.name} = ({n.index} + 0.5) * DX;'

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
    def _(self, n: DeviceFunctionDecl) -> str:
        head = (
            'const amrex::Real *const state_ptr = state.ptr(i, j, k);\n'
            'const int sy = static_cast<int>(state.stride.a[0]);\n'
            'const int sz = static_cast<int>(state.stride.a[1]);\n'
            'const std::ptrdiff_t sn = state.stride.a[2];\n'
            'amrex::Real *const rhs_ptr = rhs.ptr(i, j, k);\n'
            'const std::ptrdiff_t rn = rhs.stride.a[2];\n'
            '(void)sy;\n(void)sz;\n(void)sn;\n(void)rn;\n'
        )
        return (
            f'// {n.comment}\n'
            f'AMREX_GPU_DEVICE AMREX_FORCE_INLINE void {n.name}(\n'
            '    int i, int j, int k, amrex::Array4<amrex::Real> const &rhs,\n'
            '    amrex::Array4<amrex::Real const> const &state) const\n'
            '{\n'
            + indent(head + '\n'.join(visit_each(self, n.body)))
            + '\n}'
        )

    @visit.register
    def _(self, n: KernelClassDecl) -> str:
        guard = f'{n.namespace}_{n.name}_HPP_'.upper()
        includes = '\n'.join(f'#include {h}' if h.startswith('<') else f'#include "{h}"'
                             for h in ['<AMReX_Array4.H>', '<AMReX_MultiFab.H>', '<AMReX_REAL.H>', '<cstddef>', '<type_traits>',
                                       'StateVariables.hpp', *n.includes])
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
        launches = '\n'.join(
            f'amrex::ParallelFor(a_rhs, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)\n'
            f'                   {{ self.{f.name}(i, j, k, rhs_arrays[box_no], state_arrays[box_no]); }});'
            for f in n.functions
        )
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
        functions = '\n\n'.join(visit_each(self, n.functions))
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
