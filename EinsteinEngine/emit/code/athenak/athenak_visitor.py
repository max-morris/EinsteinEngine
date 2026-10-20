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
Prints an AthenaK kernel function: host code, then one AthenaK par_for per loop over (m, k, j, i), whose
KOKKOS_LAMBDA holds the straight-line body. Expressions are printed in C++ on Real, with the Kokkos math functions,
so the same code compiles for every Kokkos backend AthenaK supports.
"""

from multimethod import multimethod

from EinsteinEngine.common.util import indent
from EinsteinEngine.emit.code.common.code_tree import CodeNode, StandardizedFunctionCallType, IntLiteralExpr, \
    IdExpr, FloatLiteralExpr, VerbatimExpr, UnOpExpr, BinOpExpr, BinOp, NArityOpExpr, IfElseExpr, FunctionCall, \
    StandardizedFunctionCall, GroupedExpr
from EinsteinEngine.emit.code.athenak.athenak_tree import GridLoad, TempDecl, GridStore, KernelLoop, \
    KernelFunctionDecl
from EinsteinEngine.emit.tree import Identifier, LineComment, Verbatim
from EinsteinEngine.emit.visitor import Visitor, visit_each
from EinsteinEngine.generators.generator_exception import GeneratorException


class AthenaKVisitor(Visitor[CodeNode]):
    standardized_function_calls: dict[StandardizedFunctionCallType, str] = {
        StandardizedFunctionCallType.Sin: 'Kokkos::sin',
        StandardizedFunctionCallType.Cos: 'Kokkos::cos',
        StandardizedFunctionCallType.Tan: 'Kokkos::tan',
        StandardizedFunctionCallType.Sinh: 'Kokkos::sinh',
        StandardizedFunctionCallType.Cosh: 'Kokkos::cosh',
        StandardizedFunctionCallType.Tanh: 'Kokkos::tanh',
        StandardizedFunctionCallType.Exp: 'Kokkos::exp',
        StandardizedFunctionCallType.Erf: 'Kokkos::erf',
        StandardizedFunctionCallType.Log: 'Kokkos::log'
    }

    comparisons: dict[BinOp, str] = {BinOp.Gt: '>', BinOp.Lt: '<', BinOp.Gte: '>=', BinOp.Lte: '<='}

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
            return self._pow(n)
        if n.op in self.comparisons:
            return f'({self.visit(n.lhs)} {self.comparisons[n.op]} {self.visit(n.rhs)})'
        if n.op in (BinOp.Add, BinOp.Sub, BinOp.Mul, BinOp.Div):
            return f'({self.visit(n.lhs)} {n.op.representation} {self.visit(n.rhs)})'
        raise GeneratorException(f'AthenaK backend: operator {n.op} is not supported')

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
                return f'Kokkos::sqrt({base})'
            if (p, q) == (-1.0, 2.0):
                return f'(1.0 / Kokkos::sqrt({base}))'
        return f'Kokkos::pow({base}, {self.visit(exponent)})'

    @visit.register
    def _(self, n: NArityOpExpr) -> str:
        if n.op not in (BinOp.Add, BinOp.Mul):
            raise GeneratorException(f'AthenaK backend: n-ary operator {n.op} is not supported')
        return '(' + f' {n.op.representation} '.join(visit_each(self, n.args)) + ')'

    @visit.register
    def _(self, n: IfElseExpr) -> str:
        if not (isinstance(n.cond, BinOpExpr) and n.cond.op in self.comparisons):
            raise GeneratorException(f'AthenaK backend: piecewise conditions must be comparisons, got {n.cond}')
        return f'({self.visit(n.cond)} ? ({self.visit(n.then)}) : ({self.visit(n.else_)}))'

    @visit.register
    def _(self, n: FunctionCall) -> str:
        fn_name = self.visit(n.name)
        fn_args = ", ".join(visit_each(self, n.args))
        if fn_name == "noop":
            return f'({fn_args})'
        if len(n.template_args) > 0:
            raise GeneratorException(f'AthenaK backend: unexpected template call {fn_name}')
        if not fn_name.startswith('Kokkos::'):
            raise GeneratorException(f'AthenaK backend: unknown function {fn_name}')
        return f'{fn_name}({fn_args})'

    @visit.register
    def _(self, n: GroupedExpr) -> str:
        return f'({self.visit(n.expr)})'

    @visit.register
    def _(self, n: StandardizedFunctionCall) -> str:
        if n.type not in self.standardized_function_calls:
            raise GeneratorException(f'AthenaK backend: {n.type} is not supported')
        return f'{self.standardized_function_calls[n.type]}({", ".join(visit_each(self, n.args))})'

    @visit.register
    def _(self, n: GridLoad) -> str:
        idx = [f'{c}{"" if o == 0 else f" + {o}" if o > 0 else f" - {-o}"}'
               for c, o in (('k', n.offset.z), ('j', n.offset.y), ('i', n.offset.x))]
        return f'{n.view}(m, {n.enum}, {", ".join(idx)})'

    @visit.register
    def _(self, n: TempDecl) -> str:
        if n.reassign:
            line = f'{n.name} = {self.visit(n.expr)};'
        else:
            line = f'{"" if n.mutable else "const "}Real {n.name} = {self.visit(n.expr)};'
        return line + (f' // {n.comment}' if n.comment else '')

    @visit.register
    def _(self, n: GridStore) -> str:
        line = f'{n.view}(m, {n.enum}, k, j, i) = {self.visit(n.expr)};'
        return line + (f' // {n.comment}' if n.comment else '')

    @visit.register
    def _(self, n: KernelLoop) -> str:
        spacing = '\n'.join(f'const Real {d} = {e};\n(void){d};' for d, e in zip(('DXI', 'DYI', 'DZI'), n.inv_spacing))
        body = '\n'.join(visit_each(self, n.body))
        if n.launch_bounds is None:
            return (
                f'par_for("{n.label}", DevExeSpace(), {n.loop_bounds},\n'
                'KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {\n'
                + indent(spacing + '\n' + body) + '\n'
                '});'
            )
        # AthenaK's par_for (athena.hpp: 4D loop using a Kokkos 1D range), with LaunchBounds on the policy.
        bounds = [b.strip() for b in n.loop_bounds.split(',')]
        if len(bounds) != 8:
            raise GeneratorException(f'AthenaK backend: launch bounds need 8 loop bounds, got {n.loop_bounds}')
        ml, mu, kl, ku, jl, ju, il, iu = bounds
        threads, blocks = n.launch_bounds
        return (
            '{\n' + indent(
                f'const int ee_nk = ({ku}) - ({kl}) + 1, ee_nj = ({ju}) - ({jl}) + 1, ee_ni = ({iu}) - ({il}) + 1;\n'
                f'const int ee_nkji = ee_nk * ee_nj * ee_ni, ee_nji = ee_nj * ee_ni;\n'
                f'const int ee_n = (({mu}) - ({ml}) + 1) * ee_nkji;\n'
                f'Kokkos::parallel_for("{n.label}", '
                f'Kokkos::RangePolicy<DevExeSpace, Kokkos::LaunchBounds<{threads}, {blocks}>>(DevExeSpace(), 0, ee_n),\n'
                'KOKKOS_LAMBDA(const int &ee_idx) {\n'
                + indent(
                    'int m = ee_idx / ee_nkji;\n'
                    'int k = (ee_idx - m * ee_nkji) / ee_nji;\n'
                    'int j = (ee_idx - m * ee_nkji - k * ee_nji) / ee_ni;\n'
                    f'const int i = (ee_idx - m * ee_nkji - k * ee_nji - j * ee_ni) + ({il});\n'
                    f'm += ({ml});\nk += ({kl});\nj += ({jl});\n'
                    + spacing + '\n' + body) + '\n'
                '});') + '\n'
            '}'
        )

    @visit.register
    def _(self, n: KernelFunctionDecl) -> str:
        guard = f'EE_{n.namespace.replace("::", "_")}_{n.name}_HPP_'.upper()
        includes = '\n'.join(f'#include "{h}"' for h in n.includes)
        args = ', '.join(f'{t} {name}' for t, name in n.args)
        checks = '\n'.join(
            f'if ({cond}) {{\n'
            f'  std::cout << "### FATAL ERROR in {n.namespace}::{n.name}: {msg}" << std::endl;\n'
            f'  std::exit(EXIT_FAILURE);\n'
            f'}}' for cond, msg in n.checks
        )
        params = '\n'.join(f'const {p.cpp_type} {p.name} = {p.init};' for p in n.params)
        if n.scratch_count > 0:
            if n.scratch_extents is None:
                raise GeneratorException(f'AthenaK backend: {n.name} is hard split but the kernel has no scratch_extents')
            e_m, e_k, e_j, e_i = n.scratch_extents
            params += (
                '\n// Values carried between the hard-split loops: one view for the program, grown when needed and freed\n'
                '// before Kokkos::finalize.\n'
                'static DvceArray5D<Real> ee_scratch_store;\n'
                'static bool ee_scratch_hooked = false;\n'
                f'if (ee_scratch_store.extent_int(0) < ({e_m}) || ee_scratch_store.extent_int(2) != ({e_k}) ||\n'
                f'    ee_scratch_store.extent_int(3) != ({e_j}) || ee_scratch_store.extent_int(4) != ({e_i})) {{\n'
                '  if (!ee_scratch_hooked) {\n'
                '    Kokkos::push_finalize_hook([] { ee_scratch_store = DvceArray5D<Real>(); });\n'
                '    ee_scratch_hooked = true;\n'
                '  }\n'
                f'  ee_scratch_store = DvceArray5D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "ee scratch"), '
                f'({e_m}), {n.scratch_count}, ({e_k}), ({e_j}), ({e_i}));\n'
                '}\n'
                'auto ee_scratch = ee_scratch_store;'
            )
        loops = '\n\n'.join(visit_each(self, n.loops))
        body = '\n'.join(s for s in ['\n'.join(n.preamble), checks, params, loops] if s)
        return (
            f'{n.banner}\n'
            f'#ifndef {guard}\n#define {guard}\n\n'
            f'#include <cstdlib>\n#include <iostream>\n\n'
            f'{includes}\n\n'
            f'namespace {n.namespace} {{\n\n'
            f'inline void {n.name}({args}) {{\n'
            + indent(body) + '\n'
            '}\n\n'
            f'}} // namespace {n.namespace}\n\n'
            f'#endif // {guard}\n'
        )
