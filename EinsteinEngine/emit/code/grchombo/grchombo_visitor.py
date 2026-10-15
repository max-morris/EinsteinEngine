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
Prints a GRChombo compute class. Expressions are printed for `data_t`, which BoxLoops instantiates as both `double`
and GRChombo's `simd<double>`, so only operations that both types support are emitted: arithmetic, the overloads in
GRChombo's simd_base.hpp (pow, sqrt, exp, log, ...), and simd_conditional/simd_compare_gt/simd_compare_lt for
piecewise expressions.
"""

from typing import Callable

from multimethod import multimethod

from EinsteinEngine.common.util import indent
from EinsteinEngine.emit.code.common.code_tree import CodeNode, StandardizedFunctionCallType, IntLiteralExpr, \
    IdExpr, FloatLiteralExpr, VerbatimExpr, UnOpExpr, BinOpExpr, BinOp, NArityOpExpr, IfElseExpr, FunctionCall, \
    StandardizedFunctionCall, GroupedExpr
from EinsteinEngine.emit.code.grchombo.grchombo_tree import GridLoad, PointerDecl, TempDecl, StoreVar, \
    ComputeClassDecl
from EinsteinEngine.emit.tree import Identifier, LineComment, Verbatim
from EinsteinEngine.emit.visitor import Visitor, visit_each
from EinsteinEngine.generators.generator_exception import GeneratorException


class GRChomboVisitor(Visitor[CodeNode]):
    standardized_function_calls: dict[StandardizedFunctionCallType, str] = {
        StandardizedFunctionCallType.Sin: 'sin',
        StandardizedFunctionCallType.Cos: 'cos',
        StandardizedFunctionCallType.Tan: 'tan',
        StandardizedFunctionCallType.Sinh: 'sinh',
        StandardizedFunctionCallType.Cosh: 'cosh',
        StandardizedFunctionCallType.Tanh: 'tanh',
        StandardizedFunctionCallType.Exp: 'exp',
        StandardizedFunctionCallType.Log: 'log'
    }

    # Comparisons GRChombo's simd types provide; a piecewise condition must be one of these.
    simd_comparisons: dict[BinOp, str] = {BinOp.Gt: 'simd_compare_gt', BinOp.Lt: 'simd_compare_lt'}

    enum_of: Callable[[str], str]

    def __init__(self, enum_of: Callable[[str], str] = lambda var: f'c_{var}') -> None:
        self.enum_of = enum_of

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
        if n.op in self.simd_comparisons:
            return f'{self.simd_comparisons[n.op]}({self.visit(n.lhs)}, {self.visit(n.rhs)})'
        if n.op in (BinOp.Add, BinOp.Sub, BinOp.Mul, BinOp.Div):
            return f'({self.visit(n.lhs)} {n.op.representation} {self.visit(n.rhs)})'
        raise GeneratorException(f'GRChombo backend: operator {n.op} is not supported for data_t')

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
            raise GeneratorException(f'GRChombo backend: n-ary operator {n.op} is not supported')
        return '(' + f' {n.op.representation} '.join(visit_each(self, n.args)) + ')'

    @visit.register
    def _(self, n: IfElseExpr) -> str:
        if not (isinstance(n.cond, BinOpExpr) and n.cond.op in self.simd_comparisons):
            raise GeneratorException(f'GRChombo backend: piecewise conditions must be < or > comparisons, got {n.cond}')
        return f'simd_conditional({self.visit(n.cond)}, data_t({self.visit(n.then)}), data_t({self.visit(n.else_)}))'

    @visit.register
    def _(self, n: FunctionCall) -> str:
        fn_name = self.visit(n.name)
        fn_args = ", ".join(visit_each(self, n.args))
        if fn_name == "noop":
            return f'({fn_args})'
        if len(n.template_args) > 0:
            raise GeneratorException(f'GRChombo backend: unexpected template call {fn_name}')
        return f'{fn_name}({fn_args})'

    @visit.register
    def _(self, n: GroupedExpr) -> str:
        return f'({self.visit(n.expr)})'

    @visit.register
    def _(self, n: StandardizedFunctionCall) -> str:
        if n.type not in self.standardized_function_calls:
            raise GeneratorException(f'GRChombo backend: {n.type} has no simd overload in GRChombo')
        return f'{self.standardized_function_calls[n.type]}({", ".join(visit_each(self, n.args))})'

    @visit.register
    def _(self, n: GridLoad) -> str:
        # x is the unit-stride direction (BoxPointers::m_in_stride[0] == 1).
        offset = ''
        for k, stride in zip(n.offset, (None, 'sy', 'sz')):
            if k == 0:
                continue
            magnitude = abs(k)
            term = str(magnitude) if stride is None else stride if magnitude == 1 else f'{magnitude} * {stride}'
            offset += f' {"+" if k > 0 else "-"} {term}'
        # SIMDIFY's element proxies only convert to data_t; arithmetic between two proxies is ambiguous.
        return f'data_t(in_{n.var}[idx{offset}])'

    @visit.register
    def _(self, n: PointerDecl) -> str:
        return f'const auto in_{n.var} = SIMDIFY<data_t>(box_pointers.m_in_ptr[{n.enum}]);'

    @visit.register
    def _(self, n: TempDecl) -> str:
        if n.reassign:
            line = f'{n.name} = {self.visit(n.expr)};'
        else:
            line = f'{"" if n.mutable else "const "}data_t {n.name} = {self.visit(n.expr)};'
        return line + (f' // {n.comment}' if n.comment else '')

    @visit.register
    def _(self, n: StoreVar) -> str:
        line = f'current_cell.store_vars({self.visit(n.expr)}, {n.enum});'
        return line + (f' // {n.comment}' if n.comment else '')

    @visit.register
    def _(self, n: ComputeClassDecl) -> str:
        guard = f'{n.namespace}_{n.name}_HPP_'.upper()
        includes = '\n'.join(f'#include "{h}"' for h in ['Cell.hpp', 'simd.hpp', 'UserVariables.hpp', *n.includes])
        ctor_args = ',\n'.join(
            f'{t} {name}' + (f' = {default}' if default is not None else '') for t, name, default in n.ctor_args
        )
        members = '\n'.join(f'const {p.cpp_type} {p.name};' for p in n.params)
        inits = ',\n'.join(
            [f'{p.name}({p.init})' for p in n.params]
            + [f'{d}(1.0 / ({n.dx}))' for d in ('DXI', 'DYI', 'DZI')]
        )
        checks = '\n'.join(
            f'if ({cond})\n    MayDay::Error("{n.namespace}::{n.name}: {msg}");' for cond, msg in n.ctor_checks
        )
        body = '\n'.join(visit_each(self, n.body))
        compute = (
            'template <class data_t> void compute(Cell<data_t> current_cell) const\n'
            '{\n'
            + indent(
                'const auto &box_pointers = current_cell.get_box_pointers();\n'
                'const int idx = current_cell.get_in_index();\n'
                'const int sy = box_pointers.m_in_stride[1];\n'
                'const int sz = box_pointers.m_in_stride[2];\n'
                '(void)sy;\n(void)sz;\n'
                + body
            )
            + '\n}'
        )
        cls = (
            f'class {n.name}\n'
            '{\n'
            '  protected:\n'
            + indent(members + '\nconst double DXI, DYI, DZI;') + '\n\n'
            '  public:\n'
            + indent(
                f'{n.name}(\n' + indent(ctor_args) + ')\n'
                + indent(': ' + inits.replace('\n', '\n  ')) + '\n'
                '{\n' + indent(checks) + '\n}\n\n'
                + compute
            )
            + '\n};'
        )
        helpers = (
            '// |x| for double (std::abs) and, through argument-dependent lookup, GRChombo\'s simd<double> (abs).\n'
            'template <class data_t> ALWAYS_INLINE data_t ee_abs(const data_t &x)\n'
            '{\n'
            '    using std::abs;\n'
            '    return abs(x);\n'
            '}'
        )
        return (
            f'{n.banner}\n'
            f'#ifndef {guard}\n#define {guard}\n\n'
            f'{includes}\n#include <cmath>\n\n'
            f'namespace {n.namespace}\n{{\n\n{helpers}\n\n{cls}\n\n}} // namespace {n.namespace}\n\n'
            f'#endif /* {guard} */\n'
        )
