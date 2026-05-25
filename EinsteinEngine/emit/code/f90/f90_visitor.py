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

from multimethod import multimethod

import typing

from EinsteinEngine.common.util import indent
from EinsteinEngine.emit.code.f90.f90_tree import F90Expr, F90Stmt, F90Directive, F90Decl, TypeAttribute, TypeSpecifier, PrimitiveType, IntentIn, IntentOut, Dimension, VarDecl, DoLoop, Assignment, SubroutineDecl, ArrayAccess
from EinsteinEngine.emit.code.f90.f90_sympy_visitor import F90SympyVisitor
from EinsteinEngine.emit.tree import Identifier, Integer, Verbatim, LineComment, BlockComment, String, Bool, Float

from EinsteinEngine.emit.code.common.code_tree import CodeNode, StandardizedFunctionCallType, Decl, Directive, Stmt, \
    Expr, ExprStmt, IfElseStmt, IntLiteralExpr, IdExpr, FloatLiteralExpr, VerbatimExpr, \
    UnOpExpr, BinOpExpr, BinOp, NArityOpExpr, IfElseExpr, SympyExpr, FunctionCall, StandardizedFunctionCall, GroupedExpr

from EinsteinEngine.emit.visitor import Visitor, visit_each
from EinsteinEngine.emit.code.f90.f90_tree import F90CodeRoot
from EinsteinEngine.emit.code.f90.f90_tree import Allocatable


class F90Visitor(Visitor[CodeNode]):
    sympy_visitor: F90SympyVisitor

    standardized_function_calls: dict[StandardizedFunctionCallType, str] = {
        StandardizedFunctionCallType.Sin: 'SIN',
        StandardizedFunctionCallType.Cos: 'COS',
        StandardizedFunctionCallType.Tan: 'TAN',
        StandardizedFunctionCallType.Cot: 'COT',
        StandardizedFunctionCallType.Sec: 'SEC',
        StandardizedFunctionCallType.Csc: 'CSC',
        StandardizedFunctionCallType.Sinh: 'SINH',
        StandardizedFunctionCallType.Cosh: 'COSH',
        StandardizedFunctionCallType.Tanh: 'TANH',
        StandardizedFunctionCallType.Coth: 'COTH',
        StandardizedFunctionCallType.Sech: 'SECH',
        StandardizedFunctionCallType.Csch: 'CSCH',
        StandardizedFunctionCallType.Erf: 'ERF',
        StandardizedFunctionCallType.Exp: 'EXP',
        StandardizedFunctionCallType.Log: 'LOG'
    }

    def __init__(self) -> None:
        #self.sympy_visitor = F90SympyVisitor()
        pass

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
    def _(self, n: Integer) -> str:
        return f'{n.integer}'

    @visit.register
    def _(self, n: IntLiteralExpr) -> str:
        return f'{n.integer}'

    @visit.register
    def _(self, n: FloatLiteralExpr) -> str:
        return f'{n.fl:.1f}D0' if float(n.fl).is_integer() else f'{n.fl}D0'

    @visit.register
    def _(self, n: Verbatim) -> str:
        return n.text

    @visit.register
    def _(self, n: LineComment) -> str:
        return f'! {n.text}'

    @visit.register
    def _(self, n: String) -> str:
        return f'"{n.text}"' if not n.single_quotes else f"'{n.text}'"

    @visit.register
    def _(self, n: Bool) -> str:
        return '.TRUE.' if n.b else '.FALSE.'

    @visit.register
    def _(self, n: ExprStmt) -> str:
        return typing.cast(str, self.visit(n.expr))

    @visit.register
    def _(self, n: VerbatimExpr) -> str:
        return n.v.text

    @visit.register
    def _(self, n: UnOpExpr) -> str:
        expr = self.visit(n.e)
        return f'({n.op.representation}{expr})'

    @visit.register
    def _(self, n: BinOpExpr) -> str:
        lhs = self.visit(n.lhs)
        rhs = self.visit(n.rhs)

        if n.op is BinOp.Pow:
            return f'({lhs} ** {rhs})'
        if n.op is BinOp.Neq:
            return f'({lhs} /= {rhs})'


        return f'({lhs} {n.op.representation} {rhs})'

    @visit.register
    def _(self, n: NArityOpExpr) -> str:
        assert n.op != BinOp.Pow

        if len(n.args) == 0:
            return ''

        st: str = f'({self.visit(n.args[0])}'
        for a in n.args[1:]:
            st += f' {n.op.representation} {self.visit(a)}'

        return f'{st})'

    @visit.register
    def _(self, n: IfElseExpr) -> str:
        return f'MERGE({self.visit(n.then)}, {self.visit(n.else_)}, {self.visit(n.cond)})'

    @visit.register
    def _(self, n: IfElseStmt) -> str:
        then = "\n".join(visit_each(self, n.then))
        else_ = "\n".join(visit_each(self, n.else_))
        return f'if ({self.visit(n.cond)}) then\n{indent(then)}\nelse\n{indent(else_)}\nend if'

    @visit.register
    def _(self, n: FunctionCall) -> str:
        fn_name = n.name.identifier
        fn_args = ", ".join(visit_each(self, n.args))
        return f'{fn_name}({fn_args})'

    @visit.register
    def _(self, n: GroupedExpr) -> str:
        return f'({self.visit(n.expr)})'

    @visit.register
    def _(self, n: ArrayAccess) -> str:
        idxes = ", ".join(visit_each(self, n.indices))
        return f'{self.visit(n.lhs)}({idxes})'

    @visit.register
    def _(self, n: StandardizedFunctionCall) -> str:
        if n.type not in self.standardized_function_calls:
            raise NotImplementedError(f'visit(StandardizedFunctionCall@{n.type}) not implemented in F90Visitor')

        fn_name = self.standardized_function_calls[n.type]
        fn_args = ", ".join(visit_each(self, n.args))
        return f'{fn_name}({fn_args})'

    @visit.register
    def _(self, n: SympyExpr) -> str:
        #exp: Expr = self.sympy_visitor.visit(n.expr)
        #return typing.cast(str, self.visit(exp))
        self.not_implemented(n)

    @visit.register
    def _(self, n: PrimitiveType) -> str:
        return n.representation

    @visit.register
    def _(self, _: IntentIn) -> str:
        return 'INTENT(IN)'

    @visit.register
    def _(self, _: IntentOut) -> str:
        return 'INTENT(OUT)'

    @visit.register
    def _(self, _: Allocatable) -> str:
        return 'ALLOCATABLE'

    @visit.register
    def _(self, n: Dimension) -> str:
        dims = ", ".join(self.visit(d) if d is not None else ':' for d in n.dims)
        return f'DIMENSION({dims})'

    @visit.register
    def _(self, n: TypeSpecifier) -> str:
        attrs = [self.visit(a) for a in n.attributes]
        if len(attrs) == 0:
            return typing.cast(str, self.visit(n.type))
        return f'{self.visit(n.type)}, {", ".join(attrs)}'

    @visit.register
    def _(self, n: VarDecl) -> str:
        return f'{self.visit(n.type)} :: {", ".join(visit_each(self, n.names))}'

    @visit.register
    def _(self, n: DoLoop) -> str:
        loop_header = f'DO {self.visit(n.induction_var)} = {self.visit(n.lower_bound)}, {self.visit(n.upper_bound)}'
        if n.step is not None:
            loop_header += f', {self.visit(n.step)}'
        body = "\n".join(visit_each(self, n.body))
        return f'{loop_header}\n{indent(body)}\nEND DO'

    @visit.register
    def _(self, n: Assignment) -> str:
        lhs = self.visit(n.lhs)
        if n.dimensions is not None:
            lhs += f'({", ".join(visit_each(self, n.dimensions))})'
        return f'{lhs} = {self.visit(n.rhs)}'

    @visit.register
    def _(self, n: SubroutineDecl) -> str:
        args = ", ".join(visit_each(self, n.args))
        body = "\n".join(visit_each(self, n.body))
        return f'SUBROUTINE {self.visit(n.name)}({args})\n{indent(body)}\nEND SUBROUTINE {self.visit(n.name)}'

    @visit.register
    def _(self, n: F90CodeRoot) -> str:
        return '\n'.join(visit_each(self, n.children))
