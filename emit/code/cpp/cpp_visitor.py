import typing

from multimethod import multimethod

import util
from emit.code.code_tree import *
from emit.code.sympy_visitor import SympyExprVisitor
from emit.tree import *
from util import indent
from emit.visitor import Visitor, visit_each


class CppVisitor(Visitor[CodeNode]):
    sympy_visitor: SympyExprVisitor

    standardized_function_calls: dict[StandardizedFunctionCallType, str] = {
        StandardizedFunctionCallType.Sin: 'std::sin',
        StandardizedFunctionCallType.Cos: 'std::cos'
    }

    def __init__(self) -> None:
        self.sympy_visitor = SympyExprVisitor()

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
    def _(self, n: Verbatim) -> str:
        return n.text

    @visit.register
    def _(self, n: String) -> str:
        return f'"{n.text}"'

    @visit.register
    def _(self, n: Bool) -> str:
        return "true" if n.b else "false"

    @visit.register
    def _(self, n: Float) -> str:
        return f'{n.fl}'

    @visit.register
    def _(self, n: ExprStmt) -> str:
        return f'{self.visit(n.expr)};'

    @visit.register
    def _(self, n: SympyExpr) -> str:
        exp: Expr = self.sympy_visitor.visit(n.expr)
        return typing.cast(str, self.visit(exp))

    @visit.register
    def _(self, n: NArityOpExpr) -> str:
        assert n.op != Operator.Pow

        if len(n.args) == 0:
            return ''

        st: str = self.visit(n.args[0])
        for a in n.args[1:]:
            st += f' {n.op.representation} {self.visit(a)}'

        return st

    @visit.register
    def _(self, n: FunctionCall) -> str:
        fn_name = self.visit(n.name)
        fn_args = ",".join(visit_each(self, n.args))

        if len(n.template_args) == 0:
            return f'{fn_name}({fn_args})'

        template_args = ",".join(visit_each(self, n.template_args))

        return f'{fn_name}<{template_args}>({fn_args})'

    @visit.register
    def _(self, n: StandardizedFunctionCall) -> str:
        if n.type not in self.standardized_function_calls:
            raise NotImplementedError(f'visit(StandardizedFunctionCall@{n.type}) not implemented in CppVisitor')

        fn_name = self.standardized_function_calls[n.type]
        fn_args = ",".join(visit_each(self, n.args))

        return f'{fn_name}({fn_args})'

    @visit.register
    def _(self, n: CarpetXGridLoopCall) -> str:
        centering_args = [f'{n.centering.string_repr}_centered[{i}]' for i in range(3)]
        return f"grid.loop_int_device<{', '.join(centering_args)}, CCTK_VECSIZE>(grid.nghostzones, {self.visit(n.fn)});"

    @visit.register
    def _(self, n: CarpetXGridLoopLambda) -> str:
        equations = '\n'.join([f'{lhs} = {self.visit(rhs)};' for lhs, rhs in n.equations.items()])
        return (f"[=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {{\n"
                f"{indent(equations)}"
                f"\n}}")

    @visit.register
    def _(self, n: ThornFunctionDecl) -> str:
        body = '\n'.join(visit_each(self, n.body))
        return (f'void {self.visit(n.name)}(CCTK_ARGUMENTS) {{\n'
                f'{indent(body)}'
                f'\n}}')

    @visit.register
    def _(self, n: CodeRoot) -> str:
        return '\n'.join(visit_each(self, n.children))

    @visit.register
    def _(self, n: IncludeDirective) -> str:
        if n.quote_name:
            return f'#include "{self.visit(n.header_name)}"'
        else:
            return f'#include <{self.visit(n.header_name)}>'

    @visit.register
    def _(self, n: UsingNamespace) -> str:
        return f'using namespace {self.visit(n.namespace_name)};'

    @visit.register
    def _(self, n: Using) -> str:
        return f"using {','.join(visit_each(self, n.ids))};"

    @visit.register
    def _(self, n: DeclareCarpetXArgs) -> str:
        return f"DECLARE_CCTK_ARGUMENTSX_{self.visit(n.fn_name)};"

    @visit.register
    def _(self, n: DeclareCarpetArgs) -> str:
        return f"DECLARE_CCTK_ARGUMENTS_{self.visit(n.fn_name)};"

    @visit.register
    def _(self, _: DeclareCarpetParams) -> str:
        return f"DECLARE_CCTK_PARAMETERS;"

    @visit.register
    def _(self, n: ConstConstructDecl) -> str:
        return f'const {self.visit(n.type)} {self.visit(n.lhs)}({", ".join(visit_each(self, n.constructor_args))});'
