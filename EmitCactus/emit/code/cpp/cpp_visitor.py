import typing
from typing import Dict

from multimethod import multimethod

from EmitCactus.emit.ccl.schedule.schedule_tree import IntentRegion
from EmitCactus.emit.code.code_tree import CodeNode, StandardizedFunctionCallType, IdExpr, IntLiteralExpr, \
    FloatLiteralExpr, ExprStmt, SympyExpr, Expr, UnOpExpr, BinOpExpr, BinOp, NArityOpExpr, FunctionCall, \
    StandardizedFunctionCall, CarpetXGridLoopCall, CarpetXGridLoopLambda, ThornFunctionDecl, CodeRoot, IncludeDirective, \
    UsingNamespace, Using, UsingAlias, DeclareCarpetXArgs, DeclareCarpetArgs, DeclareCarpetParams, ConstAssignDecl, \
    ConstExprAssignDecl, ConstConstructDecl, VerbatimExpr, MutableAssignDecl, IfElseExpr, IfElseStmt
from EmitCactus.emit.code.sympy_visitor import SympyExprVisitor
from EmitCactus.emit.tree import Identifier, Integer, Verbatim, String, Bool, Float
from EmitCactus.emit.visitor import Visitor, visit_each
from EmitCactus.generators.cactus_generator import CactusGenerator
from EmitCactus.util import indent


class CppVisitor(Visitor[CodeNode]):
    generator: CactusGenerator
    sympy_visitor: SympyExprVisitor

    standardized_function_calls: Dict[StandardizedFunctionCallType, str] = {
        StandardizedFunctionCallType.Sin: 'sin',
        StandardizedFunctionCallType.Cos: 'cos',
        StandardizedFunctionCallType.Tan: 'tan',
        StandardizedFunctionCallType.Cot: 'cot',
        StandardizedFunctionCallType.Sec: 'sec',
        StandardizedFunctionCallType.Csc: 'csc',
        StandardizedFunctionCallType.Sinh: 'sinh',
        StandardizedFunctionCallType.Cosh: 'cosh',
        StandardizedFunctionCallType.Tanh: 'tanh',
        StandardizedFunctionCallType.Coth: 'coth',
        StandardizedFunctionCallType.Sech: 'sech',
        StandardizedFunctionCallType.Csch: 'csch',
        StandardizedFunctionCallType.Erf: 'erf',
        StandardizedFunctionCallType.Exp: 'exp',
        StandardizedFunctionCallType.Log: 'log'
    }

    def __init__(self, generator: CactusGenerator) -> None:
        self.generator = generator

        stencil_fns = {str(fn) for fn, fn_is_stencil in generator.thorn_def.is_stencil.items() if fn_is_stencil}

        def substitution_fn(name: str, in_stencil_args: bool) -> str:
            if not in_stencil_args and name in self.generator.var_names:
                return f'access({name})'
            return name

        self.sympy_visitor = SympyExprVisitor(
            stencil_fns=stencil_fns,
            substitution_fn=substitution_fn
        )

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
        return f'{n.fl:.1f}' if float(n.fl).is_integer() else f'{n.fl}'

    @visit.register
    def _(self, n: Verbatim) -> str:
        return n.text

    @visit.register
    def _(self, n: String) -> str:
        return f'"{n.text}"' if not n.single_quotes else f"'{n.text}'"

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
    def _(self, n: UnOpExpr) -> str:
        return f'({n.op.representation}({self.visit(n.e)}))'

    @visit.register
    def _(self, n: BinOpExpr) -> str:
        lhs = self.visit(n.lhs)
        rhs = self.visit(n.rhs)

        if n.op is BinOp.Pow:
            if isinstance(n.rhs, IntLiteralExpr) and n.rhs.integer == 2:
                return f'pow2({lhs})'
            elif isinstance(n.rhs, IntLiteralExpr):
                return f'pown({lhs}, {rhs})'
            elif n.rhs == BinOpExpr(lhs=FloatLiteralExpr(fl=1.0), op=BinOp.Div, rhs=FloatLiteralExpr(fl=2.0)):
                return f'sqrt({lhs})'
            elif n.rhs == BinOpExpr(lhs=FloatLiteralExpr(fl=1.0), op=BinOp.Div, rhs=FloatLiteralExpr(fl=3.0)):
                return f'cbrt({lhs})'
            else:
                return f'pow({lhs}, {rhs})'

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
        return f'if_else({self.visit(n.cond)}, {self.visit(n.then)}, {self.visit(n.else_)})'

    @visit.register
    def _(self, n: IfElseStmt) -> str:
        then = "\n".join(visit_each(self, n.then))
        else_ = "\n".join(visit_each(self, n.else_))

        return f'if ({self.visit(n.cond)}) {{\n{indent(then)}\n}} else {{\n{indent(else_)}\n}}'

    @visit.register
    def _(self, n: FunctionCall) -> str:
        fn_name = self.visit(n.name)
        fn_args = ", ".join(visit_each(self, n.args))
        if fn_name == "noop":
            return f'({fn_args})'

        if len(n.template_args) == 0:
            return f'{fn_name}({fn_args})'

        template_args = ", ".join(visit_each(self, n.template_args))

        return f'{fn_name}<{template_args}>({fn_args})'

    @visit.register
    def _(self, n: StandardizedFunctionCall) -> str:
        if n.type not in self.standardized_function_calls:
            raise NotImplementedError(f'visit(StandardizedFunctionCall@{n.type}) not implemented in CppVisitor')

        fn_name = self.standardized_function_calls[n.type]
        fn_args = ", ".join(visit_each(self, n.args))

        return f'{fn_name}({fn_args})'

    @visit.register
    def _(self, n: CarpetXGridLoopCall) -> str:
        centering_args = [f'{n.centering.string_repr}_centered[{i}]' for i in range(3)]
        loop_kind: str

        if n.write_destination is IntentRegion.Everywhere:
            loop_kind = 'all'
        elif n.write_destination is IntentRegion.Interior:
            loop_kind = 'int'
        else:
            assert n.write_destination is IntentRegion.Boundary
            loop_kind = 'bnd'

        return f"grid.loop_{loop_kind}_device<{', '.join(centering_args)}>(grid.nghostzones, {self.visit(n.fn)});"

    @visit.register
    def _(self, n: CarpetXGridLoopLambda) -> str:
        preceding = '\n'.join(visit_each(self, n.preceding))
        succeeding = '\n'.join(visit_each(self, n.succeeding))

        equations_list = list()
        for i, (lhs, rhs) in enumerate(n.equations):
            if str(lhs) in n.temporaries:
                if i in n.reassigned_lhses:
                    equations_list.append(f'{lhs} = {self.visit(rhs)};')
                else:
                    equations_list.append(f'auto {lhs} = {self.visit(rhs)};')
            else:
                equations_list.append(f'store({lhs}, {self.visit(rhs)});')

        equations = '\n'.join(equations_list)

        if len(preceding) > 0:
            preceding = '\n' + preceding

        if len(succeeding) > 0:
            succeeding = '\n' + succeeding

        return (f"[=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {{"
                f"{indent(preceding)}"
                f"\n{indent(equations)}"
                f"{indent(succeeding)}"
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
    def _(self, n: UsingAlias) -> str:
        return f"using {self.visit(n.lhs)} = {self.visit(n.rhs)};"

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
    def _(self, n: ConstAssignDecl) -> str:
        return f'const {self.visit(n.type)} {self.visit(n.lhs)} = {self.visit(n.rhs)};'

    @visit.register
    def _(self, n: MutableAssignDecl) -> str:
        return f'{self.visit(n.type)} {self.visit(n.lhs)} = {self.visit(n.rhs)};'

    @visit.register
    def _(self, n: ConstExprAssignDecl) -> str:
        return f'constexpr {self.visit(n.type)} {self.visit(n.lhs)} = {self.visit(n.rhs)};'

    @visit.register
    def _(self, n: ConstConstructDecl) -> str:
        return f'const {self.visit(n.type)} {self.visit(n.lhs)}({", ".join(visit_each(self, n.constructor_args))});'

    @visit.register
    def _(self, n: VerbatimExpr) -> str:
        return n.v.text
