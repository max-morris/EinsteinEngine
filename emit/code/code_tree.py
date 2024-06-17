from dataclasses import dataclass
from enum import auto
from typing import Optional

import sympy as sy

from emit.tree import Node, Identifier, String, Verbatim, CommonNode
from util import ReprEnum, CenteringEnum


class Centering(CenteringEnum):
    VVV = auto(), 'VVV', (0, 0, 0)
    CVV = auto(), 'CVV', (1, 0, 0)
    VCV = auto(), 'VCV', (0, 1, 0)
    VVC = auto(), 'VVC', (0, 0, 1)
    CCV = auto(), 'CCV', (1, 1, 0)
    VCC = auto(), 'VCC', (0, 1, 1)
    CVC = auto(), 'CVC', (1, 0, 1)
    CCC = auto(), 'CCC', (1, 1, 1)


class CodeNode(Node):
    pass


AnyNode = CodeNode | CommonNode


@dataclass
class Expr(CodeNode):
    pass


@dataclass
class IdExpr(Expr):
    id: Identifier


@dataclass
class VerbatimExpr(Expr):
    v: Verbatim


class Operator(ReprEnum):
    Add = auto(), "+"
    Mul = auto(), "*"
    Pow = auto(), "^"


@dataclass
class BinOpExpr(Expr):
    lhs: Expr
    op: Operator
    rhs: Expr


@dataclass
class NArityOpExpr(Expr):
    op: Operator
    args: list[Expr]


@dataclass
class SympyExpr(Expr):
    expr: sy.Expr


@dataclass
class Stmt(CodeNode):
    pass


@dataclass
class Directive(CodeNode):
    pass


@dataclass
class DeclareCarpetXArgs(Directive):
    fn_name: Identifier


@dataclass
class DeclareCarpetArgs(Directive):
    fn_name: Identifier


@dataclass
class DeclareCarpetParams(Directive):
    pass


@dataclass(init=False)
class IncludeDirective(Directive):
    header_name: Identifier
    quote_name: bool

    def __init__(self, header_name: Identifier, quote_name: bool = False):
        self.header_name = header_name
        self.quote_name = quote_name


@dataclass
class DefineDirective(Directive):
    name: Identifier
    val: Optional[AnyNode]


@dataclass
class ExprStmt(Stmt):
    expr: Expr


@dataclass
class Decl(Stmt):
    pass


@dataclass
class ConstAssignDecl(Decl):
    type: Identifier
    lhs: Identifier
    rhs: Expr


@dataclass
class ConstConstructDecl(Decl):
    type: Identifier
    lhs: Identifier
    constructor_args: list[Expr]


@dataclass
class UsingNamespace(Decl):
    namespace_name: Identifier


@dataclass
class Using(Decl):
    ids: list[Identifier]


@dataclass
class UsingAlias(Decl):
    lhs: Identifier
    rhs: AnyNode


CodeElem = Stmt | Expr | Directive


@dataclass
class ThornFunctionDecl(Decl):
    name: Identifier
    body: list[CodeElem]


@dataclass
class FunctionCall(Expr):
    name: Identifier
    args: list[Expr]
    template_args: list[Expr | Identifier]


@dataclass
class CarpetXGridLoopLambda(Expr):
    equations: dict[str, SympyExpr]


@dataclass
class CarpetXGridLoopCall(Stmt):
    centering: Centering
    fn: CarpetXGridLoopLambda


@dataclass
class CodeRoot(CodeNode):
    children: list[CodeElem]
