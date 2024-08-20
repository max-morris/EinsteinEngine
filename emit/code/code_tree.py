from dataclasses import dataclass
from enum import auto
from typing import Optional, Union, List, Dict, Sequence

import sympy as sy
from dsl.sympywrap import Math

from emit.ccl.schedule.schedule_tree import IntentRegion
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


AnyNode = Union[CodeNode, CommonNode]


@dataclass
class Expr(CodeNode):
    pass


@dataclass
class IdExpr(Expr):
    id: Identifier


@dataclass
class IntLiteralExpr(Expr):
    integer: int


@dataclass
class FloatLiteralExpr(Expr):
    fl: float


@dataclass
class VerbatimExpr(Expr):
    v: Verbatim


class Operator(ReprEnum):
    Add = auto(), "+"
    Mul = auto(), "*"
    Pow = auto(), "^"
    Div = auto(), "/"


@dataclass
class BinOpExpr(Expr):
    lhs: Expr
    op: Operator
    rhs: Expr


@dataclass
class NArityOpExpr(Expr):
    op: Operator
    args: List[Expr]


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
class ConstExprAssignDecl(Decl):
    type: Identifier
    lhs: Identifier
    rhs: Expr


@dataclass
class ConstConstructDecl(Decl):
    type: Identifier
    lhs: Identifier
    constructor_args: List[Expr]


@dataclass
class UsingNamespace(Decl):
    namespace_name: Identifier


@dataclass
class Using(Decl):
    ids: List[Identifier]


@dataclass
class UsingAlias(Decl):
    lhs: Identifier
    rhs: AnyNode


CodeElem = Union[Stmt, Expr, Directive, Verbatim]


@dataclass
class ThornFunctionDecl(Decl):
    name: Identifier
    body: List[CodeElem]


@dataclass
class FunctionCall(Expr):
    name: Identifier
    args: List[Expr]
    template_args: List[Union[Expr, Identifier]]


class StandardizedFunctionCallType(ReprEnum):
    Sin = auto(), 'sin'
    Cos = auto(), 'cos'
    # todo: There are definitely more of these


@dataclass
class StandardizedFunctionCall(Expr):
    type: StandardizedFunctionCallType
    args: List[Expr]


@dataclass
class CarpetXGridLoopLambda(Expr):
    preceding: Sequence[CodeElem]
    equations: Dict[Math, SympyExpr]
    succeeding: Sequence[CodeElem]
    temporaries: Sequence[str]


@dataclass
class CarpetXGridLoopCall(Stmt):
    centering: Centering
    write_destination: IntentRegion
    fn: CarpetXGridLoopLambda


@dataclass
class CodeRoot(CodeNode):
    children: List[CodeElem]
