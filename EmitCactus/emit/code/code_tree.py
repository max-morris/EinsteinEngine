from dataclasses import dataclass
from enum import auto
from typing import Optional, Union, List, Collection, Tuple

import sympy as sy

from EmitCactus.emit.ccl.schedule.schedule_tree import IntentRegion
from EmitCactus.emit.tree import Node, Identifier, Verbatim, CommonNode, Centering
from EmitCactus.util import ReprEnum


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


class UnOp(ReprEnum):
    Neg = auto(), "-"


class BinOp(ReprEnum):
    Add = auto(), "+"
    Mul = auto(), "*"
    Pow = auto(), "^"
    Div = auto(), "/"
    Mod = auto(), "%"
    And = auto(), "&&"
    Or = auto(), "||"
    Eq = auto(), "=="
    Neq = auto(), "!="
    Lt = auto(), "<"
    Lte = auto(), "<="
    Gt = auto(), ">"
    Gte = auto(), ">="


@dataclass
class UnOpExpr(Expr):
    op: UnOp
    e: Expr


@dataclass
class BinOpExpr(Expr):
    lhs: Expr
    op: BinOp
    rhs: Expr


@dataclass
class NArityOpExpr(Expr):
    op: BinOp
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
class MutableAssignDecl(Decl):
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
    Sinh = auto(), 'sinh'
    Cosh = auto(), 'cosh'
    Tanh = auto(), 'tanh'
    Coth = auto(), 'coth'
    Sech = auto(), 'sech'
    Csch = auto(), 'csch'
    Sin = auto(), 'sin'
    Cos = auto(), 'cos'
    Tan = auto(), 'tan'
    Cot = auto(), 'cot'
    Sec = auto(), 'sec'
    Csc = auto(), 'csc'
    Exp = auto(), 'exp'
    Erf = auto(), 'erf'
    Log = auto(), 'log'
    # todo: There are definitely more of these


@dataclass
class StandardizedFunctionCall(Expr):
    type: StandardizedFunctionCallType
    args: List[Expr]


@dataclass
class CarpetXGridLoopLambda(Expr):
    preceding: Collection[CodeElem]
    equations: List[Tuple[sy.Symbol, Expr]]
    succeeding: Collection[CodeElem]
    temporaries: Collection[str]
    reassigned_lhses: Collection[int]


@dataclass
class CarpetXGridLoopCall(Stmt):
    centering: Centering
    write_destination: IntentRegion
    fn: CarpetXGridLoopLambda


@dataclass
class CodeRoot(CodeNode):
    children: List[CodeElem]
