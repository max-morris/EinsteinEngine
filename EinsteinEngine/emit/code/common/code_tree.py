#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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

from dataclasses import dataclass
from enum import auto
from typing import Union, List, TypeAlias, Sequence

import sympy as sy

from EinsteinEngine.emit.tree import Node, Identifier, Verbatim, CommonNode, BlockComment, LineComment
from EinsteinEngine.common.util import ReprEnum


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
    Sub = auto(), "-"
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
class IfElseExpr(Expr):
    cond: Expr
    then: Expr
    else_: Expr


@dataclass
class SympyExpr(Expr):
    expr: sy.Expr


@dataclass
class Stmt(CodeNode):
    pass


@dataclass
class Directive(CodeNode):
    pass


TopLevelNode = Stmt | Directive | LineComment | BlockComment | Verbatim


@dataclass
class ExprStmt(Stmt):
    expr: Expr


@dataclass
class IfElseStmt(Stmt):
    cond: Expr
    then: Sequence[TopLevelNode]
    else_: Sequence[TopLevelNode]


@dataclass
class Decl(Stmt):
    pass


CodeElem = Union[Stmt, Expr, Directive, Verbatim]


@dataclass
class FunctionCall(Expr):
    name: Identifier
    args: List[Expr]
    template_args: List[Union[Expr, Identifier]]


@dataclass
class GroupedExpr(Expr):
    expr: Expr


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


CommonExpr: TypeAlias = (
    IdExpr
    | IntLiteralExpr
    | FloatLiteralExpr
    | VerbatimExpr
    | UnOpExpr
    | BinOpExpr
    | NArityOpExpr
    | IfElseExpr
    | SympyExpr
    | FunctionCall
    | StandardizedFunctionCall
)

