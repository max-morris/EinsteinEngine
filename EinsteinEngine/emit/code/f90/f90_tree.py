#  Copyright (C) 2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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
from typing import TypeAlias, Optional, Sequence

from EinsteinEngine.emit.code.common.code_tree import Decl, Directive, Stmt, Expr, CommonExpr, ExprStmt, IfElseStmt, CodeNode, IntLiteralExpr
from EinsteinEngine.emit.tree import LineComment, BlockComment, Verbatim, Identifier
from EinsteinEngine.common.util import ReprEnum


@dataclass
class F90Expr(Expr):
    pass


@dataclass
class F90Stmt(Stmt):
    pass


@dataclass
class F90Directive(Directive):
    pass


@dataclass
class F90Decl(Decl, F90Stmt):
    pass

F90ExprNode: TypeAlias = F90Expr | CommonExpr
F90StmtNode: TypeAlias = F90Stmt | F90Decl | ExprStmt | IfElseStmt
F90DirectiveNode: TypeAlias = F90Directive
F90TopLevelNode: TypeAlias = F90StmtNode | F90DirectiveNode | LineComment | BlockComment | Verbatim
F90CodeElem: TypeAlias = F90ExprNode | F90StmtNode | F90DirectiveNode | Verbatim

@dataclass
class TypeAttribute(CodeNode):
    pass

class PrimitiveType(ReprEnum):
    Integer = auto(), 'INTEGER'
    Double = auto(), 'DOUBLE PRECISION'

@dataclass
class IntentIn(TypeAttribute):
    pass

@dataclass
class IntentOut(TypeAttribute):
    pass

@dataclass
class Allocatable(TypeAttribute):
    pass

@dataclass
class Dimension(TypeAttribute):
    dims: Sequence[Optional[IntLiteralExpr]]

@dataclass
class TypeSpecifier(CodeNode):
    type: PrimitiveType
    attributes: Sequence[TypeAttribute]

@dataclass
class VarDecl(F90Decl):
    type: TypeSpecifier
    names: Sequence[Identifier]

@dataclass
class Block(F90Stmt):
    body: Sequence[F90TopLevelNode]

@dataclass
class DoLoop(F90Stmt):
    induction_var: Identifier
    lower_bound: F90ExprNode
    upper_bound: F90ExprNode
    step: Optional[F90ExprNode]
    body: Sequence[F90TopLevelNode]

@dataclass
class Assignment(F90Stmt):
    lhs: Identifier
    dimensions: Optional[Sequence[F90ExprNode]]
    rhs: F90ExprNode

@dataclass
class ArrayAccess(F90Expr):
    lhs: Identifier
    indices: Sequence[F90ExprNode]

@dataclass
class SubroutineDecl(F90Decl):
    name: Identifier
    args: Sequence[Identifier]
    body: Sequence[F90TopLevelNode]

@dataclass
class F90CodeRoot(CodeNode):
    children: Sequence[F90CodeElem]
