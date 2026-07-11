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
from typing import Optional, List, Collection, Tuple, TypeAlias

import sympy as sy  # type: ignore[import-untyped]

from EinsteinEngine import Identifier, Centering, IntentRegion
from EinsteinEngine.emit.tree import LineComment, BlockComment, Verbatim
from EinsteinEngine.emit.code.common.code_tree import Directive, AnyNode, Decl, Expr, Stmt, CodeNode, CommonExpr, ExprStmt, IfElseStmt
from EinsteinEngine.generators.substitute_recycled_temporaries import RecycledTemporarySubstitution


@dataclass
class CppCarpetXExpr(Expr):
    pass


@dataclass
class CppCarpetXStmt(Stmt):
    pass


@dataclass
class CppCarpetXDirective(Directive):
    pass


@dataclass
class CppCarpetXDecl(Decl, CppCarpetXStmt):
    pass

CppCarpetXExprNode: TypeAlias = CppCarpetXExpr | CommonExpr
CppCarpetXStmtNode: TypeAlias = CppCarpetXStmt | CppCarpetXDecl | ExprStmt | IfElseStmt
CppCarpetXDirectiveNode: TypeAlias = CppCarpetXDirective
CppCarpetXTopLevelNode: TypeAlias = CppCarpetXStmtNode | CppCarpetXDirectiveNode | LineComment | BlockComment | Verbatim
CppCarpetXCodeElem: TypeAlias = CppCarpetXExprNode | CppCarpetXStmtNode | CppCarpetXDirectiveNode | Verbatim


@dataclass
class DeclareCarpetXArgs(CppCarpetXDirective):
    fn_name: Identifier


@dataclass
class DeclareCarpetArgs(CppCarpetXDirective):
    fn_name: Identifier


@dataclass
class DeclareCarpetParams(CppCarpetXDirective):
    pass


@dataclass(init=False)
class IncludeDirective(CppCarpetXDirective):
    header_name: Identifier
    quote_name: bool

    def __init__(self, header_name: Identifier, quote_name: bool = False):
        self.header_name = header_name
        self.quote_name = quote_name


@dataclass
class DefineDirective(CppCarpetXDirective):
    name: Identifier
    val: Optional[AnyNode]


@dataclass
class ConstAssignDecl(CppCarpetXDecl):
    type: Identifier
    lhs: Identifier
    rhs: CppCarpetXExprNode


@dataclass
class MutableAssignDecl(CppCarpetXDecl):
    type: Identifier
    lhs: Identifier
    rhs: CppCarpetXExprNode


@dataclass
class ConstExprAssignDecl(CppCarpetXDecl):
    type: Identifier
    lhs: Identifier
    rhs: CppCarpetXExprNode


@dataclass
class ConstConstructDecl(CppCarpetXDecl):
    type: Identifier
    lhs: Identifier
    constructor_args: List[CppCarpetXExprNode]


@dataclass
class UsingNamespace(CppCarpetXDecl):
    namespace_name: Identifier


@dataclass
class Using(CppCarpetXDecl):
    ids: List[Identifier]


@dataclass
class UsingAlias(CppCarpetXDecl):
    lhs: Identifier
    rhs: AnyNode


@dataclass
class ThornFunctionDecl(CppCarpetXDecl):
    name: Identifier
    body: List[CppCarpetXTopLevelNode]


@dataclass
class CarpetXGridLoopLambda(CppCarpetXExpr):
    preceding: Collection[CppCarpetXCodeElem]
    equations: List[Tuple[sy.Symbol, CppCarpetXExprNode]]
    annotations: dict[str, str]
    succeeding: Collection[CppCarpetXCodeElem]
    temporaries: Collection[str]
    reassigned_lhses: dict[int, RecycledTemporarySubstitution]


@dataclass
class CarpetXGridLoopCall(CppCarpetXStmt):
    centering: Centering
    write_destination: IntentRegion
    fn: CarpetXGridLoopLambda
    simd: bool


@dataclass
class CppCarpetXCodeRoot(CodeNode):
    children: List[CppCarpetXCodeElem]
