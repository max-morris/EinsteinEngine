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

from dataclasses import dataclass
from typing import Type, Never, Optional, Any, cast, Sequence

from EinsteinEngine.emit.code.common.code_tree import IntLiteralExpr, FloatLiteralExpr
from multimethod import multimethod
from sympy import Symbol

from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend
from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.common.sympywrap import (
    free_symbols, mk_eq, mk_indexed_base, mk_symbol
)
from EinsteinEngine.emit.code.f90.f90_tree import PrimitiveType
from EinsteinEngine.emit.code.f90.f90_tree import F90ExprNode


@dataclass
class VanillaF90Param[T: int|float]:
    name: str
    type: Type[T]
    default: Optional[T]

    def get_type(self) -> PrimitiveType:
        if self.type is int:
            return PrimitiveType.Integer
        elif self.type is float:
            return PrimitiveType.Double
        else:
            raise ValueError(f"Unexpected type in VanillaF90Param: {self.type}")

    def get_default_expr(self) -> F90ExprNode:
        if self.default is None:
            raise ValueError(f"Param {self.name} has no default value")

        if self.type is int:
            return IntLiteralExpr(cast(int, self.default))
        elif self.type is float:
            return FloatLiteralExpr(cast(float, self.default))
        else:
            raise ValueError(f"Unexpected type in VanillaF90Param: {self.type}")


class VanillaF90Function(DslFunctionFrontend["VanillaF90Module"]):
    def __init__(self,
                 name: str,
                 frontend: "VanillaF90Module",
                 intent_override: Optional[IntentOverride] = None) -> None:
        super().__init__(name, frontend, intent_override, owner_name="VanillaF90Function")

class VanillaF90Module(DslFrontend[VanillaF90Param[Any], Never, VanillaF90Function]):
    name: str

    def __init__(
            self,
            name: str,
            *,
            dimensionality: int = 3,
            coords: Optional[Sequence[str]] = None,
            derivative_stencil_width: int = 5
    ) -> None:
        super().__init__(
            dimensionality=dimensionality,
            coords=coords,
            derivative_stencil_width=derivative_stencil_width
        )
        self.name = name


    def create_function(self,
                        name: str,
                        *,
                        intent_override: Optional[IntentOverride] = None) -> VanillaF90Function:
        tf = VanillaF90Function(name, self, intent_override)
        self.functions[name] = tf
        return tf

    @multimethod
    def add_param(self, name: str, type: Type[int] | Type[float]) -> Symbol:
        self.params[name] = VanillaF90Param(name, type, None)
        return mk_symbol(name)

    @add_param.register
    def _add_param_with_default(self, name: str, default: int|float) -> Symbol:
        self.params[name] = VanillaF90Param(name, type(default), default)
        return mk_symbol(name)
