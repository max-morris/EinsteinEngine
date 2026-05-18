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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Set, NamedTuple, Iterable, Unpack, TypedDict, Any

from sympy import Symbol, Idx, Expr, IndexedBase

from EinsteinEngine.common.sympywrap import mk_symbol, Applier, UFunc, mk_function
from EinsteinEngine.frontend.frontend import Frontend
from EinsteinEngine.frontend.dsl.finite_difference import DivMakerVisitor, ApplyDivN
from EinsteinEngine.frontend.definitions import D, div

class OverwriteSymbolRecord(NamedTuple):
    symbol: IndexedBase
    resolves_to: IndexedBase

class SymbolDeclarationKwargs(TypedDict, total=False):
    pass

class MkCoordsKwargs(TypedDict, total=False):
    pass

@dataclass
class SymbolDeclaration[KwargsType: SymbolDeclarationKwargs]:
    basename: str
    base: IndexedBase
    indices: tuple[Idx, ...]
    kwargs: KwargsType

class DslFrontend[ParamDataT, SymbolDeclarationT: SymbolDeclaration[Any]](Frontend):
    dimensionality: int
    declarations: dict[str, SymbolDeclarationT]
    coords: list[Symbol]
    params: dict[str, ParamDataT]
    var2base: dict[str, str]

    overwrite_symbols: dict[str, OverwriteSymbolRecord]

    is_stencil: dict[UFunc, bool]  # Obsolesce sometime?

    div_makers: dict[str, DivMakerVisitor]
    apply_div: Applier
    funs1: dict[tuple[UFunc, Idx], Expr]  # Steve: help me understand what this does
    funs2: dict[tuple[UFunc, Idx, Idx], Expr]  # Steve: help me understand what this does
    fun_args: dict[str, int]  # Steve: help me understand what this does

    def __init__(self, *, dimensionality: int = 3, derivative_stencil_order: int = 5):
        super().__init__()
        self.dimensionality = dimensionality
        self.declarations = dict()
        self.coords = list()
        self.params = dict()
        self.var2base = dict()

        self.overwrite_symbols = dict()

        self.is_stencil = {
            mk_function("stencil"): True
        }

        self.funs1 = dict()
        self.funs2 = dict()
        self.fun_args = dict()
        self._set_derivative_stencil(derivative_stencil_order)

        self.div_makers = dict()
        self.div_makers["div"] = DivMakerVisitor(div)
        self.div_makers["D"] = DivMakerVisitor(D)

        for dmv in self.div_makers.values():
            dmv.params = self._mk_param_set()

    @abstractmethod
    def decl(self, basename: str, indices: Iterable[Idx], **kwargs: Unpack[SymbolDeclarationKwargs]) -> IndexedBase:
        ...

    @abstractmethod
    def _decl_scalar(self, basename: str) -> Symbol:
        ...

    @abstractmethod
    def mk_coords(self, **kwargs: Unpack[MkCoordsKwargs]) -> list[Symbol]:
        ...

    def _set_derivative_stencil(self, n: int) -> None:
        assert n % 2 == 1, "n must be odd"
        assert n > 1, "n must be > 1"
        self.apply_div = ApplyDivN(n, self.funs1, self.funs2, self.fun_args, self.dimensionality)

    def _mk_param_set(self) -> Set[Symbol]:
        ret: Set[Symbol] = set()
        for k in self.params:
            ret.add(mk_symbol(k))
        return ret
