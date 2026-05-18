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

from abc import abstractmethod, ABC
from dataclasses import dataclass
# mypy: disable-error-code=no-redef
# The above line suppresses an unfortunate interaction between MyPy and the intersection of ABC and multimethod.

from typing import Optional, cast, NamedTuple, TypedDict, Unpack, Iterable, Any

from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from multimethod import multimethod

from EinsteinEngine.frontend.dsl.finite_difference import DivMakerVisitor, ApplyDiv, ApplyDivN
from EinsteinEngine.frontend.dsl.relativity.use_indices import is_relativity_lower_idx, relativity_idx_to_int
from EinsteinEngine.common.util import checked_cast
from sympy import Idx, Expr, Indexed, IndexedBase, Function, Basic, Matrix, Symbol

from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend, SymbolDeclarationKwargs, SymbolDeclaration
from EinsteinEngine.frontend.dsl.relativity.use_indices import EinsteinNotationManager, IndexSubsVisitor
from EinsteinEngine.frontend.dsl.relativity.symmetries import Sym

from EinsteinEngine.common.sympywrap import Applier, UFunc, mk_idxes, Pow, mk_idx, mk_function, mk_zeros, do_subs

from EinsteinEngine.frontend.definitions import D, div, no_idx, stencil, dummy, DD, DDI, DX, DY, DZ, DXI, DYI, DZI, \
    noop, zero, one
from EinsteinEngine.intermediate.coef import coef
import sympy as sy

class RelativitySymbolDeclarationKwargs(SymbolDeclarationKwargs, total=False):
    pass

@dataclass
class RelativitySymbolDeclaration[KwargsType: RelativitySymbolDeclarationKwargs](SymbolDeclaration[KwargsType]):
    pass

class RelativityDslFrontend[ParamDataT, SymbolDeclarationT: RelativitySymbolDeclaration[Any]](
        DslFrontend[ParamDataT, SymbolDeclarationT], ABC
):
    einstein_notation: EinsteinNotationManager
    subs: dict[Indexed | IndexedBase, Expr]
    symmetries: Sym

    def __init__(self, *, dimensionality: int = 3, derivative_stencil_order: int = 5):
        super().__init__(
            dimensionality=dimensionality,
            derivative_stencil_order=derivative_stencil_order
        )
        self.einstein_notation = EinsteinNotationManager(dimensionality=dimensionality)
        self.symmetries = Sym()
        self.subs = dict()

        self._populate_globals()

    def mk_coords(self, with_time: bool = False) -> list[Symbol]:
        # Note that x, y, and z are special symbols
        if self.dimensionality == 3:
            if with_time:
                self.coords = [self._decl_scalar("t"), self._decl_scalar("x"), self._decl_scalar("y"),
                               self._decl_scalar("z")]
            else:
                self.coords = [self._decl_scalar("x"), self._decl_scalar("y"), self._decl_scalar("z")]
        elif self.dimensionality == 4:
            # TODO: No idea whether this works
            self.coords = [self._decl_scalar("t"), self._decl_scalar("x"), self._decl_scalar("y"),
                           self._decl_scalar("z")]
        else:
            raise DslException(f"Unsupported dimensionality {self.dimensionality}")
        return self.coords

    def get_matrix(self, ind: Indexed) -> Matrix:
        values: dict[Idx, Idx] = dict()
        result = mk_zeros(*tuple([self.dimensionality] * (len(ind.args) - 1)))
        ind_args: list[Idx] = [checked_cast(x, Idx) for x in ind.args[1:]]
        while self.einstein_notation.incr(ind_args, values):
            arr_idxs = tuple([relativity_idx_to_int(checked_cast(do_subs(x, values), Idx)) for x in ind_args])
            r = self._do_subs(ind, idx_subs=values)
            result[arr_idxs] = r
        return result

    def find_symmetries(self, foo: Basic) -> list[tuple[int, int, int]]:
        m_sym_list: list[tuple[int, int, int]] = list()
        # noinspection PyUnresolvedReferences
        if foo.is_Function and hasattr(foo, "name") and foo.name in ["div", "D"]:
            # This is a derivative
            if len(foo.args) == 3:
                # This is a 2nd derivative, symmetric in the last 2 args
                foo_arg1 = len(foo.args[0].args) - 1
                foo_arg2 = foo_arg1 + 1
                m_sym: tuple[int, int, int] = (foo_arg1, foo_arg2, 1)
                m_sym_list += [m_sym]
                m_sym_list += self.find_symmetries(foo.args[0])
            elif len(foo.args) == 2:
                m_sym_list += self.find_symmetries(foo.args[0])
            else:
                assert False, "Only handle 1st and 2nd derivatives"
        elif isinstance(foo, Indexed):
            k = foo.base
            return self.symmetries.sd.get(k, list())
        return m_sym_list

    def _populate_globals(self) -> None:
        """
        Populates the global namespace with a few generic up/down index pairs for use in the DSL.
        """

        ui, li = self.einstein_notation.mk_pair('i')
        uj, lj = self.einstein_notation.mk_pair('j')
        uk, lk = self.einstein_notation.mk_pair('k')
        ua, la = self.einstein_notation.mk_pair('a')
        ub, lb = self.einstein_notation.mk_pair('b')
        uc, lc = self.einstein_notation.mk_pair('c')
        ud, ld = self.einstein_notation.mk_pair('d')
        u0, l0 = self.einstein_notation.mk_pair('0')
        u1, l1 = self.einstein_notation.mk_pair('1')
        u2, l2 = self.einstein_notation.mk_pair('2')
        u3, l3 = self.einstein_notation.mk_pair('3')
        u4, l4 = self.einstein_notation.mk_pair('4')
        u5, l5 = self.einstein_notation.mk_pair('5')
        up_indices = u0, u1, u2, u3, u4, u5
        down_indices = l0, l1, l2, l3, l4, l5

        globals().update(locals())

    def mk_pair(self, s: str | None = None) -> tuple[Idx, Idx]:
        """
        Returns a tuple containing an upper/lower index pair.
        """

        return self.einstein_notation.mk_pair(s)

    def _do_div(self, expr: Expr) -> Expr:
        params = self._mk_param_set()
        r = expr
        for k, v in self.div_makers.items():
            v.params = params
            r = v.visit(r, no_idx)
        return r

    def _do_subs(self, arg: Expr, idx_subs: Optional[dict[Idx, Idx]] = None) -> Expr:
        isub = IndexSubsVisitor(self.subs)
        arg1 = arg
        for i in range(20):
            new_arg = arg1
            new_arg = self.einstein_notation.expand_contracted_indices(new_arg, self.symmetries)
            new_arg = cast(Expr, self.symmetries.apply(new_arg))

            isub.idx_subs = idx_subs if idx_subs is not None else dict()
            new_arg = isub.visit(new_arg)
            new_arg = self._do_div(new_arg)
            if new_arg == arg1:
                return new_arg
            arg1 = new_arg
        raise Exception(arg)

    @multimethod
    def mk_stencil(self, func_name: str, idx: Idx, expr: Expr) -> UFunc:
        result = self.mk_stencil(func_name, expr, [idx])
        assert isinstance(result, UFunc)
        self.div_makers[func_name] = DivMakerVisitor(result)
        return result

    @mk_stencil.register
    def _mk_stencil(self, func_name: str, idx_a1: Idx, idx_a2: Idx, expr_a: Expr,
          idx_b1: Idx, idx_b2: Idx, expr_b: Expr) -> UFunc:
        self.mk_stencil(func_name, idx_a1, idx_a2, expr_a)
        result = self.mk_stencil(func_name, idx_b1, idx_b2, expr_b)
        assert isinstance(result, UFunc)
        return result

    @mk_stencil.register
    def _mk_stencil(self, func_name: str, idx1: Idx, idx2: Idx, expr: Expr) -> UFunc:
        result = self.mk_stencil(func_name, expr, [idx1, idx2])
        assert isinstance(result, UFunc)
        self.div_makers[func_name] = DivMakerVisitor(result)
        return result

    @mk_stencil.register
    def _mk_stencil(self, func_name: str, expr: Expr, idx_list: list[Idx]) -> UFunc:

        @multimethod
        def mk_sten(idx_map: dict[Idx, Idx], expr: Function) -> Expr:
            # TODO: Rewrite so it does not require 3 dimensions
            assert self.dimensionality == 3
            l0, l1, l2 = mk_idxes('l0 l1 l2')

            if expr.func == stencil:
                if len(expr.args) != 1:
                    raise DslException(expr)
                arg = mk_sten(idx_map, expr.args[0])
                c0 = coef(l0, arg)
                c1 = coef(l1, arg)
                c2 = coef(l2, arg)
                ret = stencil(dummy, c0, c1, c2)
                assert isinstance(ret, Expr)
                return ret
            elif expr.func == DD:
                if len(expr.args) != 1:
                    raise DslException(expr)
                arg = mk_sten(idx_map, expr.args[0])
                if arg == l0:
                    return DX
                elif arg == l1:
                    return DY
                elif arg == l2:
                    return DZ
                assert False
            elif expr.func == DDI:
                if len(expr.args) != 1:
                    raise DslException(expr)
                arg = mk_sten(idx_map, expr.args[0])
                if arg == l0:
                    return DXI
                elif arg == l1:
                    return DYI
                elif arg == l2:
                    return DZI
                assert False
            elif expr.func == noop:
                arg = mk_sten(idx_map, expr.args[0])
                retv: Expr = noop(arg)
                return retv
            else:
                raise DslException("Bad Func")

        @mk_sten.register
        def _mk_sten(_idx_map: dict[Idx, Idx], expr: sy.Float) -> Expr:
            return expr

        @mk_sten.register
        def _mk_sten(_idx_map: dict[Idx, Idx], expr: sy.Integer) -> Expr:
            return expr

        @mk_sten.register
        def _mk_sten(_idx_map: dict[Idx, Idx], expr: sy.Rational) -> Expr:
            return expr

        @mk_sten.register
        def _mk_sten(idx_map: dict[Idx, Idx], expr: sy.Pow) -> Expr:
            result: Expr = Pow(mk_sten(idx_map, expr.args[0]), expr.args[1])
            return result

        @mk_sten.register
        def _mk_sten(idx_map: dict[Idx, Idx], expr: Idx) -> Expr:
            retval = idx_map.get(expr, expr)
            return retval

        @mk_sten.register
        def _mk_sten(idx_map: dict[Idx, Idx], expr: sy.Add) -> Expr:
            ret = zero
            for a in expr.args:
                term = mk_sten(idx_map, a)
                ret += term
            return ret

        @mk_sten.register
        def _mk_sten(idx_map: dict[Idx, Idx], expr: sy.Mul) -> Expr:
            ret = one
            for a in expr.args:
                term = mk_sten(idx_map, a)
                ret *= term
            return ret

        func = mk_function(func_name)

        if len(idx_list) == 1 or (len(idx_list) == 2 and idx_list[0] == idx_list[1]):
            idx = idx_list[0]
            is_down_idx = is_relativity_lower_idx(idx)
            for i in range(self.dimensionality):
                if is_down_idx:
                    idx0 = mk_idx(f'l{i}')
                else:
                    idx0 = mk_idx(f'u{i}')
                result = mk_sten({idx: idx0}, expr)
                self.funs1[(func, idx0)] = result
        elif len(idx_list) == 2:
            idx1 = idx_list[0]
            idx2 = idx_list[1]
            is_down_idx1 = is_relativity_lower_idx(idx1)
            is_down_idx2 = is_relativity_lower_idx(idx2)
            for i in range(self.dimensionality):
                if is_down_idx1:
                    idx10 = mk_idx(f'l{i}')
                else:
                    idx10 = mk_idx(f'u{i}')
                for j in range(self.dimensionality):
                    if i == j:
                        continue
                    if is_down_idx2:
                        idx20 = mk_idx(f'l{j}')
                    else:
                        idx20 = mk_idx(f'u{j}')
                    result = mk_sten({idx1: idx10, idx2: idx20}, expr)
                    self.funs2[(func, idx10, idx20)] = result

        return func
