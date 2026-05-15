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

# mypy: disable-error-code=no-redef
# The above line suppresses an unfortunate interaction between MyPy and the intersection of ABC and multimethod.

from typing import Optional, cast

from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from multimethod import multimethod

from EinsteinEngine.frontend.dsl.relativity.use_indices import DivMakerVisitor, ApplyDiv, ApplyDivN, is_down
from sympy import Idx, Symbol, Expr, Indexed, IndexedBase, Function

from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend
from EinsteinEngine.frontend.dsl.relativity.use_indices import EinsteinNotationManager, IndexSubsVisitor
from EinsteinEngine.frontend.dsl.relativity.symmetries import Sym

from EinsteinEngine.common.sympywrap import free_symbols, Applier, UFunc, mk_idxes, Pow, mk_idx, mk_function

from EinsteinEngine.frontend.definitions import D, div, no_idx, stencil, dummy, DD, DDI, DX, DY, DZ, DXI, DYI, DZI, \
    noop, zero, one
from EinsteinEngine.intermediate.coef import coef
import sympy as sy


class RelativityDslFrontend[ParamData](DslFrontend[ParamData]):
    einstein_notation: EinsteinNotationManager
    subs: dict[Indexed | IndexedBase, Expr]
    symmetries: Sym
    div_makers: dict[str, DivMakerVisitor]
    apply_div: Applier
    funs1: dict[tuple[UFunc, Idx], Expr]
    funs2: dict[tuple[UFunc, Idx, Idx], Expr]
    fun_args: dict[str, int]

    def __init__(self, dimensionality: int = 3):
        super().__init__(dimensionality=dimensionality)
        self.einstein_notation = EinsteinNotationManager(dimensionality=dimensionality)
        self.symmetries = Sym()
        self.subs = dict()
        self.apply_div = ApplyDiv()
        self.funs1 = dict()
        self.funs2 = dict()
        self.fun_args = dict()

        self.div_makers = dict()
        self.div_makers["div"] = DivMakerVisitor(div)
        self.div_makers["D"] = DivMakerVisitor(D)

        for dmv in self.div_makers.values():
            dmv.params = self._mk_param_set()

        self._populate_globals()

    # @property
    # def lookup_pair(self) -> dict[Idx, Idx]:
    #     return self.einstein_notation.lookup_pair

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

    def set_derivative_stencil(self, n: int) -> None:
        assert n % 2 == 1, "n must be odd"
        assert n > 1, "n must be > 1"
        self.apply_div = ApplyDivN(n, self.funs1, self.funs2, self.fun_args, self.dimensionality)

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
            is_down_idx = is_down(idx)
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
            is_down_idx1 = is_down(idx1)
            is_down_idx2 = is_down(idx2)
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
