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

from typing import Any, Callable, cast, List


from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from multimethod import multimethod
from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sympy import Symbol, Expr, Basic, Indexed, Idx, IndexedBase, Matrix

from EinsteinEngine.common.util import vprint
from EinsteinEngine.common.sympywrap import free_symbols
from EinsteinEngine.frontend.dsl.dsl_frontend import mk_mk_subst
from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend
from EinsteinEngine.frontend.dsl.use_indices import do_isub, to_num_tup, idx_to_int
from EinsteinEngine.intermediate.eqnlist import EqnList


class AddEqnManager:
    def __init__(
            self,
            frontend: DslFrontend[Any, Any, Any],
            eqn_list_getter: Callable[[], EqnList],
            is_baked: Callable[[], bool],
            *,
            owner_name: str
    ) -> None:
        self.frontend = frontend
        self._eqn_list_getter = eqn_list_getter
        self._is_baked = is_baked
        self._owner_name = owner_name

    @property
    def _eqn_list(self) -> EqnList:
        return self._eqn_list_getter()

    def _assert_not_baked(self) -> None:
        if self._is_baked():
            raise DslException(f"add_eqn should not be called on a baked {self._owner_name}")

    def _base_add_eqn(self, lhs2: Symbol, rhs2: Expr) -> None:
        """
        The base case of add_eqn. Assumes the LHS has already been flattened.
        """

        rhs2 = self.frontend._do_subs(self.frontend.einstein_notation.expand_contracted_indices(rhs2, self.frontend.symmetries))
        for item in free_symbols(rhs2):
            if str(item) in self.frontend.params:
                assert item.is_Symbol
                self._eqn_list.add_param(item)
        divs = self.frontend.apply_div

        rhs2_: Basic = do_isub(rhs2)
        assert isinstance(rhs2_, Expr)
        rhs2_ = divs.apply(rhs2_)
        assert isinstance(rhs2_, Expr)
        rhs2 = rhs2_

        self._eqn_list.add_eqn(lhs2, rhs2)
        vprint(colorize("Add eqn:", "green"), lhs2, colorize("->", "cyan"), rhs2)

    @multimethod
    def add_eqn(self, lhs: Indexed, rhs: Expr) -> None:
        self.frontend.einstein_notation.check_indices(rhs, self.frontend.declarations)
        self._assert_not_baked()

        lhs2: Symbol
        if self.frontend.get_free_indices(lhs) != self.frontend.get_free_indices(rhs):
            raise DslException(f"Free indices of '{lhs}' and '{rhs}' do not match.")
        count = 0
        for tup in self.frontend.einstein_notation.expand_free_indices(lhs, self.frontend.symmetries):
            count += 1
            lhs_x, idxs, _ = tup
            lhs2_: Basic = do_isub(lhs_x, self.frontend.subs)
            if not isinstance(lhs2_, Symbol):
                mms = mk_mk_subst(repr(lhs2_))
                raise Exception(f"'{lhs2_}' does not evaluate a Symbol. Did you forget to call mk_subst({mms},...)?")
            lhs2 = lhs2_
            rhs2 = self.frontend._do_subs(rhs, idxs)
            self._base_add_eqn(lhs2, rhs2)
        if count == 0:
            for ind in lhs.args[1:]:
                assert isinstance(ind, Idx)
                assert self.frontend.einstein_notation.is_numeric_index(ind)
            lhs2 = cast(Symbol, self.frontend._do_subs(lhs))
            rhs2 = self.frontend._do_subs(rhs)
            self._base_add_eqn(lhs2, rhs2)

    @add_eqn.register
    def _(self, lhs: IndexedBase, rhs: Expr) -> None:
        self._assert_not_baked()

        lhs2 = cast(Symbol, self.frontend._do_subs(lhs))
        eci = self.frontend.einstein_notation.expand_contracted_indices(rhs, self.frontend.symmetries)
        rhs2 = do_isub(eci)
        self._base_add_eqn(lhs2, rhs2)

    @add_eqn.register
    def _(self, lhs: Indexed, rhs: Matrix) -> None:
        self._assert_not_baked()

        count = 0
        for tup in self.frontend.einstein_notation.expand_free_indices(lhs, self.frontend.symmetries):
            count += 1
            lhs_x, idxs, _ = tup
            lhs2_ = do_isub(lhs_x, self.frontend.subs)
            lhs2 = lhs2_
            arr_idxs = to_num_tup(lhs.args[1:], idxs)
            rhs0 = rhs[arr_idxs]
            rhs2 = self.frontend._do_subs(rhs0, idxs)
            assert isinstance(lhs2, Symbol)
            self._base_add_eqn(lhs2, rhs2)
        assert count > 0

    @add_eqn.register
    def _(self, lhs: Indexed, rhs: List[Expr]) -> None:
        self._assert_not_baked()

        count = 0
        for tup in self.frontend.einstein_notation.expand_free_indices(lhs, self.frontend.symmetries):
            count += 1
            lhs_x, idxs, idx = tup
            num_idx = idx_to_int(idxs[idx[0]])
            lhs2_ = do_isub(lhs_x, self.frontend.subs)
            lhs2 = lhs2_
            rhs2 = rhs[num_idx]
            assert isinstance(lhs2, Symbol)
            self._base_add_eqn(lhs2, rhs2)
        assert count > 0
