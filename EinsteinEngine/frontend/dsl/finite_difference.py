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

from typing import *

from multimethod import multimethod

import sympy as sy

# noinspection PyUnusedImports
# MyPy needs Idx, Expr
from sympy import Symbol, Basic, Idx, Expr

from EinsteinEngine.common.sympywrap import *
from EinsteinEngine.frontend.definitions import *
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.frontend.dsl.relativity.use_indices import is_numeric_relativity_index, relativity_idx_to_int

from nrpy.finite_difference import setup_FD_matrix__return_inverse_lowlevel

DXI = mk_symbol("DXI")
DYI = mk_symbol("DYI")
DZI = mk_symbol("DZI")
L0, U0, L1, U1, L2, U2 = cast(Tuple[Idx, Idx, Idx, Idx, Idx, Idx], mk_idxes('l0 u0 l1 u1 l2 u2'))


def _mk_div(div_fun: UFunc, expr: Expr, *args: Idx) -> Expr:
    r = div_fun(expr, *args)
    assert isinstance(r, Expr)
    return r


class DivMakerVisitor:
    def __init__(self, div_fun: UFunc, coords: Optional[List[Symbol]] = None) -> None:
        self.div_func = div_fun
        self.div_name = str(div_fun)
        self.params: Set[Symbol] = set()
        if coords is None:
            coords = [x, y, z]
        self.coords = coords
        self.idx_map = dict()
        for i in range(len(coords)):
            self.idx_map[coords[i]] = mk_idx(f"l{i}")

    @multimethod
    def visit(self, expr: sy.Basic, idx: sy.Idx) -> Expr:
        raise Exception(str(expr) + " " + str(type(expr)))

    standard_fn_divs: dict[sy.Function, Callable[['DivMakerVisitor', sy.Expr, sy.Idx], sy.Expr]] = {
        sy.sin: lambda self, r, idx: cos(r) * self.visit(r, idx),
        sy.cos: lambda self, r, idx: -sin(r) * self.visit(r, idx),
        sy.tan: lambda self, r, idx: sec(r) ** 2 * self.visit(r, idx),
        sy.cot: lambda self, r, idx: -csc(r) ** 2 * self.visit(r, idx),
        sy.sec: lambda self, r, idx: sec(r) * tan(r) * self.visit(r, idx),
        sy.csc: lambda self, r, idx: -csc(r) * cot(r) * self.visit(r, idx),
        sy.exp: lambda self, r, idx: exp(r) * self.visit(r, idx),
        sy.log: lambda self, r, idx: (1 / r) * self.visit(r, idx),
        sy.cosh: lambda self, r, idx: sinh(r) * self.visit(r, idx),
        sy.sinh: lambda self, r, idx: cosh(r) * self.visit(r, idx),
        sy.tanh: lambda self, r, idx: sech(r) ** 2 * self.visit(r, idx),
        sy.coth: lambda self, r, idx: -csch(r) ** 2 * self.visit(r, idx),
        sy.sech: lambda self, r, idx: -sech(r) * tanh(x) * self.visit(r, idx),
        sy.csch: lambda self, r, idx: -csch(r) * coth(x) * self.visit(r, idx),
        sy.erf: lambda self, r, idx: 2 * exp(-r ** 2) / sqrt(pi) * self.visit(r, idx)
    }

    @visit.register
    def _(self, expr: sy.Add, idx: sy.Idx) -> Expr:
        r = zero
        for a in expr.args:
            r += self.visit(a, idx)
        return r

    @visit.register
    def _(self, expr: sy.Mul, idx: sy.Idx) -> Expr:
        if idx is not no_idx:
            s = zero
            for i in range(len(expr.args)):
                term = one
                for j in range(len(expr.args)):
                    a = expr.args[j]
                    if i == j:
                        term *= self.visit(a, idx)
                    else:
                        term *= self.visit(a, no_idx)
                s += term
            return s
        else:
            s = one
            for a in expr.args:
                s *= self.visit(a, no_idx)
            return s

    @visit.register
    def _(self, expr: sy.Symbol, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        ####
        # TODO: generalize for other dimensions than 3
        # assert get_dimension()==3

        if idx == L0:
            if expr == x:
                return one
            elif expr in [y, z]:
                return zero
            elif expr in self.params:
                return zero

        elif idx == L1:
            if expr == y:
                return one
            elif expr in [x, z]:
                return zero
            elif expr in self.params:
                return zero

        elif idx == L2:
            if expr == z:
                return one
            elif expr in [x, y]:
                return zero
            elif expr in self.params:
                return zero

        else:
            raise Exception(f"Bad index passed to derivative: {expr}: idx={idx}")

        return _mk_div(self.div_func, expr, idx)

    @visit.register
    def _(self, expr: sy.Integer, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.core.numbers.Pi, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Rational, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Float, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Idx, idx: sy.Idx) -> Expr:
        raise DslException("Derivative of Index")

    @visit.register
    def _(self, expr: sy.Piecewise, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Indexed, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return _mk_div(self.div_func, expr, idx)

    @visit.register
    def _(self, expr: sy.IndexedBase, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return _mk_div(self.div_func, expr, idx)

    @visit.register
    def _(self, expr: sy.Function, idx: sy.Idx) -> Expr:
        r = expr.args[0]

        if not isinstance(r, Expr):
            raise DslException("Expected the first argument/term of " + str(expr) + " to be an expression")

        name = expr.func.__name__
        if name == self.div_name:
            # Handle div of div
            sub: Expr = self.visit(r, no_idx)
            if len(expr.args) > 2:
                for idx1 in expr.args[1:]:
                    sub = self.visit(sub, idx1)
                return sub
            if isinstance(sub, sy.Function) and sub.func.__name__ == self.div_name:
                args = sorted(sub.args[1:] + expr.args[1:], key=lambda x: str(x))
                return _mk_div(self.div_func, cast(Expr, sub.args[0]), *args)

            for idx1 in expr.args[1:]:
                sub = self.visit(sub, idx1)

            if idx is not no_idx:
                sub = self.visit(self.div_func(sub, idx), no_idx)

            return sub
        elif idx is no_idx:
            return expr
        else:
            if expr.func in self.standard_fn_divs:
                f = self.standard_fn_divs[expr.func](self, r, idx)
            elif len(expr.args) == 1:
                fd = mk_function(name + "'")
                f = fd(r) * self.visit(r, idx)
            else:
                raise DslException(f"Derivative of {expr} is not handled by EinsteinEngine")
            assert isinstance(f, Expr)
            return f

    @visit.register
    def _(self, expr: sy.Pow, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        else:
            r = expr.args[0]
            n = expr.args[1]
            ret = n * r ** (n - 1) * self.visit(r, idx)
            assert isinstance(ret, Expr)
            return ret


dmv = DivMakerVisitor(div)
dmv2 = DivMakerVisitor(D)


def do_div(expr: Basic) -> Expr:
    r = dmv.visit(expr, no_idx)
    r = dmv2.visit(r, no_idx)
    assert isinstance(r, Expr)
    return r


def mk_term(v: Basic, i: int, j: int, k: int) -> Any:
    """
    Create a stencil term for output. Note that
    the 0,0,0 element is special.
    """
    if i == 0 and j == 0 and k == 0:
        return v
    else:
        return stencil(v, i, j, k)


def sort_exprs(expr: Tuple[Any, Any]) -> float:
    sort_key: float = 2 * expr[0].p / expr[0].q
    if sort_key < 0:
        sort_key = -sort_key + 1
    return sort_key


class ApplyDivN(Applier):
    """
    Use NRPy to calculate the stencil coefficients.
    """

    def __init__(self, n: int, unary_custom_stencils: Dict[Tuple[UFunc, Idx], Expr], binary_custom_stencils: Dict[Tuple[UFunc, Idx, Idx], Expr],
                 ufunc_arities: Dict[str, int], dimensionality: int) -> None:
        self.val: Optional[Expr] = None
        self.n = n
        self.fd_matrix = setup_FD_matrix__return_inverse_lowlevel(n, 0)
        self.unary_custom_stencils = unary_custom_stencils
        self.binary_custom_stencils = binary_custom_stencils
        self.ufunc_arities = ufunc_arities
        self.dimensionality = dimensionality

    def is_user_func(self, f: Expr) -> Optional[Expr]:
        f_func = f.func
        if not f.is_Function or not isinstance(f_func, UFunc):
            return None
        # noinspection PyUnresolvedReferences
        if hasattr(f, "name") and f.name in self.ufunc_arities:
            nargs = self.ufunc_arities[f.name]
            if len(f.args) != nargs:
                raise DslException(
                    f"function {f.name} called with wrong number of args. Expected {nargs}, got {len(f.args)}. Expr: {f}")
            return None
        elif len(f.args) == 2:
            _, arg1 = f.args
            if not isinstance(arg1, Idx):
                raise DslException(f"Expected an index argument but found {type(arg1)} in call {f}")
            return self.unary_custom_stencils.get((f_func, arg1), None)
        elif len(f.args) == 3:
            _, arg1, arg2 = f.args

            if not isinstance(arg1, Idx):
                raise DslException(f"Expected an index argument but found {type(arg1)} in first argument in call {f}")
            if not isinstance(arg2, Idx):
                raise DslException(f"Expected an index argument but found {type(arg2)} in second argument in call {f}")

            if arg1 == arg2:
                return self.unary_custom_stencils.get((f_func, arg1), None)
            else:
                # noinspection PyTypeChecker
                return self.binary_custom_stencils.get((f_func, arg1, arg2), None)
        return None

    def m(self, expr: Expr) -> bool:
        # noinspection PyUnresolvedReferences
        if (fun_def := self.is_user_func(expr)) is not None:
            arg0 = expr.args[0]
            if not isinstance(arg0, Expr):
                raise DslException("Expected the first argument/term of " + str(expr) + " to be an expression")
            self.val = do_subs(fun_def, {dummy: arg0})
            return True
        elif expr.is_Function and hasattr(expr, "name") and expr.name == "stencil":
            new_expr1: List[int | sy.Integer] = list()
            for arg in expr.args[1:]:
                if isinstance(arg, Idx):
                    new_expr1.append(relativity_idx_to_int(arg))
                elif isinstance(arg, sy.Integer):
                    new_expr1.append(arg)
                else:
                    assert False, f"arg={arg}, type={type(arg)}"
            self.val = expr.func(expr.args[0], *new_expr1)
            return True

        elif expr.is_Function and hasattr(expr, "name") and expr.name in ["div", "D"]:
            new_expr = list()
            dxt = sympify(1)

            if len(expr.args) == 2:
                coefs = self.fd_matrix.col(1)
                if expr.args[1] == L0:
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mk_term(expr.args[0], i - len(coefs) // 2, 0, 0))]
                    dxt = DXI
                elif expr.args[1] == L1:
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mk_term(expr.args[0], 0, i - len(coefs) // 2, 0))]
                    dxt = DYI
                elif expr.args[1] == L2:
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mk_term(expr.args[0], 0, 0, i - len(coefs) // 2))]
                    dxt = DZI
            elif len(expr.args) == self.dimensionality:
                if expr.args[1:] == (L0, L0):
                    coefs = 2 * self.fd_matrix.col(2)
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mk_term(expr.args[0], i - len(coefs) // 2, 0, 0))]
                    dxt = DXI ** 2
                elif expr.args[1:] == (L1, L1):
                    coefs = 2 * self.fd_matrix.col(2)
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mk_term(expr.args[0], 0, i - len(coefs) // 2, 0))]
                    dxt = DYI ** 2
                elif expr.args[1:] == (L2, L2):
                    coefs = 2 * self.fd_matrix.col(2)
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mk_term(expr.args[0], 0, 0, i - len(coefs) // 2))]
                    dxt = DZI ** 2
                elif expr.args[1:] in ((L0, L1), (L1, L0)):
                    coefs = self.fd_matrix.col(1)
                    for i in range(len(coefs)):
                        term_i = coefs[i]
                        for j in range(len(coefs)):
                            term = coefs[j] * term_i
                            new_expr += [(term, mk_term(expr.args[0], i - len(coefs) // 2, j - len(coefs) // 2, 0))]
                    dxt = DXI * DYI
                elif expr.args[1:] in ((L0, L2), (L2, L0)):
                    coefs = self.fd_matrix.col(1)
                    for i in range(len(coefs)):
                        term_i = coefs[i]
                        for j in range(len(coefs)):
                            term = coefs[j] * term_i
                            new_expr += [(term, mk_term(expr.args[0], i - len(coefs) // 2, 0, j - len(coefs) // 2))]
                    dxt = DXI * DZI
                elif expr.args[1:] in ((L1, L2), (L2, L1)):
                    coefs = self.fd_matrix.col(1)
                    for i in range(len(coefs)):
                        term_i = coefs[i]
                        for j in range(len(coefs)):
                            term = coefs[j] * term_i
                            new_expr += [(term, mk_term(expr.args[0], 0, i - len(coefs) // 2, j - len(coefs) // 2))]
                    dxt = DYI * DZI
                else:
                    raise Exception()

            if len(new_expr) > 0:
                new_expr = sorted(new_expr, key=sort_exprs)
                self.val = sympify(0)
                i = 0
                while i < len(new_expr):
                    if i + 1 < len(new_expr) and abs(new_expr[i][0]) == abs(new_expr[i + 1][0]):
                        # We use noop for grouping because otherwise, Sympy will change things
                        if new_expr[i][0] != new_expr[i + 1][0]:
                            self.val += new_expr[i][0] * noop(new_expr[i][1] - new_expr[i + 1][1])
                        else:
                            self.val += new_expr[i][0] * noop(new_expr[i][1] + new_expr[i + 1][1])
                        i += 2
                    else:
                        self.val += new_expr[i][0] * new_expr[i][1]
                        i += 1
                self.val = self.val * dxt
            else:
                print("args:", expr.args)
            if self.val is None:
                raise Exception(str(expr))
            return True
        else:
            self.val = None
            return False

    def r(self, _expr: Expr) -> Optional[Expr]:
        return self.val

    def apply(self, arg: Basic) -> Basic:
        return cast(Basic, arg.replace(self.m, self.r))  # type: ignore[no-untyped-call]
