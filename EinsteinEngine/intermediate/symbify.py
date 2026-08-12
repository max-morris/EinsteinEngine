#  Copyright (C) 2026 Steven R. Brandt, Max Morris, and other Einstein Engine contributors.
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

from typing import cast, Iterable, Never, Tuple

import sympy
from multimethod import multimethod
from sympy import NumberSymbol, Number, Symbol, Function, IndexedBase, Expr, Mul, Add, Pow, Idx, Indexed

from EinsteinEngine.common.sympywrap import *
from EinsteinEngine.intermediate.intermediate_exception import IntermediateException


@multimethod
def symbify(a: Number) -> Expr:
    return a


@symbify.register
def _(a: Idx | Indexed) -> Never:
    raise IntermediateException(f"Expression {a} (type {type(a)}) should not be present while calling symbify()")


@symbify.register
def _(a: NumberSymbol) -> Expr:
    return a


@symbify.register
def _(a: IndexedBase) -> Symbol:
    if not isinstance(r := a.args[0], Symbol):
        raise IntermediateException(f"Expected a Symbol, got {r} (type {type(r)})")
    if len(a.args) == 2:
        if not isinstance(a.args[1], tuple) and not isinstance(a.args[1], sympy.core.containers.Tuple):
            raise IntermediateException(f"Expected a tuple as the second argument of IndexedBase, got {a.args[1]} (type {type(a.args[1])})")
    if len(a.args) > 2:
        raise IntermediateException(f"Expected at most 2 arguments for IndexedBase, got {len(a.args)}")
    return r


@symbify.register
def _(a: Symbol) -> Symbol:
    return a


@symbify.register
def _(a: Function) -> Expr:
    arglist = []
    for b in a.args:
        arglist.append(symbify(b))
    r = a.__class__(*arglist)
    assert isinstance(r, Expr)
    return r


@symbify.register
def _(a: Mul) -> Expr:
    r = sympify(1)
    for b in a.args:
        r *= symbify(b)
    return r


@symbify.register
def _(a: Add) -> Expr:
    r = sympify(0)
    for b in a.args:
        r += symbify(b)
    return r


@symbify.register
def _(a: Pow) -> Expr:
    r = Pow(symbify(a.args[0]), a.args[1])
    assert isinstance(r, Expr)
    return r


@symbify.register
def _(_a: sympy.logic.boolalg.BooleanTrue) -> Expr:
    return sympify(True)


@symbify.register
def _(_a: sympy.logic.boolalg.BooleanFalse) -> Expr:
    return sympify(False)


@symbify.register
def _(a: sympy.core.relational.Relational) -> Expr:
    arglist = [symbify(b) for b in a.args]
    return cast(Expr, a.func(*arglist))


@symbify.register
def _(a: sympy.Piecewise) -> Expr:
    pw_args = cast(Iterable[Tuple[Expr, Expr]], a.args)
    new_args = tuple((symbify(e), symbify(c)) for e, c in pw_args)
    return mk_piecewise(*new_args)
