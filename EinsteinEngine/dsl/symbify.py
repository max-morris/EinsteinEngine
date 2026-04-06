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

from typing import Set, Any
from sympy import NumberSymbol, Number, Symbol, Function, Indexed, IndexedBase, Idx, Basic, Expr, Mul, Add, Pow
from EinsteinEngine.dsl.sympywrap import *
from multimethod import multimethod

@multimethod
def symbify(a:Number)->Expr:
    return a

@symbify.register
def _(a:NumberSymbol)->Expr:
    return a

@symbify.register
def _(a:IndexedBase)->Expr:
    r = a.args[0]
    assert isinstance(r, Symbol)
    return r

@symbify.register
def _(a:Symbol)->Expr:
    return a

@symbify.register
def _(a:Function)->Expr:
    arglist = []
    for b in a.args:
        arglist.append(symbify(b))
    r = a.__class__(*arglist)
    assert isinstance(r, Expr)
    return r

@symbify.register
def _(a:Mul)->Expr:
    r = sympify(1)
    for b in a.args:
        r *= symbify(b)
    return r

@symbify.register
def _(a:Add)->Expr:
    r = sympify(0)
    for b in a.args:
        r += symbify(b)
    return r

@symbify.register
def _(a:Pow)->Expr:
    r = Pow(symbify(a.args[0]), a.args[1])
    assert isinstance(r, Expr)
    return r
