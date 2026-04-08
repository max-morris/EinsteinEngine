#  Copyright (C) 2025-2026 Steven R. Brandt, Max Morris, and other Einstein Engine contributors.
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

from typing import Union

from multimethod import multimethod
from sympy import Expr, Add, Mul, Symbol, Idx, Rational, Integer

from EinsteinEngine.dsl.sympywrap import sympify, simplify, mkSymbol, mkIdx

zero = sympify(0)
one  = sympify(1)

@multimethod
def coef(sym:Symbol, expr:Symbol)->Expr:
    if sym == expr:
        return one
    else:
        return zero

@coef.register
def _(sym:Idx, expr:Idx)->Expr:
    if sym == expr:
        return one
    else:
        return zero

@coef.register
def _(sym:Union[Idx,Symbol], expr:Union[Integer,Rational])->Expr:
    return zero

@coef.register
def _(sym:Union[Symbol,Idx], expr:Add)->Expr:
    ret = zero
    for a in expr.args:
        a_ret = coef(sym, a)
        ret += a_ret
    return ret

@coef.register
def _(sym:Union[Symbol,Idx], expr:Mul)->Expr:
    ret = one
    found = False
    for a in expr.args:
        if a == zero:
            return zero
        elif a == sym:
            found = True
        else:
            ret *= a
    if found:
        return ret
    else:
        return zero

if __name__ == "__main__":
    a = mkSymbol("a")
    b = mkIdx("b")
    c = mkSymbol("c")
    one = sympify(1)

    def check(expr:Expr, expected:Expr)->None:
        print("check:",expr,"->",expected)
        ret = coef(b, expr)
        print("    ->",ret)
        should_be_zero = simplify(ret - expected)
        assert zero == should_be_zero, f"{ret} - {expected} == {should_be_zero}"

    check(b, one)
    check(-b, -one)
    check(a*c, zero)
    check(b*a*c, a*c)
    check(b*c+a*b, a+c)

    d = mkIdx("d")
    check(b-d,one)
    check(b+d,one)
