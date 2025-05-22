from typing import Union
from sympy import Expr, Add, Mul, Symbol, Idx, Rational, Integer
from EmitCactus.dsl.sympywrap import do_sympify as sympify, do_simplify as simplify, mkSymbol, mkIdx

from multimethod import multimethod
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
