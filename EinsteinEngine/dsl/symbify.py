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
