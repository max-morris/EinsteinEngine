from typing import List, Dict
from EmitCactus.dsl.sympywrap import *
from sympy import Expr, Symbol, Mul, Add
from EmitCactus.dsl.eqnlist import EqnList
from multimethod import multimethod

madd = mkFunction("muladd")

def get_mul(expr: Expr) -> Expr:
    if len(expr.args)==2:
        v = expr.args[1]
    else:
        v = Mul(*expr.args[1:])
    assert isinstance(v, Expr)
    return v

def avoid(arg : Expr) -> bool:
    return arg in [-1,1,2,-2]

@multimethod
def maddify(add:Add)->Expr:
    new_args:List[Expr] = list()
    new_args.append(add.args[0])
    did_madd = False
    for i in range(1,len(add.args)):
        prev = new_args[-1]
        curr = add.args[i]
        if did_madd:
            new_args.append(curr)
            continue
        if isinstance(prev, Mul) and not avoid(arg2 := get_mul(prev)):
            arg1 = prev.args[0]
            new_args[-1] = madd(arg1, arg2, curr)
            did_madd = True
        elif isinstance(curr, Mul) and not avoid(arg2 := get_mul(curr)):
            arg1 = curr.args[0]
            new_args[-1] = madd(arg1, arg2, prev)
            did_madd = True
        else:
            new_args.append(curr)
    if len(new_args) == 1:
        return new_args[0]
    else:
        v: Expr = Add(*new_args)
        return v

@maddify.register
def _(expr:Mul)->Expr:
    new_args: List[Expr] = list()
    for arg in expr.args:
        new_args.append(maddify(arg))
    v: Expr = Mul(*new_args)
    return v

@maddify.register
def _(expr:Expr)->Expr:
    return expr

def mk_madd(eqnlist: Dict[Symbol, Expr]) -> Dict[Symbol, Expr]:
    new_eqnlist: Dict[Symbol, Expr] = dict()
    for sym, eqn in eqnlist.items():
        new_eqnlist[sym] = maddify(eqn)
    return new_eqnlist

if __name__ == "__main__":
    a = mkSymbol("a")
    r1 = mkSymbol("r1")
    r2 = mkSymbol("r2")
    r3 = mkSymbol("r3")
    b = mkSymbol("b")
    c = mkSymbol("c")

    eqnlist = EqnList({})
    eqnlist.add_eqn(r1,a+b*c)
    eqnlist.add_eqn(r2,a*b+c)
    eqnlist.add_eqn(r3,(a+b)*c+a**2)

    result = mk_madd(eqnlist.eqns)
    for k,v in result.items():
        print(k,"==",eqnlist.eqns[k],"->",v)
