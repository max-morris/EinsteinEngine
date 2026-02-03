from typing import Set, Any
from sympy import Symbol, Function, Indexed, IndexedBase, Idx, Basic, Expr
from EmitCactus.dsl.sympywrap import *
from multimethod import multimethod

@multimethod
def diag_free_symbol(arg: Symbol, rhs: Set[Symbol|Function])->None:
    rhs.add(arg)

@diag_free_symbol.register
def _(arg: IndexedBase, rhs: Set[Symbol|Function])->None:
    diag_free_symbol(arg.args[0], rhs)

@diag_free_symbol.register
def _(arg: Indexed, rhs: Set[Symbol|Function])->None:
    raise DslException(f"Not a symbol: {arg}")

@diag_free_symbol.register
def _(arg: Idx, rhs: Set[Symbol|Function])->None:
    pass #raise DslException(f"Not a symbol: {arg}")

@diag_free_symbol.register
def _(arg: Function, rhs: Set[Symbol|Function])->None:
    if arg.__class__.__name__ == "stencil":
        rhs.add(arg)
    else:
        for a in arg.args:
            diag_free_symbol(a, rhs)

@diag_free_symbol.register
def _(arg: Basic, rhs: Set[Symbol|Function])->None:
    for arg in arg.args:
        diag_free_symbol(arg, rhs)

@diag_free_symbol.register
def _(arg: Any, rhs: Set[Symbol|Function])->None:
    print("::>", arg.__class__.__name__)
    for arg in arg.args:
        diag_free_symbol(arg, rhs)

def diag_free_symbols(expr: Expr) -> Set[Symbol]:
    rhs: Set[Symbol] = set()
    diag_free_symbol(expr, rhs)
    return rhs

if __name__ == "__main__":
    stencil = mkFunction("stencil")
    u = mkSymbol("u")
    v = mkSymbol("v")

    msin = mkFunction("msin")
    x = mkSymbol("x")
    eqn = stencil(u, 1, 0, 0)*v*sin(x)
    for f in diag_free_symbols(eqn):
        print("-->",f)
