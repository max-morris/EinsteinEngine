from EmitCactus import mkSymbol
from EmitCactus.dsl.eqnlist import EqnList
from EmitCactus.dsl.madd import Maddifier

if __name__ == "__main__":
    a = mkSymbol("a")
    r1 = mkSymbol("r1")
    r2 = mkSymbol("r2")
    r3 = mkSymbol("r3")
    b = mkSymbol("b")
    c = mkSymbol("c")

    eqn_list = EqnList({})
    eqn_list.add_eqn(r1, a + b * c)
    eqn_list.add_eqn(r2, a * b + c)
    eqn_list.add_eqn(r3, (a + b) * c + a ** 2)

    result = Maddifier(eqn_list).maddify()
    for k, v in result.items():
        print(k, "==", eqn_list.eqns[k], "->", v)
