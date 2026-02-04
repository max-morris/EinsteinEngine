from sympy import *

class SymIter:
    def __init__(self, x):
        self.x = x
        self.n = 0
    def __iter__(self):
        self.n += 1
        return self
    def __next__(self):
        return Symbol(f"{self.x}{self.n}")


def batch_cse(e, batch=3, symbols=None):
    n = 0
    name_iter = SymIter(f"tmp_{n}_")
    elist = []
    out = []
    out2 = []
    for i in range(len(e)):
        elist.append(e[i])
        a, b = cse(elist, symbols=name_iter)
        if len(a) > batch:
            out += a
            out2 += b
            elist = []
            n += 1
            name_iter = SymIter(f"tmp_{n}_")
    if len(elist) > 0:
        a, b = cse(elist, symbols=name_iter)
        out += a
        out2 += b
    return (out, out2)

    return cse(e, symbols=s)

if __name__ == "__main__":
    u = Symbol("u")
    v = Symbol("v")
    e = [
        (u**n + n*v**n)*(u**n + n*v**n + n**3)
        for n in range(1,20)]
    a, b = batch_cse(e)
    for item in a:
        print(">>",item)
    for item in b:
        print(item)
    print("len:",len(a),len(b))
    print()
    a, b = cse(e)
    for item in a:
        print(">>",item)
    for item in b:
        print(item)
    print("len:",len(a),len(b))
