from EmitCactus.dsl.sympywrap import mkFunction
from EmitCactus.dsl.dimension import get_dimension

stencil = mkFunction("stencil")
DD = mkFunction("DD")
DDI = mkFunction("DDI")
noop = mkFunction("noop")
div = mkFunction("div")
D = mkFunction("D")
muladd = mkFunction("muladd")

# First derivatives
for i in range(get_dimension()):
    div_nm = "div" + "xyz"[i]
    func = mkFunction(div_nm)
    func.__module__ = "functions"
    globals()[div_nm] = func

# Second derivatives
for i in range(get_dimension()):
    for j in range(i, get_dimension()):
        div_nm = "div" + "xyz"[i] + "xyz"[j]
        func = mkFunction(div_nm)
        func.__module__ = "functions"
        globals()[div_nm] = func

for func in [stencil, DD, DDI, noop, div, D, muladd]:
    if func.__module__ is None:
        func.__module__ = "functions"
