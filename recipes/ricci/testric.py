from EmitCactus.emit.ccl.schedule.schedule_tree import ScheduleBlock, GroupOrFunction, AtOrIn
from EmitCactus.emit.tree import Identifier, String
from EmitCactus.generators.cpp_carpetx_generator import CppCarpetXGenerator
from sympy import MatrixBase

if __name__ == "__main__":
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import mkMatrix, do_sqrt, do_simplify, do_det, do_inv, do_sympify
    from EmitCactus.generators.wizards import CppCarpetXWizard

    # Create a set of grid functions
    gf = ThornDef("TestEmitCactus", "Ricci")
    gf.set_div_stencil(5)

    a = gf.add_param("a", default=10.0, desc="Just a constant")
    b = gf.add_param("b", default=0.2, desc="Just a constant")
    c = gf.add_param("c", default=0.1, desc="Just a constant")

    # Declare gfs
    g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
    x,y,z = gf.mk_coords()

    Ric = gf.decl("Ric", [la, lb])
    ZeroVal = gf.decl("ZeroVal", [], from_thorn="ZeroTest")
    G = gf.decl("Affine", [ua, lb, lc])

    gf.add_sym(g[li, lj], li, lj)
    gf.add_sym(Ric[li, lj], li, lj)
    gf.add_sym(ZeroVal[li, lj], li, lj) # this is an error

    gf.mk_subst(g[la, lb], mksymbol_for_tensor_xyz)
    gmat = gf.get_matrix(g[la,lb])
    print(gmat)
    imat = do_simplify(do_inv(gmat)*do_det(gmat)) #*idetg
    gf.mk_subst(g[ua, ub], imat)
    gf.mk_subst(ZeroVal[li,lj])
    gf.mk_subst(Ric[li,lj])

    # Metric
    grr = do_sqrt(1+c**2)*(a+b*x**2)
    gqq = do_sqrt(1+c**2)/(a+b*x**2)
    gpp = do_sympify(1)
    Z = do_sympify(0)
    gmat = mkMatrix([
    [grr,   c,   Z],
    [  c, gqq,   Z],
    [  Z,   Z, gpp]])
    assert do_det(gmat) == 1

    # Define the affine connections
    gf.mk_subst(G[la, lb, lc], (div(g[la, lb], lc) + div(g[la, lc], lb) - div(g[lb, lc], la))/2)
    gf.mk_subst(G[ud, lb, lc], g[ud,ua]*G[la, lb, lc])

    gf.mk_subst(Ric[la, lb])
    # Check on this?
    #gf.mk_subst(div(g[la, lb], lc), gmat.applyfunc(lambda x : div(x, lc)))
    #gf.mk_subst(div(g[la, lb], lc, ld), gmat.applyfunc(lambda x : div(x, lc, ld)))

    fun = gf.create_function("setGL", ScheduleBin.Analysis)

    fun.add_eqn(Ric[li, lj],
                 div(G[ua, li, lj], la) - div(G[ua, la, li], lj) +
                 G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])

    #x, y, z = gf.mk_coords()

    #fun = gf.create_function("setGL", ScheduleBin.Analysis)
    #fun.add_eqn(Ric[la,lb], RicVal[la,lb])
    fun.bake()

    fun = gf.create_function("MetricSet", ScheduleBin.Analysis, schedule_before=["setGL"])
    fun.add_eqn(g[li,lj],gmat)
    fun.bake()

    fun = gf.create_function("RicZero", ScheduleBin.Analysis, schedule_after=["setGL"])
    fun.add_eqn(ZeroVal, Ric[l0,l0]-b*(a*c**2 + a - 3*b*c**2*x**2 - 3*b*x**2)/(a**2 + 2*a*b*x**2 + b**2*x**4))
    fun.bake()

    check_zero = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier('CheckZeroGroup'),
        at_or_in=AtOrIn.At,
        schedule_bin=Identifier('analysis'),
        description=String('Do the check'),
        after=[Identifier('RicZero')]
    )

    CppCarpetXWizard(gf, CppCarpetXGenerator(gf, extra_schedule_blocks=[check_zero])).generate_thorn()
