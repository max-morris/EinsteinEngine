if __name__ == "__main__":
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import mkMatrix, do_sqrt
    from EmitCactus.generators.wizards import CppCarpetXWizard

    # Create a set of grid functions
    gf = ThornDef("TestRic", "TestRicVal")

    a = gf.add_param("a", default=10.0, desc="Just a constant")
    b = gf.add_param("b", default=0.2, desc="Just a constant")
    c = gf.add_param("c", default=0.1, desc="Just a constant")

    # Declare gfs
    g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
    Ric = gf.decl("Ric", [la, lb], from_thorn="Ricci")
    RicVal = gf.decl("RicVal", [la, lb])

    gf.add_sym(g[li, lj], li, lj)
    gf.add_sym(Ric[li, lj], li, lj)
    gf.add_sym(RicVal[li, lj], li, lj)

    gf.mk_subst(g[la, lb], mksymbol_for_tensor_xyz)
    gf.mk_subst(RicVal[li,lj])
    gf.mk_subst(Ric[li,lj])

    x, y, z = gf.mk_coords()

    fun = gf.create_function("MetricSet", ScheduleBin.Analysis, schedule_before=["setGL"])
    grr = do_sqrt(1+c**2)*(a+b*x**2)
    gqq = do_sqrt(1+c**2)/(a+b*x**2)
    gpp = 1
    gmat = mkMatrix([
    [grr,   c,   0],
    [  c, gqq,   0],
    [  0,   0, gpp]])
    assert gmat.det() == 1
    fun.add_eqn(g[li,lj],gmat)
    fun.bake()

    fun = gf.create_function("RicZero", ScheduleBin.Analysis, schedule_after=["setGL"])
    fun.add_eqn(RicVal[l0,l0], Ric[l0,l0]-b*(a*c**2 + a - 3*b*c**2*x**2 - 3*b*x**2)/(a**2 + 2*a*b*x**2 + b**2*x**4))
    fun.bake()


    CppCarpetXWizard(gf).generate_thorn()
