if __name__ == "__main__":

    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import do_inv, do_det, do_simplify, do_sqrt, mkSymbol, mkMatrix
    from EmitCactus.generators.wizards import CppCarpetXWizard

    # Create a set of grid functions
    gf = ThornDef("TestRicci", "Ricci")
    #gf.set_div_stencil(3)

    # Declare gfs
    g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
    x,y,z = gf.mk_coords()
    G = gf.decl("Affine", [ua, lb, lc])
    Ric = gf.decl("Ric", [la, lb])
    idetg = gf.decl("idetg", [])

    gf.add_sym(g[li, lj], li, lj)
    gf.add_sym(G[ua, lb, lc], lb, lc)
    gf.add_sym(Ric[la,lb], la, lb)

    gf.mk_subst(g[la, lb], mksymbol_for_tensor_xyz)

    gmat = gf.get_matrix(g[la,lb])
    imat = do_simplify(do_inv(gmat)*do_det(gmat)) #*idetg
    gf.mk_subst(g[ua, ub], imat)
    #gf.mk_subst(div(g[la, lb], lc)) # div(g[l0,l1],l2) -> gDD01_dD2
    gf.mk_subst(div(g[la, lb], lc))
    gf.mk_subst(div(g[la, lb], lc, ld))
    #gf.mk_subst(div(g[ua, ub], lc))
    #gf.mk_subst(div(g[ua, ub], lc, ld))

    # Values for matrices
    a = gf.add_param("a", default=10.0, desc="Just a constant")
    b = gf.add_param("b", default=0.2, desc="Just a constant")
    c = gf.add_param("c", default=0.1, desc="Just a constant")
    grr = do_sqrt(1+c**2)*(a+b*x**2)
    gqq = do_sqrt(1+c**2)/(a+b*x**2)
    gpp = 1
    vmat = mkMatrix([
    [grr,   c,   0],
    [  c, gqq,   0],
    [  0,   0, gpp]])
    assert vmat.det() == 1

    # Define the affine connections
    gf.mk_subst(G[la, lb, lc], (div(g[la, lb], lc) + div(g[la, lc], lb) - div(g[lb, lc], la))/2)
    gf.mk_subst(G[ud, lb, lc], g[ud,ua]*G[la, lb, lc])

    gf.mk_subst(Ric[la, lb])
    gf.mk_subst(div(g[la, lb], lc), vmat.applyfunc(lambda x : div(x, lc)))

    fun = gf.create_function("setGL", ScheduleBin.Analysis)

    fun.add_eqn(div(g[la, lb], lc, ld), vmat.applyfunc(lambda x : div(x, lc, ld)))
    fun.add_eqn(Ric[li, lj],
                 div(G[ua, li, lj], la) - div(G[ua, la, li], lj) +
                 G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])
    # Optimizations:
    # Does it pull the 2nd G out of the sum of: G[ua, la, lb] * G[ub, li, lj]? Check
    # -1*Foo should be -Foo
    # Comment:
    # Check non-diagonal metric

    # Ensure the equations make sense
    fun.bake(do_recycle_temporaries=False)

    CppCarpetXWizard(gf).generate_thorn()
