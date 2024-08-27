if __name__ == "__main__":

    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import do_inv
    from EmitCactus.generators.wizards import CppCarpetXWizard

    # Create a set of grid functions
    gf = ThornDef("TestRicci", "Ricci")
    gf.set_div_stencil(3)

    # Declare gfs
    g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
    x,y,z = gf.mk_coords()
    G = gf.decl("Affine", [ua, lb, lc])
    Ric = gf.decl("Ric", [la, lb])

    gf.add_sym(g[li, lj], li, lj)
    gf.add_sym(G[ua, lb, lc], lb, lc)
    gf.add_sym(Ric[la,lb], la, lb)

    gf.mk_subst(g[la, lb], mksymbol_for_tensor_xyz)

    gmat = gf.get_matrix(g[la,lb])
    imat = do_inv(gmat)
    opt1 = False
    if opt1:
        gf.mk_subst(g[ua,ub]) # g[u0,u0] -> gUU00
    else:
        gf.mk_subst(g[ua, ub], imat)
    gf.mk_subst(div(g[la, lb], lc)) # div(g[l0,l1],l2) -> gDD01_dD2
    gf.mk_subst(div(g[ua, ub], lc))
    opt2 = False
    if opt2:
        gf.mk_subst(G[la, lb, lc])
        gf.mk_subst(G[ua, lb, lc])
    else:
        gf.mk_subst(G[la, lb, lc], (div(g[la, lb], lc) + div(g[la, lc], lb) - div(g[lb, lc], la))/2)
        gf.mk_subst(G[ud, lb, lc], g[ud,ua]*G[la, lb, lc])

    gf.mk_subst(Ric[la, lb])
    #gf.mk_subst(div(G[ud, la, lb], lc))

    fun = gf.create_function("setGL", ScheduleBin.Analysis)

    if opt1:
        fun.add_eqn(g[ua, ub], imat)
    if opt2:
        fun.add_eqn(G[la, lb, lc], (div(g[la, lb], lc) + div(g[la, lc], lb) - div(g[lb, lc], la))/2)
        fun.add_eqn(G[ua, lb, lc], g[ud,ua]*G[la, lb, lc])

    #gf.mk_subst(Ric[li, lj],
    fun.add_eqn(Ric[li, lj],
                 div(G[ua, li, lj], la) - div(G[ua, la, li], lj) +
                 G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])
    # Optimizations:
    # Does it pull the 2nd G out of the sum of: G[ua, la, lb] * G[ub, li, lj]? Check
    # -1*Foo should be -Foo
    # Comment:
    # Check non-diagonal metric

    # Ensure the equations make sense
    fun.bake()

    CppCarpetXWizard(gf).generate_thorn()
