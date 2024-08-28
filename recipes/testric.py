if __name__ == "__main__":
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import mkMatrix, do_inv

    # Create a set of grid functions
    gf = ThornDef("TestRic", "TestRic")

    a = gf.add_param("a", default=10.0, desc="Just a constant")
    b = gf.add_param("b", default=0.2, desc="Just a constant")
    c = gf.add_param("c", default=-0.8, desc="Just a constant")

    # Declare gfs
    g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
    G = gf.decl("Affine", [ua, lb, lc])
    Ric = gf.decl("Ric", [la, lb], from_thorn="Ricci")
    RicVal = gf.decl("RicVal", [la, lb])

    gf.add_sym(g[li, lj], li, lj)
    gf.add_sym(Ric[li, lj], li, lj)
    gf.add_sym(RicVal[li, lj], li, lj)
    gf.add_sym(G[ua, lb, lc], lb, lc)

    x, y, z= gf.mk_coords()

    # Figure out what the answer ought to be
    grr = a+b*x**2
    gqq = a+b*x**2
    gpp = 1
    gmat = mkMatrix([
    [grr,   c,   0],
    [  c, gqq,   0],
    [  0,   0, gpp]])
    gf.mk_subst(g[la,lb], gmat)
    imat = do_inv(gmat)
    gf.mk_subst(g[ua, ub], imat)

    gf.mk_subst(G[la, lb, lc], (div(g[la, lb], lc) + div(g[la, lc], lb) - div(g[lb, lc], la))/2)
    gf.mk_subst(G[ud, lb, lc], g[ud,ua]*G[la, lb, lc])
    gf.mk_subst(Ric[li, lj],
                 div(G[ua, li, lj], la) - div(G[ua, la, li], lj) +
                 G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])
