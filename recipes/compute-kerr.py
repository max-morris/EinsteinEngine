if __name__ == "__main__":
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import mkMatrix, do_inv, do_sympify
    from sympy import Expr, cos, sin
    from EmitCactus.emit.tree import Centering

    set_dimension(4)

    # Create a set of grid functions
    gf = ThornDef("TestKerr", "Kerr")

    # Declare gfs
    g = gf.decl("g", [li, lj], sym=[(li,lj,1)], centering=Centering.VVC)
    G = gf.decl("Affine", [ua, lb, lc], sym=[(lb,lc,1)], centering=Centering.VVC)
    Ric = gf.decl("Ric", [la, lb], sym=[(la,lb,1)], centering=Centering.VVC)

    spin = False
    a: Expr
    if spin:
        # This is very slow
        a = gf.add_param("a", default=0.5, desc="The black hole spin")
    else:
        a = do_sympify(0)
    m = gf.add_param("m", default=0.5, desc="The black hole mass")
    t, r, th, phi = gf.mk_coords()

    sigma = r ** 2 + a ** 2 * cos(th) ** 2
    delta = r ** 2 - 2 * m * r + a ** 2

    gtt = -(1 - 2 * m * r / sigma)
    grr = sigma / delta
    gqq = sigma
    gpp = (r ** 2 + a ** 2 + (2 * m * r ** 2 * a ** 2 / sigma) * sin(th) ** 2) * sin(th) ** 2
    gtp = -4 * m * r * a * sin(th) ** 2 / sigma

    Z = do_sympify(0)
    gmat = mkMatrix([
        [gtt, Z, Z, gtp],
        [Z, grr, Z, Z],
        [Z, Z, gqq, Z],
        [gtp, Z, Z, gpp]])

    gf.mk_subst(g[la, lb], gmat)
    imat = do_inv(gmat)
    gf.mk_subst(g[ua, ub], imat)
    gf.mk_subst(G[la, lb, lc], (D(g[la, lb], lc) + D(g[la, lc], lb) - D(g[lb, lc], la)) / 2)
    gf.mk_subst(G[ud, lb, lc], g[ud, ua] * G[la, lb, lc])
    gf.mk_subst(Ric[li, lj],
                D(G[ua, li, lj], la) - D(G[ua, la, li], lj) +
                G[ua, la, lb] * G[ub, li, lj] - G[ua, li, lb] * G[ub, la, lj])

    for i in range(4):
        for j in range(i + 1, 4):
            ixi = [l0, l1, l2, l3][i]
            ixj = [l0, l1, l2, l3][j]
            print("Checking:", Ric[ixi, ixj])
            assert gf.do_subs(Ric[ixi, ixj]) == do_sympify(0)
