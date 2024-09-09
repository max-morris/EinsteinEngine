# see https://docs.einsteintoolkit.org/et-docs/images/0/05/PeterDiener15-MacLachlan.pdf
if __name__ == "__main__":

    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import do_inv, do_det, do_replace
    from EmitCactus.generators.wizards import CppCarpetXWizard
    from sympy import exp
    import sys

    # Create a set of grid functions
    gf = ThornDef("PY_BSSN", "py_bssn")
    gf.set_div_stencil(5) # 4th order

    # From ADMBaseX
    g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
    gf.add_sym(g[li, lj], li, lj)

    k = gf.decl("k", [li, lj], from_thorn="ADMBaseX")
    gf.add_sym(k[li, lj], li, lj)

    alp = gf.decl("alp", [], from_thorn="ADMBaseX")
    beta = gf.decl("beta", [ua], from_thorn="ADMBaseX")
    ###

    x,y,z = gf.mk_coords()

    # BSSN Vars
    gt = gf.decl("gt", [li,lj]) # \tilde{g}
    gf.add_sym(gt[li, lj], li, lj)
    gt_dt = gf.decl("gt_dt", [li,lj]) # \tilde{g}
    gf.add_sym(gt_dt[li, lj], li, lj)
    Affinet = gf.decl("Affinet", [ua, lb, lc])
    gf.add_sym(Affinet[la,lb,lc], lb, lc)

    phi = gf.decl("phi", [])
    phi_dt = gf.decl("phi_dt", [])

    At = gf.decl("At", [li,lj]) # \tilde{A}
    gf.add_sym(At[li, lj], li, lj)
    At_dt = gf.decl("At_dt", [li,lj])
    gf.add_sym(At_dt[li, lj], li, lj)

    trK = gf.decl("trK", []) # trace of Extrinsic Curvature 
    trK_dt = gf.decl("trK_dt", []) 

    Gt = gf.decl("Gt", [ui]) # \tilde{\Gamma}^i
    Gt_dt = gf.decl("Gt_dt", [ui])

    ddA = gf.decl("ddA", [li,lj]) # D_i D_j alp
    gf.add_sym(ddA[li,lj], li, lj)

    ddphi = gf.decl("ddphi", [li,lj]) # D_i D_j phi
    gf.add_sym(ddphi[li,lj], li, lj)

    ric = gf.decl("ric", [li,lj])
    gf.add_sym(ric[li,lj], li, lj)
    ###

    def sym(expr, ind1, ind2):
        return (expr + expr.subs({ind1:u1, ind2:u2}).subs({u1:ind2, u2:ind1}))/2

    gf.mk_subst(gt_dt[li,lj])
    gf.mk_subst(gt[li,lj])
    gf.mk_subst(At[li,lj])
    gf.mk_subst(beta[ui])
    gf.mk_subst(Gt[ui])
    gmat = gf.get_matrix(gt[li,lj])
    imat = do_inv(gmat)*do_det(gmat) # Use the fact that det(gmat) = 1
    gf.mk_subst(gt[ui,uj], imat)
    gf.mk_subst(Gt_dt[ui])
    gf.mk_subst(At_dt[li,lj])
    gf.mk_subst(At[ui,uj])
    gf.mk_subst(At[ui,lj])
    gf.mk_subst(Affinet[ua,lb,lc])
    gf.mk_subst(Affinet[la,lb,lc])
    gf.mk_subst(ddA[li,lj])
    gf.mk_subst(ddphi[li,lj])
    gf.mk_subst(ric[li,lj])

    fun = gf.create_function("evo", ScheduleBin.Evolve)
    fun.add_eqn(gt_dt[li,lj], -2*alp*At[li,lj] + beta[uk]*div(gt[li,lj],lk) + gt[li,lk]*div(beta[uk],lj) - (2/3)*gt[li,lj]*div(beta[uk],lk))
    fun.add_eqn(phi_dt, -(1/6)*alp*trK + div(phi,lk)*beta[uk] + (1/6)*div(beta[uk],lk))
    fun.add_eqn(At[ui,lj], At[la,lj]*gt[ua,ui])
    fun.add_eqn(At[ui,uj], At[ui,lb]*gt[ub,uj])
    fun.add_eqn(Affinet[la, lb, lc], (div(gt[la, lb], lc) + div(gt[la, lc], lb) - div(gt[lb, lc], la))/2)
    fun.add_eqn(Affinet[ud, lb, lc], gt[ud,ua]*Affinet[la, lb, lc])
    fun.add_eqn(Gt_dt[ui], gt[uj,uk]*div(beta[ui],lj,lk) + (1/3)*gt[ui,uj]*div(beta[uk],lj,lk) +
        beta[uj]*div(Gt[ui],lj) - Gt[uj]*div(beta[ui],lj) + (2/3)*Gt[ui]*div(beta[uj],lj) -
        2*At[ui,uj]*div(alp,lj) + 2*alp*(
            Affinet[ui,lj,lk]*At[uj,uk] + 6*At[ui,uj]*div(phi,lj) - (2/3)*gt[ui,uj]*div(trK,lj)))
    # See: https://en.wikipedia.org/wiki/Covariant_derivative
    # \lambda_{a;c} = \partial_c \lambda_a - \Gamma^{b}_{ca} \lambda_b
    fun.add_eqn(ddA[li,lj], div(alp,li,lj) - Affinet[uk,li,lj]*div(alp,lk))
    fun.add_eqn(ddphi[li,lj], div(phi,li,lj) - Affinet[uk,li,lj]*div(phi,lk))
    fun.add_eqn(trK_dt, ddA[li,lj]*gt[ui,uj] +
        alp*(At[ui,uj]*At[li,lj]+(1/3)*trK**2) + beta[uk]*div(trK,lk))
    fun.add_eqn(ric[li,lj], -2*ddphi[li,lj]-2*gt[li,lj]*ddphi[la,lb]*gt[ua,ub] \
        - (1/2)*gt[ua,ub]*div(gt[li,lj],la,lb) \
        + sym(gt[lk,li]*div(Gt[uk],lk,lj),li,lj) \
        + sym(Gt[uk]*Affinet[li,lj,lk],li,lj) \
        + sym(gt[ua,ub]*2*Affinet[uk,la,li]*Affinet[lj,lk,lb],li,lj) \
        + sym(gt[ua,ub]*Affinet[uk,li,lb]*Affinet[lk,la,lj],li,lj))
    fun.add_eqn(At_dt[li,lj], exp(-4*phi)*(ddA[li,lj] + alp*ric[li,lj])  + 
        alp*(trK*At[li,lj] - 2*At[li,lk]*At[uk,lj]) +
        beta[uk]*div(At[li,lj],lk) + At[li,lk]*div(beta[uk],lj) + At[lj,lk]*div(beta[uk],li)  -
        (2/3)*At[li,lj]*div(beta[uk],lk))

    gf.check_globals()

    fun.bake()

    CppCarpetXWizard(gf).generate_thorn()

