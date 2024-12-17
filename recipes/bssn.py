# see https://docs.einsteintoolkit.org/et-docs/images/0/05/PeterDiener15-MacLachlan.pdf
if __name__ == "__main__":

    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import do_inv, do_det, do_replace, mkSymbol
    from EmitCactus.generators.wizards import CppCarpetXWizard
    from sympy import exp, log
    import sys

    ###
    # Index symmetrizer
    ###
    def sym(expr, ind1, ind2):
        return (expr + expr.subs({ind1:u1, ind2:u2}).subs({u1:ind2, u2:ind1}))/2

    ###
    # Create a set of grid functions
    ###
    gf = ThornDef("PY_BSSN", "py_bssn")
    gf.set_div_stencil(5) # 4th order

    ###
    # ADMBaseX vars
    ###
    g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
    gf.add_sym(g[li, lj], li, lj)
    gf.mk_subst(g[li,lj], mksymbol_for_tensor_xyz)

    k = gf.decl("k", [li, lj], from_thorn="ADMBaseX")
    gf.add_sym(k[li, lj], li, lj)
    gf.mk_subst(k[li,lj], mksymbol_for_tensor_xyz)

    alp = gf.decl("alp", [], from_thorn="ADMBaseX")
    beta = gf.decl("beta", [ua], from_thorn="ADMBaseX")
    gf.mk_subst(beta[ua], mksymbol_for_tensor_xyz)

    x,y,z = gf.mk_coords()

    ###
    # BSSN Vars
    ###
    gt = gf.decl("gt", [li,lj]) # \tilde{\gamma_{ij}}
    gf.add_sym(gt[li, lj], li, lj)
    gt_dt = gf.decl("gt_dt", [li,lj])
    gf.add_sym(gt_dt[li, lj], li, lj)

    phi = gf.decl("phi", []) # \phi
    phi_dt = gf.decl("phi_dt", [])
    
    At = gf.decl("At", [li,lj]) # \tilde{A}_{ij}
    gf.add_sym(At[li, lj], li, lj)
    At_dt = gf.decl("At_dt", [li,lj])
    gf.add_sym(At_dt[li, lj], li, lj)

    trK = gf.decl("trK", []) # trace of Extrinsic Curvature 
    trK_dt = gf.decl("trK_dt", [])

    Gt = gf.decl("Gt", [ui]) # \tilde{\Gamma}^i
    Gt_dt = gf.decl("Gt_dt", [ui])
    
    Gammat = gf.decl("Gammat", [ua, lb, lc]) # \tilde{\Gamma}^a_{bc}
    gf.add_sym(Gammat[ua,lb,lc], lb, lc)

    Gamma = gf.decl("Gamma", [ua, lb, lc]) # \Gamma^a_{bc}
    gf.add_sym(Gamma[ua,lb,lc], lb, lc)

    ric = gf.decl("ric", [li,lj]) # R_{ij} = \tilde{R}_{ij} + R^\phi_{ij}
    gf.add_sym(ric[li,lj], li, lj)

    ddA = gf.decl("ddA", [li,lj]) # D_i D_j \alpha
    gf.add_sym(ddA[li,lj], li, lj)

    ddtphi = gf.decl("ddtphi", [li,lj]) # \tilde{D}_i \tilde{D}_j \phi
    gf.add_sym(ddtphi[li,lj], li, lj)

    T = gf.decl("T", [li,lj]) # T_{ij} = -D_i D_j \alpha + \alpha R_{ij}
    gf.add_sym(T[li,lj], li, lj)
    
    ###
    # Substitution rules for the BSSN variables
    ###
    g_mat = gf.get_matrix(g[li,lj])
    detg = do_det(g_mat)

    gf.mk_subst(gt_dt[li,lj])
    gf.mk_subst(gt[li,lj])
    gt_mat = gf.get_matrix(gt[li,lj])
    gf.mk_subst(g[ui,uj], g_mat)
    detgt = do_det(gt_mat)
    gt_imat = do_inv(gt_mat) * detgt # Use the fact that det(gt) = 1
    gf.mk_subst(gt[ui,uj], gt_imat)
    
    gf.mk_subst(At[li,lj])
    gf.mk_subst(At_dt[li,lj])
    gf.mk_subst(At[ui,uj])
    gf.mk_subst(At[ui,lj])
    
    gf.mk_subst(Gt[ui])
    gf.mk_subst(Gt_dt[ui])
    
    gf.mk_subst(Gammat[ua,lb,lc])
    gf.mk_subst(Gammat[la,lb,lc])

    gf.mk_subst(Gamma[ua,lb,lc])

    gf.mk_subst(ric[li,lj])
    
    gf.mk_subst(ddA[li,lj])
    gf.mk_subst(ddtphi[li,lj])

    gf.mk_subst(T[li,lj])
    
    gf.mk_subst(beta[ui])

    ###
    # BSSN Evolution equations
    # Following [1], we will replace \tilde{\Gamma}^i with
    # \tilde{\gamma}^{ij} \tilde{\Gamma}^i_{jk} whenever the
    # \tilde{\Gamma}^i are needed without derivatives.
    ###
    fun = gf.create_function("evo", ScheduleBin.Evolve)
    
    # Auxiliary Equations

    fun.add_eqn(T[li,lj], -ddA[li,lj] + alp * ric[li,lj])
    
    # See: [3]
    # \lambda_{a;c} = \partial_c \lambda_a - \Gamma^{b}_{ca} \lambda_b
    fun.add_eqn(ddA[li,lj], div(alp, lj,li) - Gamma[uk,li,lj] * div(alp,lk))
    fun.add_eqn(ddtphi[li,lj], div(phi, lj,li) - Gammat[uk,li,lj] * div(phi,lk))

    fun.add_eqn(
        Gammat[la, lb, lc],
        (div(gt[la, lb], lc) + div(gt[la, lc], lb) - div(gt[lb, lc], la))/2
    )
    
    fun.add_eqn(Gammat[ua, lb, lc], gt[ua,ud] * Gammat[ld, lb, lc])

    fun.add_eqn(
        Gamma[ua, lb, lc],
        (1/2) * g[ua,ud] * (
            div(g[ld,lb], lc) + div(g[ld,lc], lb) - div(g[lb,lc], ld)
        )
    )

    fun.add_eqn(At[ui,lj], At[la,lj] * gt[ua,ui])
    fun.add_eqn(At[ui,uj], At[ui,lb] * gt[ub,uj])

    fun.add_eqn(
        ric[li,lj],
        - (1/2) * gt[ua,ub] * div(gt[li,lj], la,lb) \
        + sym(gt[lk,li] * div(Gt[uk], lj), li,lj) \
        + sym(gt[ua,ub] * Gammat[uk,la,lb] * Gammat[li,lj,lk], li,lj) \
        + sym(gt[ua,ub] * 2 * Gammat[uk,la,li] * Gammat[lj,lk,lb], li,lj) \
        + gt[ua,ub] * Gammat[uk,li,lb] * Gammat[lk,la,lj] \
        -2 * ddtphi[li,lj] \
        -2 * gt[li,lj] * ddtphi[la,lb] * gt[ua,ub] \
        +4 * div(phi, li) * div(phi, lj) \
        -4 * gt[li, lj] * div(phi, la) * div(phi, lb) * gt[ua, ub]
    )

    # Evolution equations

    fun.add_eqn(
        gt_dt[li,lj],
        -2 * alp * At[li,lj] \
        + beta[uk] * div(gt[li,lj], lk) \
        + gt[li,lk] * div(beta[uk], lj) \
        + gt[lj,lk] * div(beta[uk], li) \
        - (2/3) * gt[li,lj] * div(beta[uk], lk)
    )
    
    fun.add_eqn(
        phi_dt,
        -(1/6) * alp * trK \
        + div(phi, lk) * beta[uk] \
        + (1/6) * div(beta[uk], lk)
    )
    
    # See [4]
    # Let T_{ij} \equiv -D_i D_j \alpha + \alpha R_{ij}
    # The trace free part of T, T^{(0)}_{ij} is then
    # T^{(0)}_{ij} = T_{ij} - 1/3 \gamma_{ij} \gamma^{ab} T_{ab}
    fun.add_eqn(
        At_dt[li,lj],
        exp(-4 * phi) * (
            T[li,lj] \
            -(1/3) * g[li, lj] * g[ua,ub] * T[la,lb]
        ) \
        + alp * (trK * At[li,lj] - 2 * At[li,lk] * At[uk,lj]) \
        + beta[uk] * div(At[li,lj], lk) \
        + At[li,lk] * div(beta[uk], lj) \
        + At[lj,lk] * div(beta[uk], li) \
        - (2/3) * At[li,lj] * div(beta[uk], lk)
        )
    
    fun.add_eqn(
        trK_dt,
        ddA[li,lj] * gt[ui,uj] \
        + alp * (At[ui,uj] * At[li,lj] + (1/3) * trK**2) \
        + beta[uk] * div(trK, lk)
    )
    
    fun.add_eqn(
        Gt_dt[ui],
        gt[uj,uk] * div(beta[ui], lj,lk) \
        + (1/3) * gt[ui,uj] * div(beta[uk],lj,lk) \
        + beta[uj] * div(Gt[ui], lj) \
        - gt[ua,ub] * Gammat[uj,la,lb] * div(beta[ui], lj) \
        + (2/3) * gt[ua,ub] * Gammat[ui,la,lb] * div(beta[uj], lj) \
        - 2 * At[ui,uj] * div(alp, lj) \
        + 2 * alp * (
            Gammat[ui,lj,lk] * At[uj,uk] \
            + 6 * At[ui,uj] * div(phi, lj) \
            - (2/3) * gt[ui,uj] * div(trK, lj)
        )
    )

    ###
    # Load from and store to ADMBaseX
    ###
    phi_tmp = mkSymbol("phi_tmp")
    funload = gf.create_function("load", ScheduleBin.Evolve)
    funload.add_eqn(gt[li,lj], exp(-4*phi_tmp)*g[li,lj])
    funload.add_eqn(trK, g[ui,uj]*k[li,lj])
    funload.add_eqn(phi_tmp, (1/12)*log(detg))
    funload.add_eqn(phi, phi_tmp)
    funload.add_eqn(At[li,lj], exp(-4*phi_tmp)*(k[li,lj]-(1/3)*g[li,lj]*trK))
    funload.bake()

    funstore = gf.create_function("store", ScheduleBin.Evolve)
    funstore.add_eqn(g[li,lj], exp(4*phi)*gt[li,lj])
    funstore.add_eqn(k[li,lj], exp(4*phi)*(At[li,lj]+(1/3)*trK*gt[li,lj]))
    funstore.bake()

    ###
    # Thorn creation
    ###

    fun.bake()

    CppCarpetXWizard(gf).generate_thorn()

# References
# [1] https://docs.einsteintoolkit.org/et-docs/images/0/05/PeterDiener15-MacLachlan.pdf
# [2] https://arxiv.org/abs/gr-qc/9810065
# https://en.wikipedia.org/wiki/Covariant_derivative
# [4] https://arxiv.org/pdf/2109.11743.
