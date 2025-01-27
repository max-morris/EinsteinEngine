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

    x,y,z = gf.mk_coords() # TODO: is this needed?

    ###
    # Evolved BSSN Vars
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
    
    ###
    # Evolved Gauge Vars
    ###
    
    alpha = gf.decl("alpha", []) # Lapse
    alpha_dt = gf.decl("alpha_dt", [])
    
    shift = gf.decl("shift", [ui]) # Shift vector
    shift_dt = gf.decl("shift_dt", [ui])
    
    B = gf.decl("B", [ui]) # Aux. vector B^i
    B_dt = gf.decl("B_dt", [ui])

    ###
    # Aux. Vars
    ###
    
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
    # Substitution rules
    ###

    g_mat = gf.get_matrix(g[li,lj])
    g_imat = do_inv(g_mat) 
    detg = do_det(g_mat)
    gf.mk_subst(g[ui,uj], g_imat)

    gf.mk_subst(gt_dt[li,lj])
    gf.mk_subst(gt[li,lj])

    gt_mat = gf.get_matrix(gt[li,lj])
    detgt = do_det(gt_mat)
    gt_imat = do_inv(gt_mat) * detgt # Use the fact that det(gt) = 1
    gf.mk_subst(gt[ui,uj], gt_imat)
    
    gf.mk_subst(At[li,lj])
    gf.mk_subst(At_dt[li,lj])
    gf.mk_subst(At[ui,uj])
    gf.mk_subst(At[ui,lj])
    
    gf.mk_subst(Gt[ui])
    gf.mk_subst(Gt_dt[ui])

    gf.mk_subst(shift[ui])
    gf.mk_subst(shift_dt[ui])

    gf.mk_subst(B[ui])
    gf.mk_subst(B_dt[ui])
    
    gf.mk_subst(Gammat[ua,lb,lc])
    gf.mk_subst(Gammat[la,lb,lc])

    gf.mk_subst(Gamma[ua,lb,lc])

    gf.mk_subst(ric[li,lj])
    
    gf.mk_subst(ddA[li,lj])
    gf.mk_subst(ddtphi[li,lj])

    gf.mk_subst(T[li,lj])

    ###
    # BSSN Evolution equations
    # Following [1], we will replace \tilde{\Gamma}^i with
    # \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk} whenever the
    # \tilde{\Gamma}^i are needed without derivatives.
    #
    # TODO: Following [5] FD stencils are centered except for terms
    # of the form (\shift^i \partial_i u) which are calculated
    # using an “upwind” stencil which is shifted by one point in
    # the direction of the shift, and of the same order
    ###
    fun = gf.create_function("bssn_rhs", ScheduleBin.Evolve)
    
    # Auxiliary Equations

    fun.add_eqn(T[li,lj], -ddA[li,lj] + alpha * ric[li,lj])
    
    # See: [3]
    # \lambda_{a;c} = \partial_c \lambda_a - \Gamma^{b}_{ca} \lambda_b
    fun.add_eqn(ddA[li,lj], div(alpha, lj,li) - Gamma[uk,li,lj] * div(alpha,lk))
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
        -2 * alpha * At[li,lj] \
        + shift[uk] * div(gt[li,lj], lk) \
        + gt[li,lk] * div(shift[uk], lj) \
        + gt[lj,lk] * div(shift[uk], li) \
        - (2/3) * gt[li,lj] * div(shift[uk], lk)
    )
    
    fun.add_eqn(
        phi_dt,
        -(1/6) * alpha * trK \
        + div(phi, lk) * shift[uk] \
        + (1/6) * div(shift[uk], lk)
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
        + alpha * (trK * At[li,lj] - 2 * At[li,lk] * At[uk,lj]) \
        + shift[uk] * div(At[li,lj], lk) \
        + At[li,lk] * div(shift[uk], lj) \
        + At[lj,lk] * div(shift[uk], li) \
        - (2/3) * At[li,lj] * div(shift[uk], lk)
        )
    
    fun.add_eqn(
        trK_dt,
        ddA[li,lj] * gt[ui,uj] \
        + alpha * (At[ui,uj] * At[li,lj] + (1/3) * trK**2) \
        + shift[uk] * div(trK, lk)
    )
    
    fun.add_eqn(
        Gt_dt[ui],
        gt[uj,uk] * div(shift[ui], lj,lk) \
        + (1/3) * gt[ui,uj] * div(shift[uk],lj,lk) \
        + shift[uj] * div(Gt[ui], lj) \
        - gt[ua,ub] * Gammat[uj,la,lb] * div(shift[ui], lj) \
        + (2/3) * gt[ua,ub] * Gammat[ui,la,lb] * div(shift[uj], lj) \
        - 2 * At[ui,uj] * div(alpha, lj) \
        + 2 * alpha * (
            Gammat[ui,lj,lk] * At[uj,uk] \
            + 6 * At[ui,uj] * div(phi, lj) \
            - (2/3) * gt[ui,uj] * div(trK, lj)
        )
    )

    # 1 + log lapse
    fun.add_eqn(alpha_dt, shift[ui] * div(alpha, li) - 2 * alpha * trK)
    
    # Gamma Driver shift
    fun.add_eqn(
        shift_dt[ua],
        shift[ui] * div(shift[ua], li) + 3/4 * alpha * B[ua]
    )

    eta = 1 # TODO: Make eta a parameter
    fun.add_eqn(
        B_dt[ua],
        shift[uj] * div(B[ua], lj) + Gt_dt[ua] \
        - shift[ui] * div(Gt[ua], li) - eta * B[ua]
    )

    fun.bake()
    
    ###
    # Convert ADM to BSSN variables
    ###
    funload = gf.create_function("adm2bssn", ScheduleBin.Evolve) # TODO: Schedule in the propper place

    phi_tmp = mkSymbol("phi_tmp")
    trK_tmp = mkSymbol("trK_tmp")
    
    funload.add_eqn(phi_tmp, (1/12) * log(detg))
    funload.add_eqn(trK_tmp, g[ui,uj] * k[li,lj])

    funload.add_eqn(gt[li,lj], exp(-4 * phi_tmp) * g[li,lj])
    funload.add_eqn(phi, phi_tmp)
    funload.add_eqn(At[li,lj], exp(-4 * phi_tmp) * (k[li,lj] - (1/3) * g[li,lj] * trK_tmp))
    funload.add_eqn(trK, trK_tmp)
    funload.add_eqn(Gt[ui], -div(exp(-4 * phi_tmp) * g[ui,uj], lj)) #TODO: Can gt[li,lj] be used here?

    funload.add_eqn(alpha, alp)
    funload.add_eqn(shift[ua], beta[ua])

    funload.bake()

    ###
    # Convert BSSN to ADM variables
    ###
    funstore = gf.create_function("bssn2adm", ScheduleBin.Evolve) # TODO: Schedule in the propper place
    funstore.add_eqn(g[li,lj], exp(4 * phi) * gt[li,lj])
    funstore.add_eqn(k[li,lj], exp(4 * phi) * At[li,lj] + (1/3) * gt[li,lj] * trK)
    
    funstore.add_eqn(alp, alpha)
    funstore.add_eqn(beta[ua], shift[ua])
    
    funstore.bake()

    ###
    # Thorn creation
    ###

    CppCarpetXWizard(gf).generate_thorn()

# References
# [1] https://docs.einsteintoolkit.org/et-docs/images/0/05/PeterDiener15-MacLachlan.pdf
# [2] https://arxiv.org/abs/gr-qc/9810065
# https://en.wikipedia.org/wiki/Covariant_derivative
# [4] https://arxiv.org/pdf/2109.11743.
# [5] https://arxiv.org/pdf/0910.3803
