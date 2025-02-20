if __name__ == "__main__":
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.use_indices import parities
    from EmitCactus.dsl.sympywrap import do_inv, do_det, mkSymbol, do_subs
    from EmitCactus.generators.wizards import CppCarpetXWizard
    from sympy import exp, log, Idx, Expr

    ###
    # Thorn definitions
    ###
    gf = ThornDef("PyBSSN", "BSSN")
    gf.set_div_stencil(5)  # 4th order. TODO: Use upwind stencils for the shift

    ###
    # Thorn parameters
    ###
    g_driver_eta = gf.add_param(
        "g_driver_eta",
        default=1.0,
        desc="The eta parameter of the Hyperbolic Gamma Driver shift"
    )

    ###
    # Tensor parities
    ###
    # fmt: off
    parity_scalar = parities(+1,+1,+1)
    parity_vector = parities(1,+1,+1,  +1,-1,+1,  +1,+1,-1)
    parity_sym2ten = parities(+1,+1,+1,  -1,-1,+1,  -1,+1,-1,  +1,+1,+1,  +1,-1,-1,  +1,+1,+1)
    # fmt: on

    ###
    # ADMBaseX vars.
    ###
    g = gf.decl("g", [li, lj], from_thorn="ADMBaseX")
    gf.add_sym(g[li, lj], li, lj)
    gf.mk_subst(g[li, lj], mksymbol_for_tensor_xyz)

    k = gf.decl("k", [li, lj], from_thorn="ADMBaseX")
    gf.add_sym(k[li, lj], li, lj)
    gf.mk_subst(k[li, lj], mksymbol_for_tensor_xyz)

    alp = gf.decl("alp", [], from_thorn="ADMBaseX")

    beta = gf.decl("beta", [ua], from_thorn="ADMBaseX")
    gf.mk_subst(beta[ua], mksymbol_for_tensor_xyz)

    dtbeta = gf.decl("dtbeta", [ua], from_thorn="ADMBaseX")
    gf.mk_subst(dtbeta[ua], mksymbol_for_tensor_xyz)

    ###
    # Evolved Gauge Vars.
    ###
    evo_lapse_rhs = gf.decl("evo_lapse_rhs", [], parity=parity_scalar)
    evo_lapse = gf.decl("evo_lapse", [], rhs=evo_lapse_rhs,
                        parity=parity_scalar)

    evo_shift_rhs = gf.decl("evo_shift_rhs", [ui], parity=parity_vector)
    evo_shift = gf.decl("evo_shift", [ui],
                        rhs=evo_shift_rhs, parity=parity_vector)

    g_driver_B_rhs = gf.decl("g_driver_B_rhs", [ui], parity=parity_vector)
    g_driver_B = gf.decl(
        "g_driver_B", [ui], rhs=g_driver_B_rhs, parity=parity_vector)

    ###
    # Evolved BSSN Vars.
    ###
    # \phi
    phi_rhs = gf.decl("phi_rhs", [], parity=parity_scalar)
    phi = gf.decl("phi", [], rhs=phi_rhs, parity=parity_scalar)

    # \tilde{\gamma_{ij}}
    gt_rhs = gf.decl("gt_rhs", [li, lj], parity=parity_sym2ten)
    gf.add_sym(gt_rhs[li, lj], li, lj)
    gt = gf.decl("gt", [li, lj], rhs=gt_rhs, parity=parity_sym2ten)
    gf.add_sym(gt[li, lj], li, lj)

    # \tilde{A}_{ij}
    At_rhs = gf.decl("At_rhs", [li, lj], parity=parity_sym2ten)
    gf.add_sym(At_rhs[li, lj], li, lj)
    At = gf.decl("At", [li, lj], rhs=At_rhs, parity=parity_sym2ten)
    gf.add_sym(At[li, lj], li, lj)

    # trace of Extrinsic Curvature
    trK_rhs = gf.decl("trK_rhs", [], parity=parity_scalar)
    trK = gf.decl("trK", [], rhs=trK_rhs, parity=parity_scalar)

    # \tilde{\Gamma}^i
    ConfConnect_rhs = gf.decl("ConfConnect_rhs", [ui], parity=parity_vector)
    ConfConnect = gf.decl("ConfConnect", [ui],
                          rhs=ConfConnect_rhs, parity=parity_vector)

    ###
    # Constraint Vars.
    ###
    HamCons = gf.decl("HamCons", [], parity=parity_scalar)

    ###
    # Aux. Vars.
    ###
    Gammat = gf.decl("Gammat", [ua, lb, lc])  # \tilde{\Gamma}^a_{bc}
    gf.add_sym(Gammat[ua, lb, lc], lb, lc)

    Gamma = gf.decl("Gamma", [ua, lb, lc])  # \Gamma^a_{bc}
    gf.add_sym(Gamma[ua, lb, lc], lb, lc)

    ric = gf.decl("ric", [li, lj])  # R_{ij} = \tilde{R}_{ij} + R^\phi_{ij}
    gf.add_sym(ric[li, lj], li, lj)

    ddA = gf.decl("ddA", [li, lj])  # D_i D_j \alpha
    gf.add_sym(ddA[li, lj], li, lj)

    ddtphi = gf.decl("ddtphi", [li, lj])  # \tilde{D}_i \tilde{D}_j \phi
    gf.add_sym(ddtphi[li, lj], li, lj)

    T = gf.decl("T", [li, lj])  # T_{ij} = -D_i D_j \alpha + \alpha R_{ij}
    gf.add_sym(T[li, lj], li, lj)

    # Prevents the elimination of ConfConnect_rhs
    ConfConnect_rhs_tmp = gf.decl("ConfConnect_rhs_tmp", [ui])

    ###
    # Substitution rules
    ###
    g_mat = gf.get_matrix(g[li, lj])
    g_imat = do_inv(g_mat)
    detg = do_det(g_mat)
    gf.mk_subst(g[ui, uj], g_imat)

    gf.mk_subst(gt_rhs[li, lj])
    gf.mk_subst(gt[li, lj])

    gt_mat = gf.get_matrix(gt[li, lj])
    detgt = do_det(gt_mat)
    gt_imat = do_inv(gt_mat) * detgt  # Use the fact that det(gt) = 1
    gf.mk_subst(gt[ui, uj], gt_imat)

    gf.mk_subst(At[li, lj])
    gf.mk_subst(At_rhs[li, lj])
    gf.mk_subst(At[ui, uj])
    gf.mk_subst(At[ui, lj])

    gf.mk_subst(ConfConnect[ui])
    gf.mk_subst(ConfConnect_rhs[ui])

    gf.mk_subst(evo_shift[ui])
    gf.mk_subst(evo_shift_rhs[ui])

    gf.mk_subst(g_driver_B[ui])
    gf.mk_subst(g_driver_B_rhs[ui])

    gf.mk_subst(Gammat[ua, lb, lc])
    gf.mk_subst(Gammat[la, lb, lc])

    gf.mk_subst(Gamma[ua, lb, lc])

    gf.mk_subst(ric[li, lj])

    gf.mk_subst(ddA[li, lj])
    gf.mk_subst(ddtphi[li, lj])

    gf.mk_subst(T[li, lj])

    gf.mk_subst(ConfConnect_rhs_tmp[ui])

    ###
    # Aux. functions
    ###
    def sym(expr: Expr, ind1: Idx, ind2: Idx) -> Expr:
        """
        Index symmetrizer
        FIXME: The return type of this function gets, for some reason, converted to Any instead of Expr.
        """
        return (expr + do_subs(do_subs(expr, {ind1: u1, ind2: u2}), {u1: ind2, u2: ind1})) / 2

    def compute_ricci(function: ThornFunction) -> None:
        """
        Adds equations to a function that compute the Ricci tensor.
        """
        function.add_eqn(
            Gammat[la, lb, lc],
            (div(gt[la, lb], lc) + div(gt[la, lc], lb) - div(gt[lb, lc], la))/2
        )

        function.add_eqn(Gammat[ua, lb, lc], gt[ua, ud] * Gammat[ld, lb, lc])

        # See: [3]
        # \lambda_{a;c} = \partial_c \lambda_a - \Gamma^{b}_{ca} \lambda_b
        function.add_eqn(ddtphi[li, lj], div(phi, lj, li) -
                         Gammat[uk, li, lj] * div(phi, lk))

        function.add_eqn(
            ric[li, lj],
            - (1/2) * gt[ua, ub] * div(gt[li, lj], la, lb)
            + sym(gt[lk, li] * div(ConfConnect[uk], lj), li, lj)
            + sym(gt[ua, ub] * Gammat[uk, la, lb] * Gammat[li, lj, lk], li, lj)
            + sym(gt[ua, ub] * 2 * Gammat[uk, la, li]
                  * Gammat[lj, lk, lb], li, lj)
            + gt[ua, ub] * Gammat[uk, li, lb] * Gammat[lk, la, lj]
            - 2 * ddtphi[li, lj]
            - 2 * gt[li, lj] * ddtphi[la, lb] * gt[ua, ub]
            + 4 * div(phi, li) * div(phi, lj)
            - 4 * gt[li, lj] * div(phi, la) * div(phi, lb) * gt[ua, ub]
        )

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
    fun = gf.create_function("bssn_rhs", ScheduleBin.ODESolvers_RHS)

    # Aux. Equations

    fun.add_eqn(At[ui, lj], At[la, lj] * gt[ua, ui])
    fun.add_eqn(At[ui, uj], At[ui, lb] * gt[ub, uj])

    fun.add_eqn(
        Gamma[ua, lb, lc],
        (1/2) * g[ua, ud] * (
            div(g[ld, lb], lc) + div(g[ld, lc], lb) - div(g[lb, lc], ld)
        )
    )

    compute_ricci(fun)

    # See: [3]
    # \lambda_{a;c} = \partial_c \lambda_a - \Gamma^{b}_{ca} \lambda_b
    fun.add_eqn(ddA[li, lj], div(evo_lapse, lj, li) -
                Gamma[uk, li, lj] * div(evo_lapse, lk))

    fun.add_eqn(T[li, lj], -ddA[li, lj] + evo_lapse * ric[li, lj])

    # Evolution equations

    fun.add_eqn(
        gt_rhs[li, lj],
        -2 * evo_lapse * At[li, lj]
        + evo_shift[uk] * div(gt[li, lj], lk)
        + gt[li, lk] * div(evo_shift[uk], lj)
        + gt[lj, lk] * div(evo_shift[uk], li)
        - (2/3) * gt[li, lj] * div(evo_shift[uk], lk)
    )

    fun.add_eqn(
        phi_rhs,
        -(1/6) * evo_lapse * trK
        + div(phi, lk) * evo_shift[uk]
        + (1/6) * div(evo_shift[uk], lk)
    )

    # See [4]
    # Let T_{ij} \equiv -D_i D_j \alpha + \alpha R_{ij}
    # The trace free part of T, T^{(0)}_{ij} is then
    # T^{(0)}_{ij} = T_{ij} - 1/3 \gamma_{ij} \gamma^{ab} T_{ab}
    fun.add_eqn(
        At_rhs[li, lj],
        exp(-4 * phi) * (
            T[li, lj]
            - (1/3) * g[li, lj] * g[ua, ub] * T[la, lb]
        )
        + evo_lapse * (trK * At[li, lj] - 2 * At[li, lk] * At[uk, lj])
        + evo_shift[uk] * div(At[li, lj], lk)
        + At[li, lk] * div(evo_shift[uk], lj)
        + At[lj, lk] * div(evo_shift[uk], li)
        - (2/3) * At[li, lj] * div(evo_shift[uk], lk)
    )

    fun.add_eqn(
        trK_rhs,
        ddA[li, lj] * gt[ui, uj]
        + evo_lapse * (At[ui, uj] * At[li, lj] + (1/3) * trK**2)
        + evo_shift[uk] * div(trK, lk)
    )

    fun.add_eqn(
        ConfConnect_rhs_tmp[ui],
        gt[uj, uk] * div(evo_shift[ui], lj, lk)
        + (1/3) * gt[ui, uj] * div(evo_shift[uk], lj, lk)
        + evo_shift[uj] * div(ConfConnect[ui], lj)
        - gt[ua, ub] * Gammat[uj, la, lb] * div(evo_shift[ui], lj)
        + (2/3) * gt[ua, ub] * Gammat[ui, la, lb] * div(evo_shift[uj], lj)
        - 2 * At[ui, uj] * div(evo_lapse, lj)
        + 2 * evo_lapse * (
            Gammat[ui, lj, lk] * At[uj, uk]
            + 6 * At[ui, uj] * div(phi, lj)
            - (2/3) * gt[ui, uj] * div(trK, lj)
        )
    )
    fun.add_eqn(ConfConnect_rhs[ui], ConfConnect_rhs_tmp[ui])

    # 1 + log lapse
    fun.add_eqn(evo_lapse_rhs,
                evo_shift[ui] * div(evo_lapse, li) - 2 * evo_lapse * trK
                )

    # Hyperbolic Gamma Driver shift
    fun.add_eqn(
        evo_shift_rhs[ua],
        evo_shift[ui] * div(evo_shift[ua], li) + 3/4 *
        evo_lapse * g_driver_B[ua]
    )

    fun.add_eqn(
        g_driver_B_rhs[ua],
        evo_shift[uj] * div(g_driver_B[ua], lj) + ConfConnect_rhs_tmp[ua]
        - evo_shift[ui] * div(ConfConnect[ua], li) -
        g_driver_eta * g_driver_B[ua]
    )

    fun.bake()

    ###
    # Convert ADM to BSSN variables
    ###
    funload = gf.create_function(
        "adm2bssn",
        ScheduleBin.ODESolvers_Initial,
        schedule_after=["ADMBaseX_PostInitial"]
    )

    phi_tmp = (1/12) * log(detg)
    trK_tmp = g[ui, uj] * k[li, lj]

    funload.add_eqn(gt[li, lj], exp(-4 * phi_tmp) * g[li, lj])
    funload.add_eqn(phi, phi_tmp)
    funload.add_eqn(At[li, lj], exp(-4 * phi_tmp) *
                    (k[li, lj] - (1/3) * g[li, lj] * trK_tmp))
    funload.add_eqn(trK, trK_tmp)
    funload.add_eqn(ConfConnect[ui], -div(exp(-4 * phi_tmp) * g[ui, uj], lj))

    funload.add_eqn(evo_lapse, alp)
    funload.add_eqn(evo_shift[ua], beta[ua])

    funload.add_eqn(
        g_driver_B[ua],
        4.0 * (dtbeta[ua] - beta[ui] * div(beta[ua], li)) / (3.0 * alp)
    )

    funload.bake()

    ###
    # Convert BSSN to ADM variables
    ###
    funstore = gf.create_function(
        "bssn2adm",
        ScheduleBin.ODESolvers_PostStep,
        schedule_before=["ADMBaseX_SetADMVars"]
    )

    funstore.add_eqn(g[li, lj], exp(4 * phi) * gt[li, lj])
    funstore.add_eqn(k[li, lj], exp(4 * phi) *
                     At[li, lj] + (1/3) * gt[li, lj] * trK)

    funstore.add_eqn(alp, evo_lapse)
    funstore.add_eqn(beta[ua], evo_shift[ua])

    funstore.bake()

    ###
    # Compute constraints
    ###
    funcons = gf.create_function(
        "bssn_cons",
        ScheduleBin.Analysis
    )

    function.add_eqn(
        Gamma[ua, lb, lc],
        (1/2) * g[ua, ud] * (
            div(g[ld, lb], lc) + div(g[ld, lc], lb) - div(g[lb, lc], ld)
        )
    )

    compute_ricci(funcons)

    funcons.add_eqn(HamCons,
                    g[ui, ua] * g[uj, ub] * ric[la, lb] * ric[li, lj]
                    + (2/3) * trK * trK
                    - gt[ui, ua] * gt[uj, ub] * At[la, lb] * At[li, lj]
                    )

    funcons.bake()

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
