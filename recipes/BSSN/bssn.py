

if __name__ == "__main__":
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.carpetx import ExplicitSyncBatch
    from EmitCactus.dsl.sympywrap import do_inv, do_det, do_subs, mkMatrix
    from EmitCactus.dsl.use_indices import parities
    from EmitCactus.emit.ccl.schedule.schedule_tree import AtOrIn, GroupOrFunction, ScheduleBlock
    from EmitCactus.emit.tree import Identifier, Language, String
    from EmitCactus.generators.wizards import CppCarpetXWizard
    from EmitCactus.generators.cpp_carpetx_generator import CppCarpetXGenerator
    from EmitCactus.generators.cactus_generator import InteriorSyncMode

    from sympy import exp, log, Idx, Expr, cbrt

    ###
    # Thorn definitions
    ###
    pybssn = ThornDef("PyBSSN", "BSSN")
    # 4th order. TODO: Use upwind stencils for the shift
    pybssn.set_div_stencil(5)

    ###
    # Thorn parameters
    ###
    zeta_alpha = pybssn.add_param(
        "zeta_alpha",
        default=1.0,
        desc="partial_t alpha = zeta_alpha * beta^i partial_i alpha ..."
    )

    kappa_alpha = pybssn.add_param(
        "kappa_alpha",
        default=2.0,
        desc="partial_t alpha = ... - kappa_alpha alpha trK"
    )

    zeta_beta = pybssn.add_param(
        "zeta_beta",
        default=1.0,
        desc="partial_t beta^i = zeta_beta * beta^j partial_j beta^i ..."
    )

    beta_Gamma = pybssn.add_param(
        "beta_Gamma",
        default=0.75,
        desc="partial_t beta^i = ... beta_Gamma * alph^beta_Alp * Gammat^i ..."
    )

    beta_Alp = pybssn.add_param(
        "beta_Alp",
        default=0.0,
        desc="partial_t beta^i = ... beta_Gamma * alph^beta_Alp * Gammat^i ..."
    )

    eta_beta = pybssn.add_param(
        "eta_beta",
        default=1.0,
        desc="partial_t beta^i = ... - eta_beta * beta^i"
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
    g = pybssn.decl("g", [li, lj], from_thorn="ADMBaseX")
    pybssn.add_sym(g[li, lj], li, lj)
    pybssn.mk_subst(g[li, lj], mksymbol_for_tensor_xyz)

    k = pybssn.decl("k", [li, lj], from_thorn="ADMBaseX")
    pybssn.add_sym(k[li, lj], li, lj)
    pybssn.mk_subst(k[li, lj], mksymbol_for_tensor_xyz)

    alp = pybssn.decl("alp", [], from_thorn="ADMBaseX")

    beta = pybssn.decl("beta", [ua], from_thorn="ADMBaseX")
    pybssn.mk_subst(beta[ua], mksymbol_for_tensor_xyz)

    dtbeta = pybssn.decl("dtbeta", [ua], from_thorn="ADMBaseX")
    pybssn.mk_subst(dtbeta[ua], mksymbol_for_tensor_xyz)

    ###
    # Evolved Gauge Vars.
    ###
    evo_lapse_rhs = pybssn.decl("evo_lapse_rhs", [], parity=parity_scalar)
    evo_lapse = pybssn.decl("evo_lapse", [], rhs=evo_lapse_rhs,
                            parity=parity_scalar)

    evo_shift_rhs = pybssn.decl("evo_shift_rhs", [ui], parity=parity_vector)
    evo_shift = pybssn.decl("evo_shift", [ui],
                            rhs=evo_shift_rhs, parity=parity_vector)

    ###
    # Evolved BSSN Vars.
    ###
    # \phi
    phi_rhs = pybssn.decl("phi_rhs", [], parity=parity_scalar)
    phi = pybssn.decl("phi", [], rhs=phi_rhs, parity=parity_scalar)

    # \tilde{\gamma_{ij}}
    gt_rhs = pybssn.decl("gt_rhs", [li, lj], parity=parity_sym2ten)
    pybssn.add_sym(gt_rhs[li, lj], li, lj)
    gt = pybssn.decl("gt", [li, lj], rhs=gt_rhs, parity=parity_sym2ten)
    pybssn.add_sym(gt[li, lj], li, lj)

    # \tilde{A}_{ij}
    At_rhs = pybssn.decl("At_rhs", [li, lj], parity=parity_sym2ten)
    pybssn.add_sym(At_rhs[li, lj], li, lj)
    At = pybssn.decl("At", [li, lj], rhs=At_rhs, parity=parity_sym2ten)
    pybssn.add_sym(At[li, lj], li, lj)

    # trace of Extrinsic Curvature
    trK_rhs = pybssn.decl("trK_rhs", [], parity=parity_scalar)
    trK = pybssn.decl("trK", [], rhs=trK_rhs, parity=parity_scalar)

    # \tilde{\Gamma}^i
    ConfConnect_rhs = pybssn.decl(
        "ConfConnect_rhs", [ui], parity=parity_vector)
    ConfConnect = pybssn.decl("ConfConnect", [ui],
                              rhs=ConfConnect_rhs, parity=parity_vector)

    ###
    # Constraint Vars.
    ###
    HamCons = pybssn.decl("HamCons", [], parity=parity_scalar)
    MomCons = pybssn.decl("MomCons", [ui], parity=parity_vector)

    ###
    # Aux. Vars.
    ###
    Gammat = pybssn.decl("Gammat", [ua, lb, lc])  # \tilde{\Gamma}^a_{bc}
    pybssn.add_sym(Gammat[ua, lb, lc], lb, lc)

    Gamma = pybssn.decl("Gamma", [ua, lb, lc])  # \Gamma^a_{bc}
    pybssn.add_sym(Gamma[ua, lb, lc], lb, lc)

    ric = pybssn.decl("ric", [li, lj])  # R_{ij} = \tilde{R}_{ij} + R^\phi_{ij}
    pybssn.add_sym(ric[li, lj], li, lj)

    DD_lapse = pybssn.decl("DD_lapse", [li, lj])  # D_i D_j \alpha
    pybssn.add_sym(DD_lapse[li, lj], li, lj)

    DD_div_lapse = pybssn.decl("DD_div_lapse", [])  # D^i D_i \alpha

    T = pybssn.decl("T", [li, lj])  # T_{ij} = -D_i D_j \alpha + \alpha R_{ij}
    pybssn.add_sym(T[li, lj], li, lj)

    # Prevents the elimination of ConfConnect_rhs
    ConfConnect_rhs_tmp = pybssn.decl("ConfConnect_rhs_tmp", [ui])

    ###
    # Kronecker Delta
    ###
    kronecker_delta_mat = mkMatrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    kronecker_delta = pybssn.decl("kronecker_delta", [ui, lj])
    pybssn.add_sym(kronecker_delta[ui, lj], ui, lj)

    pybssn.mk_subst(kronecker_delta[ui, lj], kronecker_delta_mat)

    ###
    # Substitution rules
    ###
    g_mat = pybssn.get_matrix(g[li, lj])
    g_imat = do_inv(g_mat)
    detg = do_det(g_mat)
    pybssn.mk_subst(g[ui, uj], g_imat)

    pybssn.mk_subst(gt_rhs[li, lj])
    pybssn.mk_subst(gt[li, lj])

    gt_mat = pybssn.get_matrix(gt[li, lj])
    detgt = do_det(gt_mat)
    gt_imat = do_inv(gt_mat) * detgt  # Use the fact that det(gt) = 1
    pybssn.mk_subst(gt[ui, uj], gt_imat)

    pybssn.mk_subst(At[li, lj])
    pybssn.mk_subst(At_rhs[li, lj])
    pybssn.mk_subst(At[ui, uj])
    pybssn.mk_subst(At[ui, lj])

    pybssn.mk_subst(ConfConnect[ui])
    pybssn.mk_subst(ConfConnect_rhs[ui])

    pybssn.mk_subst(evo_shift[ui])
    pybssn.mk_subst(evo_shift_rhs[ui])

    pybssn.mk_subst(Gammat[ua, lb, lc])
    pybssn.mk_subst(Gammat[la, lb, lc])

    pybssn.mk_subst(Gamma[ua, lb, lc])

    pybssn.mk_subst(ric[li, lj])

    pybssn.mk_subst(DD_lapse[li, lj])

    pybssn.mk_subst(T[li, lj])

    pybssn.mk_subst(ConfConnect_rhs_tmp[ui])

    pybssn.mk_subst(MomCons[ui])

    ###
    # Aux. functions
    ###
    def sym(expr: Expr, ind1: Idx, ind2: Idx) -> Expr:
        """
        Index symmetrizer
        """
        uA, lA = mkPair()
        # swap ind1 and ind2
        x1: Expr = do_subs(expr, {ind1: uA, ind2: lA})
        x2: Expr = do_subs(x1,   {uA: ind2, lA: ind1})
        # add expr to itself with swapped indices
        x3: Expr = (expr + x2)/2
        return x3

    def compute_ricci(function: ThornFunction) -> None:
        """
        Adds equations to a function that compute the Ricci tensor.
        """

        # B-S Eq. (1.18)
        function.add_eqn(
            Gammat[ld, lb, lc],
            1 / 2 * (
                div(gt[ld, lb], lc) + div(gt[ld, lc], lb) - div(gt[lb, lc], ld)
            )
        )

        function.add_eqn(Gammat[ua, lb, lc], gt[ua, ud] * Gammat[ld, lb, lc])

        function.add_eqn(
            ric[li, lj],

            # \tilde{R}_{ij}
            - (1/2) * gt[ua, ub] * div(gt[li, lj], lb, la)
            + sym(gt[lk, li] * div(ConfConnect[uk], lj), li, lj)
            + sym(gt[ua, ub] * Gammat[uk, la, lb] * Gammat[li, lj, lk], li, lj)
            + gt[ua, ub] * (
                2 * sym(Gammat[uk, la, li] * Gammat[lj, lk, lb], li, lj)
                + Gammat[uk, li, lb] * Gammat[lk, la, lj]
            )

            # R^{\phi}_{ij}
            - 2 * (div(phi, lj, li) - Gammat[uk, li, lj] * div(phi, lk))
            - 2 * gt[li, lj] * div(gt[ua, ub] * div(phi, lb), la)
            + 4 * div(phi, li) * div(phi, lj)
            - 4 * gt[li, lj] * gt[ua, ub] * div(phi, la) * div(phi, lb)
        )

    ###
    # Aux. groups
    ###
    # Initialization
    initial_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("BSSN_InitialGroup"),
        at_or_in=AtOrIn.In,
        schedule_bin=Identifier("ODESolvers_Initial"),
        description=String("BSSN initialization routines"),
        after=[Identifier("ADMBaseX_PostInitial")]
    )

    # RHS
    rhs_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("BSSN_RHSGroup"),
        at_or_in=AtOrIn.In,
        schedule_bin=Identifier("ODESolvers_RHS"),
        description=String("BSSN equations RHS computation"),
    )

    # Post-step
    poststep_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("BSSN_PostStepGroup"),
        at_or_in=AtOrIn.In,
        schedule_bin=Identifier("ODESolvers_PostStep"),
        description=String("BSSN post time step routines"),
        before=[Identifier("ADMBaseX_SetADMVars")]
    )

    # Analysis
    analysis_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("BSSN_AnalysisGroup"),
        at_or_in=AtOrIn.At,
        schedule_bin=Identifier("analysis"),
        description=String("BSSN analysis routones"),
    )

    ###
    # State synchronization
    ###
    state_sync = ExplicitSyncBatch(
        vars=[gt, phi, At, trK, ConfConnect, evo_lapse, evo_shift],
        schedule_target=poststep_group,
        name="state_sync"
    )

    ###
    # Convert ADM to BSSN variables
    ###
    fun_adm2bssn = pybssn.create_function(
        "adm2bssn",
        initial_group
    )

    fun_adm2bssn.add_eqn(
        gt[li, lj],
        cbrt(1 / detg) * g[li, lj]
    )

    fun_adm2bssn.add_eqn(phi, 1 / 12 * log(detg))

    fun_adm2bssn.add_eqn(
        At[li, lj],
        cbrt(1 / detg) * (
            k[li, lj] - (1/3) * g[li, lj] * g[ua, ub] * k[la, lb]
        )
    )

    fun_adm2bssn.add_eqn(trK, g[ua, ub] * k[la, lb])

    fun_adm2bssn.add_eqn(
        ConfConnect[ui],
        -div(cbrt(detg) * g[ui, uj], lj)  # TODO: Expand this derivative?
    )

    fun_adm2bssn.add_eqn(evo_lapse, alp)
    fun_adm2bssn.add_eqn(evo_shift[ua], beta[ua])

    fun_adm2bssn.bake()

    ###
    # Convert BSSN to ADM variables
    ###
    fun_bssn2adm = pybssn.create_function(
        "bssn2adm",
        poststep_group,
        schedule_after=["state_sync"]
    )

    fun_bssn2adm.add_eqn(g[li, lj], exp(4 * phi) * gt[li, lj])

    fun_bssn2adm.add_eqn(
        k[li, lj],
        exp(4 * phi) * At[li, lj] + (1/3) * exp(4 * phi) * gt[li, lj] * trK
    )

    fun_bssn2adm.add_eqn(alp, evo_lapse)
    fun_bssn2adm.add_eqn(beta[ua], evo_shift[ua])

    fun_bssn2adm.bake()

    ###
    # Compute constraints
    ###
    fun_bssn_cons = pybssn.create_function(
        "bssn_cons",
        analysis_group
    )

    compute_ricci(fun_bssn_cons)

    # TODO: Different than canoli
    fun_bssn_cons.add_eqn(
        HamCons,
        exp(-4 * phi) * gt[ui, uj] * ric[li, lj]
        + (2/3) * trK * trK
        - gt[ui, ua] * gt[uj, ub] * At[la, lb] * At[li, lj]
    )

    # TODO: Different than canoli
    fun_bssn_cons.add_eqn(
        MomCons[ui],
        gt[ui, ua] * gt[uj, ub] * (
            div(At[la, lb], lj)
            - Gammat[uk, lj, la] * At[lk, lb]
            - Gammat[uk, lj, lb] * At[la, lk]
        )
        + 6 * gt[ui, ua] * gt[uj, ub] * At[la, lb] * div(phi, lj)
        - (2/3) * gt[ui, uj] * div(trK, lj)
    )

    fun_bssn_cons.bake()

    ###
    # BSSN Evolution equations
    # Following [1], we will replace \tilde{\Gamma}^i with
    # \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk} whenever the
    # \tilde{\Gamma}^i are needed without derivatives.
    #
    # TODO: Following [4] FD stencils are centered except for terms
    # of the form (\shift^i \partial_i u) which are calculated
    # using an “upwind” stencil which is shifted by one point in
    # the direction of the shift, and of the same order
    ###
    fun_bssn_rhs = pybssn.create_function(
        "rhs",
        rhs_group
    )

    # Aux. Equations
    fun_bssn_rhs.add_eqn(At[ui, lj], At[la, lj] * gt[ua, ui])
    fun_bssn_rhs.add_eqn(At[ui, uj], At[ui, lb] * gt[ub, uj])

    compute_ricci(fun_bssn_rhs)

    fun_bssn_rhs.add_eqn(
        DD_lapse[lj, lk],
        div(evo_lapse, lk, lj)
        - Gammat[ui, lj, lk] * div(evo_lapse, li)
        - 2 * kronecker_delta[ui, lj] * div(phi, lk) * div(evo_lapse, li)
        - 2 * kronecker_delta[ui, lk] * div(phi, lj) * div(evo_lapse, li)
        + 2 * gt[ui, ua] * gt[lj, lk] * div(phi, la) * div(evo_lapse, li)
    )

    fun_bssn_rhs.add_eqn(
        DD_div_lapse,
        exp(-4 * phi) * (
            2 * gt[ui, uj] * div(phi, li) * div(evo_lapse, lj)
            - gt[ui, uk] * Gammat[uj, li, lk] * div(phi, lj)
            + gt[ui, uj] * div(phi, lj, li)
        )
    )

    fun_bssn_rhs.add_eqn(
        T[li, lj],
        -DD_lapse[li, lj] + evo_lapse * ric[li, lj]
    )

    # Evolution equations
    fun_bssn_rhs.add_eqn(
        gt_rhs[li, lj],
        -2 * evo_lapse * At[li, lj]
        + evo_shift[uk] * div(gt[li, lj], lk)
        + gt[li, lk] * div(evo_shift[uk], lj)
        + gt[lj, lk] * div(evo_shift[uk], li)
        - (2/3) * gt[li, lj] * div(evo_shift[uk], lk)
    )

    fun_bssn_rhs.add_eqn(
        phi_rhs,
        -(1/6) * evo_lapse * trK
        + div(phi, lk) * evo_shift[uk]
        + (1/6) * div(evo_shift[uk], lk)
    )

    # See [3]
    # Let T_{ij} \equiv -D_i D_j \alpha + \alpha R_{ij}
    # The trace free part of T, T^{(0)}_{ij} is then
    # T^{(0)}_{ij} = T_{ij} - 1/3 \gamma_{ij} \gamma^{ab} T_{ab}
    fun_bssn_rhs.add_eqn(
        At_rhs[li, lj],
        exp(-4 * phi) * (
            T[li, lj] - (1/3) * gt[li, lj] * gt[ua, ub] * T[la, lb]
        )
        + evo_lapse * (trK * At[li, lj] - 2 * At[li, lk] * At[uk, lj])
        + evo_shift[uk] * div(At[li, lj], lk)
        + At[li, lk] * div(evo_shift[uk], lj)
        + At[lj, lk] * div(evo_shift[uk], li)
        - (2/3) * At[li, lj] * div(evo_shift[uk], lk)
    )

    fun_bssn_rhs.add_eqn(
        trK_rhs,
        -DD_div_lapse
        + evo_lapse * (At[ui, uj] * At[li, lj] + (1/3) * trK**2)
        + evo_shift[uk] * div(trK, lk)
    )

    fun_bssn_rhs.add_eqn(
        ConfConnect_rhs_tmp[ui],
        gt[uj, uk] * div(evo_shift[ui], lk, lj)
        + (1/3) * gt[ui, uj] * div(evo_shift[uk], lk, lj)
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
    fun_bssn_rhs.add_eqn(ConfConnect_rhs[ui], ConfConnect_rhs_tmp[ui])

    # 1 + log lapse. See [6]
    fun_bssn_rhs.add_eqn(
        evo_lapse_rhs,
        zeta_alpha * evo_shift[ui] * div(evo_lapse, li)
        - kappa_alpha * evo_lapse * trK
    )

    # Hyperbolic Gamma Driver shift
    fun_bssn_rhs.add_eqn(
        evo_shift_rhs[ua],
        zeta_beta * evo_shift[uj] * div(evo_shift[ua], lj)
        + beta_Gamma * evo_lapse**beta_Alp * ConfConnect[ua]
        - eta_beta * evo_shift[ua]
    )

    fun_bssn_rhs.bake()

    ###
    # Thorn creation
    ###
    CppCarpetXWizard(
        pybssn,
        CppCarpetXGenerator(
            pybssn,
            interior_sync_mode=InteriorSyncMode.Never,
            extra_schedule_blocks=[
                initial_group,
                rhs_group,
                poststep_group,
                analysis_group
            ],
            explicit_syncs=[state_sync]
        )
    ).generate_thorn()

# References
# [1] https://docs.einsteintoolkit.org/et-docs/images/0/05/PeterDiener15-MacLachlan.pdf
# [2] https://arxiv.org/abs/gr-qc/9810065
# [3] https://arxiv.org/pdf/2109.11743.
# [4] https://arxiv.org/pdf/0910.3803
# [6] https://arxiv.org/abs/gr-qc/0605030.
