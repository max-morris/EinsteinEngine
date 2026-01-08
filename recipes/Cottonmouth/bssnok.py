from EmitCactus import *
from sympy import Rational

###
# Thorn definitions
###
cottonmouth_bssnok = ThornDef("Cottonmouth", "CottonmouthBSSNOK")

###
# Code generation options
###
gen_opts = {
    "do_cse": False,
    "do_madd": False,
    "do_recycle_temporaries": False,
    "do_split_output_eqns": True
}

###
# Finite difference stencils
###

# Fifth order Kreiss-Oliger disspation stencil
div_diss = cottonmouth_bssnok.mk_stencil(
    "div_diss",
    la,
    Rational(1, 64) * DDI(la) * (
        noop(stencil(-3*la) + stencil(3*la))
        - 6.0 * noop(stencil(-2*la) + stencil(2*la))
        + 15.0 * noop(stencil(-la) + stencil(la))
        - 20.0 * stencil(0)
    )
)

###
# Extra math functions
###
max = cottonmouth_bssnok.decl_fun("max", args=2, is_stencil=False)

###
# Thorn parameters
###
eta_B = cottonmouth_bssnok.add_param(
    "eta_b",
    default=1.0,
    desc="Mass dependent damping coefficient for the hyperbolic gamma driver shift"
)

conformal_factor_floor = cottonmouth_bssnok.add_param(
    "conformal_factor_floor",
    default=1.0e-10,
    desc="The conformal factor W will never be smaller than this value"
)

evolved_lapse_floor = cottonmouth_bssnok.add_param(
    "evolved_lapse_floor",
    default=1.0e-10,
    desc="The evolved lapse will never be smaller than this value"
)

dissipation_epsilon = cottonmouth_bssnok.add_param(
    "dissipation_epsilon",
    default=0.2,
    desc="The ammount of dissipation to add. Should be in the [0, 1/3[ range"
)

###
# Tensor parities
###
# fmt: off
parity_scalar = parities(+1,+1,+1)
parity_vector = parities(-1,+1,+1,  +1,-1,+1,  +1,+1,-1)
parity_sym2ten = parities(+1,+1,+1,  -1,-1,+1,  -1,+1,-1,  +1,+1,+1,  +1,-1,-1,  +1,+1,+1)
# fmt: on

###
# ADMBaseX vars.
###
g = cottonmouth_bssnok.decl(
    "g",
    [la, lb],
    symmetries=[(la, lb)],
    from_thorn="ADMBaseX"
)

k = cottonmouth_bssnok.decl(
    "k",
    [la, lb],
    symmetries=[(la, lb)],
    from_thorn="ADMBaseX"
)

alp = cottonmouth_bssnok.decl("alp", [], from_thorn="ADMBaseX")

beta = cottonmouth_bssnok.decl("beta", [ua], from_thorn="ADMBaseX")

dtbeta = cottonmouth_bssnok.decl("dtbeta", [ua], from_thorn="ADMBaseX")

###
# TmunuBaseX vars.
###
eTtt = cottonmouth_bssnok.decl("eTtt", [], from_thorn="TmunuBaseX")

eTti = cottonmouth_bssnok.decl("eTt", [la], from_thorn="TmunuBaseX")


eTij = cottonmouth_bssnok.decl(
    "eT",
    [la, lb],
    symmetries=[(la, lb)],
    from_thorn="TmunuBaseX"
)

###
# Evolved Gauge Vars.
###
evo_lapse_rhs = cottonmouth_bssnok.decl(
    "evo_lapse_rhs",
    [],
    parity=parity_scalar
)

evo_lapse = cottonmouth_bssnok.decl(
    "evo_lapse",
    [],
    rhs=evo_lapse_rhs,
    parity=parity_scalar
)

evo_shift_rhs = cottonmouth_bssnok.decl(
    "evo_shift_rhs",
    [ua],
    parity=parity_vector
)

evo_shift = cottonmouth_bssnok.decl(
    "evo_shift",
    [ua],
    rhs=evo_shift_rhs,
    parity=parity_vector
)

shift_B_rhs = cottonmouth_bssnok.decl(
    "shift_B_rhs",
    [ua],
    parity=parity_vector
)

shift_B = cottonmouth_bssnok.decl(
    "shift_B",
    [ua],
    rhs=shift_B_rhs,
    parity=parity_vector
)

###
# Evolved BSSN Vars.
###
# w (conformal factor)
w_rhs = cottonmouth_bssnok.decl("w_rhs", [], parity=parity_scalar)
w = cottonmouth_bssnok.decl("w", [], rhs=w_rhs, parity=parity_scalar)

# \tilde{\gamma_{a b}}
gt_rhs = cottonmouth_bssnok.decl(
    "gt_rhs",
    [la, lb],
    symmetries=[(la, lb)],
    parity=parity_sym2ten
)

gt = cottonmouth_bssnok.decl(
    "gt",
    [la, lb],
    symmetries=[(la, lb)],
    rhs=gt_rhs,
    parity=parity_sym2ten
)

# \tilde{A}_{a b}
At_rhs = cottonmouth_bssnok.decl(
    "At_rhs",
    [la, lb],
    symmetries=[(la, lb)],
    parity=parity_sym2ten
)

At = cottonmouth_bssnok.decl(
    "At",
    [la, lb],
    symmetries=[(la, lb)],
    rhs=At_rhs,
    parity=parity_sym2ten
)

# K (trace of Extrinsic Curvature)
trK_rhs = cottonmouth_bssnok.decl("trK_rhs", [], parity=parity_scalar)
trK = cottonmouth_bssnok.decl("trK", [], rhs=trK_rhs, parity=parity_scalar)

# \tilde{\Gamma}^a
ConfConnect_rhs = cottonmouth_bssnok.decl(
    "ConfConnect_rhs",
    [ua],
    parity=parity_vector
)

ConfConnect = cottonmouth_bssnok.decl(
    "ConfConnect",
    [ua],
    rhs=ConfConnect_rhs,
    parity=parity_vector
)

###
# Monitored constraint Vars.
###
HamCons = cottonmouth_bssnok.decl("HamCons", [], parity=parity_scalar)
MomCons = cottonmouth_bssnok.decl("MomCons", [ua], parity=parity_vector)
DeltaCons = cottonmouth_bssnok.decl("DeltaCons", [ua], parity=parity_vector)

###
# Enforced Constraint Vars.
###
# TODO: It would be good if this was not required.
w_enforce = cottonmouth_bssnok.decl(
    "w_enforce",
    [],
    parity=parity_scalar
)

evo_lapse_enforce = cottonmouth_bssnok.decl(
    "evo_lapse_enforce",
    [],
    parity=parity_scalar
)

gt_enforce = cottonmouth_bssnok.decl(
    "gt_enforce",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

At_enforce = cottonmouth_bssnok.decl(
    "At_enforce",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

###
# Ricci tensor.
# We single out the Ricci tensor and compute it on its own function in order
# to increase efficiency
###

# \tilde{R}_{a b}
Rt = cottonmouth_bssnok.decl("Rt", [la, lb], symmetries=[(la, lb)])

# \tilde{R}^{\phi}_{a b}
RPhi = cottonmouth_bssnok.decl("RPhi", [la, lb], symmetries=[(la, lb)])

# R_{a b} = \tilde{R}_{a b} + R^\phi_{a b}
R = cottonmouth_bssnok.decl(
    "R",
    [la, lb],
    symmetries=[(la, lb)],
    parity=parity_sym2ten
)

###
# Matter terms.
###

#  n^a n^b T_{ab}
rho = cottonmouth_bssnok.decl("rho", [])

# -p^a_i n^b T_{ab}, where p^a_i = \delta^a_i + n^a n_i
S = cottonmouth_bssnok.decl("S", [la])

# \gamma^{ij} T_{ij}
trS = cottonmouth_bssnok.decl("trS", [])

###
# Aux. Vars.
###
# \tilde{\Gamma}_{abc}
Gammat = cottonmouth_bssnok.decl("Gammat", [la, lb, lc], symmetries=[(lb, lc)])

# Temporary storage for \partial_t \tilde{\Gamma}^{a}
# This is required because this quantity is both written to ConfConnect_rhs
# and read in the gamma driver shift evolution
ConfConnect_rhs_tmp = cottonmouth_bssnok.decl("ConfConnect_rhs_tmp", [ua])

# \tilde{\gamma}^{i, j} \tilde{\Gamma}^a_{a b}
Delta = cottonmouth_bssnok.decl("Delta", [ua])

# -D_a D_b \alpha + \alpha R_{a b}
Ats = cottonmouth_bssnok.decl("Ats", [la, lb], symmetries=[(la, lb)])

# \tilde{D}_a \phi
cdphi = cottonmouth_bssnok.decl("cdphi", [la])

# \tilde{D}_a \tilde{D}_b \phi
cdphi2 = cottonmouth_bssnok.decl("cdphi2", [la, lb], symmetries=[(la, lb)])

###
# Substitution rules
###
# Conformal metric and its inverse
g_mat = cottonmouth_bssnok.get_matrix(g[la, lb])
g_imat = inv(g_mat)
detg = det(g_mat)
cottonmouth_bssnok.add_substitution_rule(g[ua, ub], g_imat)

gt_mat = cottonmouth_bssnok.get_matrix(gt[la, lb])
detgt = det(gt_mat)
gt_imat = inv(gt_mat) * detgt  # Use the fact that det(gt) = 1
cottonmouth_bssnok.add_substitution_rule(gt[ua, ub], gt_imat)

# At
cottonmouth_bssnok.add_substitution_rule(At[ua, lb], gt[ua, uc] * At[lc, lb])
cottonmouth_bssnok.add_substitution_rule(At[ua, ub], gt[ub, uc] * At[ua, lc])

# Gammat (Conformal connection)
cottonmouth_bssnok.add_substitution_rule(
    Gammat[lc, la, lb],
    Rational(1, 2) * (
        D(gt[lc, la], lb) + D(gt[lc, lb], la) - D(gt[la, lb], lc)
    )
)

cottonmouth_bssnok.add_substitution_rule(
    Gammat[ua, lb, lc], gt[ua, ud] * Gammat[ld, lb, lc]
)

cottonmouth_bssnok.add_substitution_rule(
    Gammat[la, lb, uc], gt[uc, ud] * Gammat[la, lb, ld]
)

cottonmouth_bssnok.add_substitution_rule(
    Delta[ua],
    gt[ub, uc] * gt[ua, ud] * Gammat[ld, lb, lc]
)

# Phi derivatives w.r.t the conformal metric
cottonmouth_bssnok.add_substitution_rule(
    cdphi[la],
    -Rational(1, 2) * (1 / w) * D(w, la)
)

# Matter
cottonmouth_bssnok.add_substitution_rule(
    rho,
    1 / evo_lapse**2 * (
        eTtt - 2 * evo_shift[ua] * eTti[la]
        + evo_shift[ua] * evo_shift[ub] * eTij[la, lb]
    )
)

cottonmouth_bssnok.add_substitution_rule(
    S[la],
    -1/evo_lapse * (eTti[la] - evo_shift[ub] * eTij[la, lb])
)

cottonmouth_bssnok.add_substitution_rule(
    trS,
    w**2 * gt[ua, ub] * eTij[la, lb]
)

###
# Aux. groups
###

# Initialization
initial_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthBSSNOK_InitialGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_Initial"),
    after=[Identifier("ADMBaseX_PostInitial")],
    description=String("BSSNOK initialization routines")
)

# RHS
rhs_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthBSSNOK_RHSGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_RHS"),
    description=String("BSSNOK equations RHS computation"),
)

# Ricci tensor
ricci_group_rhs = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthBSSNOK_RicciGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_RHS"),
    before=[Identifier("CottonmouthBSSNOK_RHSGroup")],
    description=String("BSSNOK Ricci tensor computation"),
)

ricci_group_analysis = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthBSSNOK_RicciGroup"),
    at_or_in=AtOrIn.At,
    schedule_bin=Identifier("analysis"),
    before=[Identifier("CottonmouthBSSNOK_AnalysisGroup")],
    description=String("BSSNOK Ricci tensor computation"),
)

# Post-step
post_step_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthBSSNOK_PostStepGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_PostStep"),
    before=[Identifier("ADMBaseX_SetADMVars")],
    description=String("BSSNOK post-step routines")
)

# Analysis
analysis_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthBSSNOK_AnalysisGroup"),
    at_or_in=AtOrIn.At,
    schedule_bin=Identifier("analysis"),
    description=String("BSSNOK analysis routines"),
)

###
# Enforce algebraic constraints
###
fun_bssn_enforce_pt1 = cottonmouth_bssnok.create_function(
    "cottonmouth_bssnok_enforce_pt1",
    post_step_group,
    schedule_after=["StateSync"],
    schedule_before=["cottonmouth_bssnok_enforce_pt2"]
)

# Enforce \det(\tilde{\gamma}) = 1
fun_bssn_enforce_pt1.add_eqn(
    gt_enforce[li, lj],
    gt[li, lj] / (cbrt(detgt))
)

# Enforce \tilde{\gamma}^{i j} \tilde{A}_{ij} = 0
fun_bssn_enforce_pt1.add_eqn(
    At_enforce[li, lj],
    At[li, lj] - Rational(1, 3) * gt[li, lj] * gt[ua, ub] * At[la, lb]
)

# Enforce conformal factor floor
fun_bssn_enforce_pt1.add_eqn(
    w_enforce,
    max(w, conformal_factor_floor)
)

# Enforce conformal factor floor
fun_bssn_enforce_pt1.add_eqn(
    evo_lapse_enforce,
    max(evo_lapse, evolved_lapse_floor)
)

fun_bssn_enforce_pt2 = cottonmouth_bssnok.create_function(
    "cottonmouth_bssnok_enforce_pt2",
    post_step_group,
    schedule_after=["cottonmouth_bssnok_enforce_pt1"],
    schedule_before=["cottonmouth_bssnok_bssn2adm"]
)

fun_bssn_enforce_pt2.add_eqn(gt[li, lj], gt_enforce[li, lj])
fun_bssn_enforce_pt2.add_eqn(At[li, lj], At_enforce[li, lj])
fun_bssn_enforce_pt2.add_eqn(w, w_enforce)
fun_bssn_enforce_pt2.add_eqn(evo_lapse, evo_lapse_enforce)


###
# Convert ADM to BSSN variables
###
fun_adm2bssn = cottonmouth_bssnok.create_function(
    "cottonmouth_bssnok_adm2bssn",
    initial_group
)

fun_adm2bssn.add_eqn(
    gt[la, lb],
    (1 / cbrt(detg)) * g[la, lb]
)

fun_adm2bssn.add_eqn(w, 1 / (sqrt(cbrt(detg))))

fun_adm2bssn.add_eqn(
    At[la, lb],
    (1 / cbrt(detg)) * (
        k[la, lb]
        - Rational(1, 3) * g[la, lb] * g[uc, ud] * k[lc, ld]
    )
)

fun_adm2bssn.add_eqn(trK, g[ua, ub] * k[la, lb])

fun_adm2bssn.add_eqn(
    ConfConnect[ua],
    -Rational(1, 3) * (1 / (cbrt(detg)**2)) * (
        3 * detg * D(g[ua, ub], lb)
        + g[ua, ub] * D(detg, lb)
    )
)

fun_adm2bssn.add_eqn(evo_lapse, alp)
fun_adm2bssn.add_eqn(evo_shift[ua], beta[ua])

fun_adm2bssn.add_eqn(
    shift_B[ua],
    Rational(4, 3) * (1 / alp) * (
        dtbeta[ua]
        - beta[ub] * D(beta[ua], lb)
    )
)


###
# Convert BSSN to ADM variables
###
fun_bssn2adm = cottonmouth_bssnok.create_function(
    "cottonmouth_bssnok_bssn2adm",
    post_step_group,
    schedule_after=["cottonmouth_bssnok_enforce_pt2"]
)

fun_bssn2adm.add_eqn(g[li, lj], (1/(w**2)) * gt[li, lj])

fun_bssn2adm.add_eqn(
    k[li, lj],
    (1 / (w**2)) * (
        At[li, lj]
        + Rational(1, 3) * gt[li, lj] * trK
    )
)

fun_bssn2adm.add_eqn(alp, evo_lapse)
fun_bssn2adm.add_eqn(beta[ua], evo_shift[ua])


###
# Compute the Ricci tensor
###
fun_bssn_ricci = cottonmouth_bssnok.create_function(
    "cottonmouth_bssnok_compute_ricci",
    ricci_group_analysis
)

# Aux. equations
fun_bssn_ricci.add_eqn(
    cdphi2[la, lb],
    -Rational(1, 2) * (1 / w) * (
        D(w, la, lb)
        - Gammat[uc, la, lb] * D(w, lc)
    )
    + Rational(1, 2) * (1 / (w**2)) * D(w, la) * D(w, lb)
)

fun_bssn_ricci.add_eqn(
    Rt[la, lb],
    - Rational(1, 2) * gt[uc, ud] * D(gt[la, lb], lc, ld)
    + Rational(1, 2) * gt[lc, la] * D(ConfConnect[uc], lb)
    + Rational(1, 2) * gt[lc, lb] * D(ConfConnect[uc], la)
    + Rational(1, 2) * Delta[uc] * Gammat[la, lb, lc]
    + Rational(1, 2) * Delta[uc] * Gammat[lb, la, lc]
    + (
        + Gammat[uc, la, ld] * Gammat[lb, lc, ud]
        + Gammat[uc, lb, ld] * Gammat[la, lc, ud]
        + Gammat[uc, la, ld] * Gammat[lc, lb, ud]
    )
)

fun_bssn_ricci.add_eqn(
    RPhi[la, lb],
    - 2 * cdphi2[lb, la]
    - 2 * gt[la, lb] * gt[uc, ud] * cdphi2[lc, ld]
    + 4 * cdphi[la] * cdphi[lb]
    - 4 * gt[la, lb] * gt[uc, ud] * cdphi[lc] * cdphi[ld]
)

# Ricci tensor
fun_bssn_ricci.add_eqn(R[la, lb], Rt[la, lb] + RPhi[la, lb])


###
# Compute non enforced constraints
###
fun_bssn_cons = cottonmouth_bssnok.create_function(
    "cottonmouth_bssnok_constraints",
    analysis_group
)

# Hamiltonian constraint
fun_bssn_cons.add_eqn(
    HamCons,
    (w**2) * gt[ua, ub] * R[la, lb]
    - At[ua, lb] * At[ub, la]
    + Rational(2, 3) * (trK**2)
    # Matter
    - 16 * pi * rho
)

# Momentum constraint
fun_bssn_cons.add_eqn(
    MomCons[ua],
    + gt[ua, uc] * gt[ub, ud] * (
        D(At[lc, ld], lb)
        - Gammat[uk, lc, lb] * At[lk, ld]
        - Gammat[uk, ld, lb] * At[lc, lk]
    )
    + 6 * At[ua, ub] * cdphi[lb]
    - Rational(2, 3) * gt[ua, ub] * D(trK, lb)
    # Matter
    - 8 * pi * w**2 * gt[ua, ub] * S[lb]
)

fun_bssn_cons.add_eqn(
    DeltaCons[ua],
    ConfConnect[ua] - Delta[ua]
)

###
# BSSN Evolution equations
# Following [1], we will replace \tilde{\Gamma}^i with
# \Delta^i \equiv \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk}
# whenever \tilde{\Gamma}^i are needed without derivatives.
#
# Following [4] FD stencils are centered except for terms
# of the form (\shift^i \partial_i u) which are calculated
# using an "upwind" stencil which is shifted by one point in
# the direction of the shift, and of the same order
###
fun_bssn_rhs = cottonmouth_bssnok.create_function(
    "cottonmouth_bssnok_rhs",
    rhs_group
)

# Aux. equations
fun_bssn_rhs.add_eqn(
    Ats[la, lb],
    (
        -D(evo_lapse, la, lb)
        + Gammat[uc, la, lb] * D(evo_lapse, lc)
    )
    + 2 * (
        D(evo_lapse, la) * cdphi[lb]
        + D(evo_lapse, lb) * cdphi[la]
    )
    + evo_lapse * R[la, lb]
)

# Evolution equations
fun_bssn_rhs.add_eqn(
    gt_rhs[la, lb],
    - 2 * evo_lapse * At[la, lb]
    + gt[la, lc] * D(evo_shift[uc], lb)
    + gt[lb, lc] * D(evo_shift[uc], la)
    - Rational(2, 3) * gt[la, lb] * D(evo_shift[uc], lc)
    # TODO: Advection: + Upwind[beta[uc], gt[la,lb], lc]
    + evo_shift[uc] * D(gt[la, lb], lc)
    # Dissipation:
    + dissipation_epsilon * (
        div_diss(gt[la, lb], l0)
        + div_diss(gt[la, lb], l1)
        + div_diss(gt[la, lb], l2)
    )
)

fun_bssn_rhs.add_eqn(
    w_rhs,
    Rational(1, 3) * w * (
        evo_lapse * trK
        - D(evo_shift[ua], la)
    )
    # TODO: Advection: + Upwind[beta[ua], phi, la]
    + evo_shift[ua] * D(w, la)
    # Dissipation:
    + dissipation_epsilon * (
        div_diss(w, l0)
        + div_diss(w, l1)
        + div_diss(w, l2)
    )
)

fun_bssn_rhs.add_eqn(
    At_rhs[la, lb],
    (w**2) * (
        Ats[la, lb]
        - Rational(1, 3) * gt[la, lb] * gt[uc, ud] * Ats[lc, ld]
    )
    + evo_lapse * (
        + trK * At[la, lb]
        - 2 * At[la, lc] * At[uc, lb]
    )
    + At[la, lc] * D(evo_shift[uc], lb)
    + At[lb, lc] * D(evo_shift[uc], la)
    - Rational(2, 3) * At[la, lb] * D(evo_shift[uc], lc)
    # Matter
    - w**2 * evo_lapse * 8 * pi * (
        eTij[la, lb] - Rational(1, 3) * gt[la, lb] * trS
    )
    # TODO: Advection: + Upwind[beta[uc], At[la,lb], lc]
    + evo_shift[uc] * D(At[la, lb], lc)
    # Dissipation:
    + dissipation_epsilon * (
        div_diss(At[la, lb], l0)
        + div_diss(At[la, lb], l1)
        + div_diss(At[la, lb], l2)
    )
)

fun_bssn_rhs.add_eqn(
    trK_rhs,
    - (w**2) * (
        gt[ua, ub] * (
            + D(evo_lapse, la, lb)
            + 2 * cdphi[la] * D(evo_lapse, lb)
        )
        - Delta[ua] * D(evo_lapse, la)
    )
    + evo_lapse * (
        At[ua, lb] * At[ub, la]
        + Rational(1, 3) * (trK**2)
    )
    # Matter
    + 4 * pi * evo_lapse * (rho + trS)
    # TODO: Advection: + Upwind[beta[ua], trK, la]
    + evo_shift[ua] * D(trK, la)
    # Dissipation:
    + dissipation_epsilon * (
        div_diss(trK, l0)
        + div_diss(trK, l1)
        + div_diss(trK, l2)
    )
)

fun_bssn_rhs.add_eqn(
    ConfConnect_rhs_tmp[ua],
    - 2 * At[ua, ub] * D(evo_lapse, lb)
    + 2 * evo_lapse * (
        + Gammat[ua, lb, lc] * At[ub, uc]
        - Rational(2, 3) * gt[ua, ub] * D(trK, lb)
        + 6 * At[ua, ub] * cdphi[lb]
    )
    + gt[ub, uc] * D(evo_shift[ua], lb, lc)
    + Rational(1, 3) * gt[ua, ub] * D(evo_shift[uc], lb, lc)
    - Delta[ub] * D(evo_shift[ua], lb)
    + Rational(2, 3) * Delta[ua] * D(evo_shift[ub], lb)
    # Matter
    - 16 * pi * evo_lapse * gt[ua, ub] * S[lb]
    # TODO: Advection: + Upwind[beta[ub], Xt[ua], lb]
    + evo_shift[ub] * D(ConfConnect[ua], lb)
    # Dissipation:
    + dissipation_epsilon * (
        div_diss(ConfConnect[ua], l0)
        + div_diss(ConfConnect[ua], l1)
        + div_diss(ConfConnect[ua], l2)
    )
)
fun_bssn_rhs.add_eqn(ConfConnect_rhs[ua], ConfConnect_rhs_tmp[ua])

# 1 + log lapse.
fun_bssn_rhs.add_eqn(
    evo_lapse_rhs,
    - 2 * evo_lapse * trK
    # TODO: Advection: Upwind[beta[ua], alpha, la]
    + evo_shift[ua] * D(evo_lapse, la)
    # Dissipation
    + dissipation_epsilon * (
        div_diss(evo_lapse, l0)
        + div_diss(evo_lapse, l1)
        + div_diss(evo_lapse, l2)
    )
)

# Hyperbolic Gamma Driver shift
fun_bssn_rhs.add_eqn(
    evo_shift_rhs[ua],
    Rational(3, 4) * evo_lapse * shift_B[ua]
    # TODO: Advection
    + evo_shift[ub] * D(evo_shift[ua], lb)
    # Dissipation
    + dissipation_epsilon * (
        div_diss(evo_shift[ua], l0)
        + div_diss(evo_shift[ua], l1)
        + div_diss(evo_shift[ua], l2)
    )
)

fun_bssn_rhs.add_eqn(
    shift_B_rhs[ua],
    ConfConnect_rhs_tmp[ua]
    - evo_shift[ub] * D(ConfConnect[ua], lb)
    - eta_B * shift_B[ua]
    # TODO: Advection
    + evo_shift[ub] * D(shift_B[ua], lb)
    # Dissipation
    + dissipation_epsilon * (
        div_diss(shift_B[ua], l0)
        + div_diss(shift_B[ua], l1)
        + div_diss(shift_B[ua], l2)
    )
)


fun_bssn_rhs.bake(**gen_opts)
fun_bssn_enforce_pt1.bake(**gen_opts)
fun_bssn_enforce_pt2.bake(**gen_opts)
fun_adm2bssn.bake(**gen_opts)
fun_bssn2adm.bake(**gen_opts)
fun_bssn_ricci.bake(**gen_opts)
fun_bssn_cons.bake(**gen_opts)

cottonmouth_bssnok.do_global_cse(promote_all())


###
# Thorn creation
###
CppCarpetXWizard(
    cottonmouth_bssnok,
    CppCarpetXGenerator(
        cottonmouth_bssnok,
        interior_sync_mode=InteriorSyncMode.MixedRhs,
        interior_sync_schedule_target=post_step_group,
        extra_schedule_blocks=[
            initial_group,
            rhs_group,
            analysis_group,
            ricci_group_rhs,
            post_step_group,
            ricci_group_analysis,
        ]
    )
).generate_thorn()

# References
# [1] https://docs.einsteintoolkit.org/et-docs/images/0/05/PeterDiener15-MacLachlan.pdf
# [2] https://github.com/nrpy/nrpy/blob/main/nrpy/equations/general_relativity/nrpylatex/test_parse_BSSN.py
# [3] https://arxiv.org/abs/gr-qc/9810065
# [4] https://arxiv.org/pdf/0910.3803
# [5] https://arxiv.org/abs/gr-qc/0605030.
# [6] https://arxiv.org/abs/1212.2901
