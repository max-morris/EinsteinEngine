from EinsteinEngine import *
from sympy import Rational
from enum import Enum


class Formulation(Enum):
    BSSNOK = 0
    CCZ4 = 1
    Z4C = 2


###
# Thorn definitions
###
cottonmouth_z4 = ThornDef("Cottonmouth", "CottonmouthZ4")

###
# Some more indices
###
ul, ll = mk_pair("l")
um, lm = mk_pair("m")

###
# Finite difference stencils
###

# Fourth order centered
cottonmouth_z4.set_derivative_stencil(5)

# Fifth order Kreiss-Oliger disspation stencil
div_diss = cottonmouth_z4.mk_stencil(
    "div_diss",
    li,
    Rational(1, 64) * DDI(li) * (
        noop(stencil(-3*li) + stencil(3*li))
        - 6.0 * noop(stencil(-2*li) + stencil(2*li))
        + 15.0 * noop(stencil(-li) + stencil(li))
        - 20.0 * stencil(0)
    )
)

###
# Extra math functions
###
def_max = cottonmouth_z4.decl_fun("max", args=2, is_stencil=False)

###
# Thorn parameters
###
chi_floor = cottonmouth_z4.add_param(
    "chi_floor",
    default=1.0e-4,
    desc="Chi will never be smaller than this value"
)

evolved_lapse_floor = cottonmouth_z4.add_param(
    "evolved_lapse_floor",
    default=1.0e-4,
    desc="The evolved lapse will never be smaller than this value"
)

dissipation_epsilon = cottonmouth_z4.add_param(
    "dissipation_epsilon",
    default=0.2,
    desc="The ammount of dissipation to add."
)

# TODO: Set range in [0,1]
par_s = cottonmouth_z4.add_param(
    "s",
    default=1,
    desc="TODO: Give this a better desc"
)

# TODO: Set range in [0,1]
par_c = cottonmouth_z4.add_param(
    "c",
    default=0,
    desc="TODO: Give this a better desc"
)

kappa = cottonmouth_z4.add_param(
    "kappa",
    default=1,
    desc="Constraint damping parameter. Must be of order 1 / L wehre L is the typical simulation scale."
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
g = cottonmouth_z4.decl(
    "g",
    [li, lj],
    symmetries=[(li, lj)],
    from_thorn="ADMBaseX"
)

k = cottonmouth_z4.decl(
    "k",
    [li, lj],
    symmetries=[(li, lj)],
    from_thorn="ADMBaseX"
)

alp = cottonmouth_z4.decl(
    "alp",
    [],
    from_thorn="ADMBaseX"
)

beta = cottonmouth_z4.decl(
    "beta",
    [ui],
    from_thorn="ADMBaseX"
)

dtbeta = cottonmouth_z4.decl(
    "dtbeta",
    [ui],
    from_thorn="ADMBaseX"
)

###
# TmunuBaseX vars.
###
eTtt = cottonmouth_z4.decl(
    "eTtt",
    [],
    from_thorn="TmunuBaseX"
)

eTti = cottonmouth_z4.decl(
    "eTt",
    [li],
    from_thorn="TmunuBaseX"
)

eTij = cottonmouth_z4.decl(
    "eT",
    [li, lj],
    symmetries=[(li, lj)],
    from_thorn="TmunuBaseX"
)

###
# Evolved Gauge Vars.
###
evo_lapse_rhs = cottonmouth_z4.decl(
    "evo_lapse_rhs",
    [],
    parity=parity_scalar
)

evo_lapse = cottonmouth_z4.decl(
    "evo_lapse",
    [],
    rhs=evo_lapse_rhs,
    parity=parity_scalar
)

evo_shift_rhs = cottonmouth_z4.decl(
    "evo_shift_rhs",
    [ui],
    parity=parity_vector
)

evo_shift = cottonmouth_z4.decl(
    "evo_shift",
    [ui],
    rhs=evo_shift_rhs,
    parity=parity_vector
)

shift_B_rhs = cottonmouth_z4.decl(
    "shift_B_rhs",
    [ui],
    parity=parity_vector
)

shift_B = cottonmouth_z4.decl(
    "shift_B",
    [ui],
    rhs=shift_B_rhs,
    parity=parity_vector
)

###
# Evolved Z4 vars.
###
# \Theta
Theta_rhs = cottonmouth_z4.decl(
    "Theta_rhs",
    [],
    parity=parity_scalar
)

Theta = cottonmouth_z4.decl(
    "Theta",
    [],
    rhs=Theta_rhs,
    parity=parity_scalar
)

# \chi
chi_rhs = cottonmouth_z4.decl(
    "chi_rhs",
    [],
    parity=parity_scalar
)

chi = cottonmouth_z4.decl(
    "chi",
    [],
    rhs=chi_rhs,
    parity=parity_scalar
)

# K (trace of Extrinsic Curvature)
trK_rhs = cottonmouth_z4.decl(
    "trK_rhs",
    [],
    parity=parity_scalar
)

trK = cottonmouth_z4.decl(
    "trK",
    [],
    rhs=trK_rhs,
    parity=parity_scalar
)

# \hat{\Gamma}^i
GammaHat_rhs = cottonmouth_z4.decl(
    "GammaHat_rhs",
    [ui],
    parity=parity_vector
)

GammaHat = cottonmouth_z4.decl(
    "GammaHat",
    [ui],
    rhs=GammaHat_rhs,
    parity=parity_vector
)

# \tilde{\gamma_{i j}}
gt_rhs = cottonmouth_z4.decl(
    "gt_rhs",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

gt = cottonmouth_z4.decl(
    "gt",
    [li, lj],
    symmetries=[(li, lj)],
    rhs=gt_rhs,
    parity=parity_sym2ten
)

# \tilde{A}_{ij}
At_rhs = cottonmouth_z4.decl(
    "At_rhs",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

At = cottonmouth_z4.decl(
    "At",
    [li, lj],
    symmetries=[(li, lj)],
    rhs=At_rhs,
    parity=parity_sym2ten
)

###
# Monitored constraint Vars.
###
HamCons = cottonmouth_z4.decl(
    "HamCons",
    [],
    parity=parity_scalar
)

MomCons = cottonmouth_z4.decl(
    "MomCons",
    [li],
    parity=parity_vector
)

ZtCons = cottonmouth_z4.decl(
    "ZtCons",
    [ui], parity=parity_vector
)

###
# Ricci tensor.
###
R = cottonmouth_z4.decl(
    "R",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

Rchi = cottonmouth_z4.decl(
    "Rchi",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

Rt = cottonmouth_z4.decl(
    "Rt",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

###
# Matter terms.
###

#  n^i n^j T_{ij}
rho = cottonmouth_z4.decl("rho", [])

# -p^j_i n^k T_{jk}, where p^a_i = \delta^a_i + n^a n_i
S = cottonmouth_z4.decl("S", [li])

# \gamma^{ij} T_{ij}
trS = cottonmouth_z4.decl("trS", [])

###
# Aux. Vars.
###
# \tilde{\Gamma}_{ijk}
Gammat = cottonmouth_z4.decl(
    "Gammat",
    [li, lj, lk],
    symmetries=[(lj, lk)]
)

# Delta = \tilde{\gamma}^{jk}\tilde{Gamma}^i_{jk}
Delta = cottonmouth_z4.decl("Delta", [ui])

# \hat{\Gamma}^i_c
GammaHatC = cottonmouth_z4.decl("GammaHatC", [ui])

# \hat{\Gamma}^i_{sc}
GammaHatSC = cottonmouth_z4.decl("GammaHatSC", [ui])

# \tilde{Z}^i
Zt = cottonmouth_z4.decl("Zt", [ui])

# Trace free part of \tilde{A}_{ij}
AtTF = cottonmouth_z4.decl("AtTF", [li, lj], symmetries=[(li, lj)])

# Temp. var to place the RHS of \pd_t\hat{Gamma}^i.
# It is required because this RHS is used in the evolution for B^i
GammaHat_rhs_tmp = cottonmouth_z4.decl("GammaHat_rhs_tmp", [ui])

###
# Substitution rules
###
# Physical metric and its inverse
g_mat = cottonmouth_z4.get_matrix(g[li, lj])
g_imat = inv(g_mat)
detg = det(g_mat)
cottonmouth_z4.add_substitution_rule(g[ui, uj], g_imat)

# Conformal metric and its inverse
gt_mat = cottonmouth_z4.get_matrix(gt[li, lj])
detgt = det(gt_mat)

# Use the fact that det(gt) = 1 to simplify the inverse expression
# Note that det(gt) = 1 is an *enforced* constraint
gt_imat = inv(gt_mat) * detgt
cottonmouth_z4.add_substitution_rule(gt[ui, uj], gt_imat)

# \tilde{\Gamma}_{ijk}. Eq. (2.16) of [1]
cottonmouth_z4.add_substitution_rule(
    Gammat[lk, li, lj],
    Rational(1, 2) * (
        D(gt[lj, lk], li) + D(gt[li, lk], lj) - D(gt[li, lj], lk)
    )
)

# \tilde{\Gamma}^i_{jk}. Eq. (2.14) of [1]
cottonmouth_z4.add_substitution_rule(
    Gammat[uk, li, lj], gt[uk, ul] * Gammat[ll, li, lj]
)

cottonmouth_z4.add_substitution_rule(
    Delta[ui], gt[uj, uk] * Gammat[ui, lj, lk]
)

# GammaHatC. Eq. (2.15) of [1]
cottonmouth_z4.add_substitution_rule(
    GammaHatC[ui],
    par_c * GammaHat[ui] + (1 - par_c) * Delta[ui]
)

# GammaHatSC. Eq. (2.15) of [1]
cottonmouth_z4.add_substitution_rule(
    GammaHatSC[ui],
    par_s * par_c * GammaHat[ui] + (1 - par_s * par_c) * Delta[ui]
)

# \tilde{Z}^i. Eq. (2.14) of [1]
cottonmouth_z4.add_substitution_rule(
    Zt[ui],
    Rational(1, 2) * (GammaHat[ui] - Delta[ui])
)

# At
cottonmouth_z4.add_substitution_rule(At[ui, lj], gt[ui, uk] * At[lk, lj])
cottonmouth_z4.add_substitution_rule(At[ui, uj], gt[uj, uk] * At[ui, lk])

# Matter term definitions. TODO: Review
cottonmouth_z4.add_substitution_rule(
    rho,
    1 / evo_lapse**2 * (
        eTtt - 2 * evo_shift[ui] * eTti[li]
        + evo_shift[ui] * evo_shift[uj] * eTij[li, lj]
    )
)

cottonmouth_z4.add_substitution_rule(
    S[li],
    -1/evo_lapse * (eTti[li] - evo_shift[uj] * eTij[li, lj])
)

cottonmouth_z4.add_substitution_rule(
    trS,
    chi * gt[ui, uj] * eTij[li, lj]
)

###
# Aux. groups
###

# Initialization
initial_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthZ4_InitialGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_Initial"),
    after=[Identifier("ADMBaseX_PostInitial")],
    description=String("Z4 initialization routines")
)

# Post-step
post_step_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthZ4_PostStepGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_PostStep"),
    before=[Identifier("ADMBaseX_SetADMVars")],
    description=String("Z4 post-step routines")
)

# RHS
rhs_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthZ4_RHSGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_RHS"),
    description=String("Z4 equations RHS computation"),
)

# Analysis
analysis_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthZ4_AnalysisGroup"),
    at_or_in=AtOrIn.At,
    schedule_bin=Identifier("analysis"),
    description=String("Z4 analysis routines"),
)

###
# Convert ADM to Z4 variables
#
###
fun_adm_to_z4_pt1 = cottonmouth_z4.create_function(
    "adm_to_z4_pt1",
    initial_group,
    intent_override=IntentOverride.WriteInterior
)

fun_adm_to_z4_pt1.add_eqn(
    Theta,
    sympify(0)
)

# Eq. (11) of [2]
fun_adm_to_z4_pt1.add_eqn(
    chi,
    1 / (cbrt(detg))
)

# Eq. (12) of [2]
fun_adm_to_z4_pt1.add_eqn(
    trK,
    g[ui, uj] * k[li, lj]
)

# Eq. (11) of [2]
fun_adm_to_z4_pt1.add_eqn(
    gt[li, lj],
    (1 / cbrt(detg)) * g[li, lj]
)

# Eq. (12) of [2]
fun_adm_to_z4_pt1.add_eqn(
    At[li, lj],
    (1 / cbrt(detg)) * (
        k[li, lj]
        - Rational(1, 3) * g[li, lj] * g[uk, ul] * k[lk, ll]
    )
)

fun_adm_to_z4_pt1.add_eqn(
    evo_lapse,
    alp
)

fun_adm_to_z4_pt1.add_eqn(
    evo_shift[ui],
    beta[ui]
)

# Eq. (2.27) of [1]
fun_adm_to_z4_pt1.add_eqn(
    shift_B[ui],
    Rational(4, 3) * (
        dtbeta[ui]
        - beta[uj] * D(beta[ui], lj)
    )
)

fun_adm_to_z4_pt2 = cottonmouth_z4.create_function(
    "adm_to_z4_pt2",
    initial_group,
    schedule_after=["adm_to_z4_pt1"]
)

# Eq. (2.6) of [1]
fun_adm_to_z4_pt2.add_eqn(
    GammaHat[ui],
    Delta[ui]
)

###
# Enforce algebraic constraints
###
fun_z4_enforce_pt1 = cottonmouth_z4.create_function(
    "z4_enforce_pt1",
    post_step_group
)

# Enforce chi floor
chi_enforce = cottonmouth_z4.overwrite(chi)

fun_z4_enforce_pt1.add_eqn(
    chi_enforce,
    def_max(chi, chi_floor)
)

# Enforce lapse floor
evo_lapse_enforce = cottonmouth_z4.overwrite(evo_lapse)

fun_z4_enforce_pt1.add_eqn(
    evo_lapse_enforce,
    def_max(evo_lapse, evolved_lapse_floor)
)

# Enforce \det(\tilde{\gamma}) = 1
gt_enforce = cottonmouth_z4.overwrite(gt)

fun_z4_enforce_pt1.add_eqn(
    gt_enforce[li, lj],
    gt[li, lj] / (cbrt(detgt))
)

fun_z4_enforce_pt2 = cottonmouth_z4.create_function(
    "z4_enforce_pt2",
    post_step_group,
    schedule_after=["enforce_pt1"]
)

# Enforce \tilde{\gamma}^{i j} \tilde{A}_{ij} = 0 (A)
At_enforce = cottonmouth_z4.overwrite(At)

fun_z4_enforce_pt2.add_eqn(
    At_enforce[li, lj],
    At[li, lj] - Rational(1, 3) * gt[li, lj] * gt[uk, ul] * At[lk, ll]
)

###
# Convert Z4 to ADM variables
###
fun_z4_to_adm = cottonmouth_z4.create_function(
    "z4_to_adm",
    post_step_group,
    schedule_after=["enforce_pt2"],
    intent_override=IntentOverride.E2E
)

# Eq. (2.4) of [1]
fun_z4_to_adm.add_eqn(
    g[li, lj],
    (1 / chi) * gt[li, lj]
)

# Eq. (2.5) of [1]
fun_z4_to_adm.add_eqn(
    k[li, lj],
    (1 / chi) * (
        At[li, lj]
        + Rational(1, 3) * gt[li, lj] * trK
    )
)

fun_z4_to_adm.add_eqn(
    alp,
    evo_lapse
)

fun_z4_to_adm.add_eqn(
    beta[ui],
    evo_shift[ui]
)

###
# Compute monitored constraints
###
fun_z4_constraints = cottonmouth_z4.create_function(
    "z4_constraints",
    analysis_group
)

# Eq. (2.18) of [1]
fun_z4_constraints.add_eqn(
    Rchi[li, lj],
    chi * (
        - Rational(1, 2) * gt[uk, ul] * D(gt[li, lj], lk, ll)

        + Rational(1, 2) * (
            + gt[lk, li] * D(GammaHat[uk], lj)
            + gt[lk, lj] * D(GammaHat[uk], li)
        )

        + Rational(1, 2) * (
            + Gammat[li, lj, lk] * GammaHatC[uk]
            + Gammat[lj, li, lk] * GammaHatC[uk]
        )

        + gt[uk, ul] * (
            + Gammat[um, lk, li] * Gammat[lm, ll, lj]
            + Gammat[um, lk, li] * Gammat[lj, lm, ll]
            + Gammat[um, lk, lj] * Gammat[li, lm, ll]
        )
    )
)

fun_z4_constraints.split_loop()

fun_z4_constraints.add_eqn(
    Rt[li, lj],
    Rational(1, 2) * (
        D(chi, li, lj)
        - Rational(1, 2) * (1 / chi) * D(chi, li) * D(chi, lj)
        + gt[li, lj] * gt[uk, ul] * (
            D(chi, lk, ll)
            - Rational(3, 2) * (1 / chi) * D(chi, lk) * D(chi, ll)
        )
        - Gammat[uk, li, lj] * D(chi, lk)
        - gt[li, lj] * GammaHatC[uk] * D(chi, lk)
    )
)

fun_z4_constraints.split_loop()

fun_z4_constraints.add_eqn(
    R[li, lj],
    Rchi[li, lj] + Rt[li, lj]
)

fun_z4_constraints.split_loop()

# Z constraint. Eq. (2.22) of [1]
fun_z4_constraints.add_eqn(
    ZtCons[ui],
    Zt[ui]
)

# Hamiltonian constraint. Eq. (2.23) of [1] and Eq. (6) of [2]
fun_z4_constraints.add_eqn(
    HamCons,
    - Rational(1, 3) * trK**2
    + Rational(1, 2) * At[li, lj] * At[ui, uj]
    - Rational(1, 2) * gt[ui, uj] * R[li, lj]
    - par_c * Zt[ui] * D(chi, li)
    # Matter. TODO: Check this
    - 16 * pi * rho
)

# Momentum constraint Eq. (2.24) of [1] and Eq. (7) of [2]
fun_z4_constraints.add_eqn(
    MomCons[li],
    -gt[uj, uk] * (
        D(At[lk, li], lj)
        - At[ll, li] * Gammat[ul, lk, lj]
        - At[lk, ll] * Gammat[ul, li, lj]
        - Rational(3, 2) * At[li, lj] * (1 / chi) * D(chi, lk)
    )
    + Rational(2, 3) * D(trK, li)
    # Matter. TODO: Check
    - 8 * pi * chi * S[li]
)

# We will explicitly sync the monitored constraints, because they are
# written on the interior only, and It would be nice to have them available
# everywhere, for monitoring, debuging, etc
sync_monitored_constraints = ExplicitSyncBatch(
    [HamCons, MomCons, ZtCons],
    analysis_group,
    schedule_after=["constraints"],
    name="sync_z4_monitored_constraints"
)

###
# Z4 Evolution equations
###
fun_z4_rhs = cottonmouth_z4.create_function(
    "z4_rhs",
    rhs_group
)

# Eq. (2.18) of [1]
fun_z4_rhs.add_eqn(
    Rchi[li, lj],
    chi * (
        - Rational(1, 2) * gt[uk, ul] * D(gt[li, lj], lk, ll)

        + Rational(1, 2) * (
            + gt[lk, li] * D(GammaHat[uk], lj)
            + gt[lk, lj] * D(GammaHat[uk], li)
        )

        + Rational(1, 2) * (
            + Gammat[li, lj, lk] * GammaHatC[uk]
            + Gammat[lj, li, lk] * GammaHatC[uk]
        )

        + gt[uk, ul] * (
            + Gammat[um, lk, li] * Gammat[lm, ll, lj]
            + Gammat[um, lk, li] * Gammat[lj, lm, ll]
            + Gammat[um, lk, lj] * Gammat[li, lm, ll]
        )
    )
)

fun_z4_rhs.split_loop()

fun_z4_rhs.add_eqn(
    Rt[li, lj],
    Rational(1, 2) * (
        D(chi, li, lj)
        - Rational(1, 2) * (1 / chi) * D(chi, li) * D(chi, lj)
        + gt[li, lj] * gt[uk, ul] * (
            D(chi, lk, ll)
            - Rational(3, 2) * (1 / chi) * D(chi, lk) * D(chi, ll)
        )
        - Gammat[uk, li, lj] * D(chi, lk)
        - gt[li, lj] * GammaHatC[uk] * D(chi, lk)
    )
)

fun_z4_rhs.split_loop()

fun_z4_rhs.add_eqn(
    R[li, lj],
    Rchi[li, lj] + Rt[li, lj]
)

fun_z4_rhs.split_loop()

# Eq. (2.7) of [1]
fun_z4_rhs.add_eqn(
    Theta_rhs,
    evo_lapse * (
        Rational(1, 3) * trK**2
        - (1 - Rational(4, 3) * par_s) * trK * Theta
        - kappa * Theta
        - Rational(2, 3) * par_s * Theta**2
        - Rational(1, 2) * At[li, lj] * At[ui, uj]
        + Rational(1, 2) * gt[ui, uj] * R[li, lj]
    )
    + Zt[ui] * (
        evo_lapse * D(chi, li)
        - chi * D(evo_lapse, li)
    )
    # TODO: Advection
    + evo_shift[ui] * D(Theta, li)
)

# Eq. (2.8) of [1]
fun_z4_rhs.add_eqn(
    chi_rhs,
    Rational(2, 3) * chi * (
        evo_lapse * (trK + 2 * par_s * Theta)
        - D(evo_shift[ui], li)
    )
    # TODO: Advection
    + evo_shift[ui] * D(chi, li)
)

# Eq. (2.9) of [1]
fun_z4_rhs.add_eqn(
    trK_rhs,
    Rational(2, 3) * chi * (
        evo_lapse * (
            (1 - Rational(2, 3) * par_s) * trK**2
            - 2 * (1 - Rational(5, 3) * par_s) * trK * Theta
            - (3 - 4 * par_s) * kappa * Theta
            + Rational(4, 3) * par_s * Theta**2
            + par_s * At[li, lj] * At[ui, uj]
            + 2 * (1 - par_s) * par_c * Zt[ui] * D(chi, li)
        )
        + chi * GammaHatSC[ui] * D(evo_lapse, li)
        + gt[ui, uj] * (
            - chi * D(evo_lapse, li, lj)
            + Rational(1, 2) * D(chi, li) * D(evo_lapse, lj)
            + evo_lapse * (
                (1 - par_s) * R[li, lj]
                # Matter. TODO: Review this
                # + Rational(1, 2) * chi * S[li, lj]
            )
        )
    )
    # TODO: Advection
    + evo_shift[ui] * D(trK, li)
)

# Eq. (2.10) of [1]
fun_z4_rhs.add_eqn(
    GammaHat_rhs_tmp[ui],
    2 * evo_lapse * (
        Gammat[ui, lj, lk] * At[uj, uk]
        - Rational(3, 2) * At[ui, uj] * (1 / chi) * D(chi, lj)
        + gt[ui, uj] * (
            (1 - Rational(4, 3) * par_s) * D(Theta, lj)
            - Rational(2, 3) * D(trK, lj)
        )
        - par_c * Zt[ui] * (
            Rational(2, 3) * trK
            + Rational(4, 3) * par_s * Theta
            + kappa
        )
    )
    - 2 * par_c * Theta * gt[ui, uj] * D(evo_lapse, lj)
    - 2 * At[ui, uj] * D(evo_lapse, lj)
    + gt[uj, uk] * D(evo_shift[ui], lj, lk)
    + Rational(1, 3) * gt[ui, uj] * D(evo_shift[uk], lj, lk)
    - GammaHatC[uj] * D(evo_shift[ui], lj)
    + Rational(2, 3) * GammaHatC[ui] * D(evo_shift[uj], lj)
    # TODO: Advection
    + evo_shift[uj] * D(GammaHat[ui], lj)
)

fun_z4_rhs.add_eqn(
    GammaHat_rhs[ui],
    GammaHat_rhs_tmp[ui]
)

# Eq. (2.11) of [1]
fun_z4_rhs.add_eqn(
    gt_rhs[li, lj],
    -2 * evo_lapse * At[li, lj]
    + gt[li, lk] * D(evo_shift[uk], lj)
    + gt[lj, lk] * D(evo_shift[uk], li)
    - Rational(2, 3) * gt[li, lj] * D(evo_shift[uk], lk)
    # TODO: Advection
    + evo_shift[uk] * D(gt[li, lj], lk)
)

# Eq. (2.12) of [1]
fun_z4_rhs.add_eqn(
    AtTF[li, lj],
    chi * (
        -D(evo_lapse, li, lj)
        + Gammat[uk, li, lj] * D(evo_lapse, lk)
    )
    - Rational(1, 2) * (
        D(chi, li) * D(evo_lapse, lj)
        + D(chi, lj) * D(evo_lapse, li)
    )
    + par_c * evo_lapse * Zt[uk] * (
        gt[lk, li] * D(chi, lj)
        + gt[lk, lj] * D(chi, li)
    )
    + evo_lapse * (
        R[li, lj]
        # Matter. TODO: Review
        # - chi * S[li, lj]
    )
)

fun_z4_rhs.add_eqn(
    At_rhs[li, lj],
    evo_lapse * (
        -2 * gt[uk, ul] * At[li, lk] * At[lj, ll]
        + (trK - 2 * (1 - par_s) * Theta) * At[li, lj]
    )
    + At[li, lk] * D(evo_shift[uk], lj)
    + At[lj, lk] * D(evo_shift[uk], li)
    - Rational(2, 3) * At[li, lj] * D(evo_shift[uk], lk)
    + AtTF[li, lj]
    - Rational(1, 3) * gt[li, lj] * gt[uk, ul] * AtTF[lk, ll]
    # TODO: Advection
    + evo_shift[uk] * D(At[li, lj], lk)
)

# 1 + log lapse, Eq. (2.26) of [1]
fun_z4_rhs.add_eqn(
    evo_lapse_rhs,
    - 2 * evo_lapse * (
        trK - 2 * (1 - par_s) * Theta
    )
    # TODO: Advection
    + evo_shift[ui] * D(evo_lapse, li)
)

# Hyperbolic Gamma Driver shift, Eq. (2.27) of [1]
fun_z4_rhs.add_eqn(
    evo_shift_rhs[ui],
    Rational(3, 4) * shift_B[ui]
    # TODO: Advection
    + evo_shift[uj] * D(evo_shift[ui], lj)
)

fun_z4_rhs.add_eqn(
    shift_B_rhs[ui],
    GammaHat_rhs_tmp[ui]
    # TODO: Advection
    - evo_shift[uj] * D(GammaHat[ui], lj)
    - shift_B[ui]
    # TODO: Advection
    + evo_shift[uj] * D(shift_B[ui], lj)
)

# Dissipation
fun_z4_diss = cottonmouth_z4.create_function(
    "z4_apply_dissipation",
    rhs_group,
    schedule_after=["z4_rhs"]
)

Theta_rhs_diss = cottonmouth_z4.overwrite(Theta_rhs)
fun_z4_diss.add_eqn(
    Theta_rhs_diss,
    Theta_rhs + dissipation_epsilon * (
        + div_diss(Theta, l0)
        + div_diss(Theta, l1)
        + div_diss(Theta, l2)
    )
)

chi_rhs_diss = cottonmouth_z4.overwrite(chi_rhs)
fun_z4_diss.add_eqn(
    chi_rhs_diss,
    chi_rhs + dissipation_epsilon * (
        + div_diss(chi, l0)
        + div_diss(chi, l1)
        + div_diss(chi, l2)
    )
)

trK_rhs_diss = cottonmouth_z4.overwrite(trK_rhs)
fun_z4_diss.add_eqn(
    trK_rhs_diss,
    trK_rhs + dissipation_epsilon * (
        + div_diss(trK, l0)
        + div_diss(trK, l1)
        + div_diss(trK, l2)
    )
)

GammaHat_rhs_diss = cottonmouth_z4.overwrite(GammaHat_rhs)
fun_z4_diss.add_eqn(
    GammaHat_rhs_diss[ui],
    GammaHat_rhs[ui] + dissipation_epsilon * (
        + div_diss(GammaHat[ui], l0)
        + div_diss(GammaHat[ui], l1)
        + div_diss(GammaHat[ui], l2)
    )
)

gt_rhs_diss = cottonmouth_z4.overwrite(gt_rhs)
fun_z4_diss.add_eqn(
    gt_rhs_diss[li, lj],
    gt_rhs[li, lj] + dissipation_epsilon * (
        + div_diss(gt[li, lj], l0)
        + div_diss(gt[li, lj], l1)
        + div_diss(gt[li, lj], l2)
    )
)

At_rhs_diss = cottonmouth_z4.overwrite(At_rhs)
fun_z4_diss.add_eqn(
    At_rhs_diss[li, lj],
    At_rhs[li, lj] + dissipation_epsilon * (
        + div_diss(At[li, lj], l0)
        + div_diss(At[li, lj], l1)
        + div_diss(At[li, lj], l2)
    )
)

evo_lapse_rhs_diss = cottonmouth_z4.overwrite(evo_lapse_rhs)
fun_z4_diss.add_eqn(
    evo_lapse_rhs_diss,
    evo_lapse_rhs + dissipation_epsilon * (
        + div_diss(evo_lapse, l0)
        + div_diss(evo_lapse, l1)
        + div_diss(evo_lapse, l2)
    )
)

evo_shift_rhs_diss = cottonmouth_z4.overwrite(evo_shift_rhs)
fun_z4_diss.add_eqn(
    evo_shift_rhs_diss[ui],
    evo_shift_rhs[ui] + dissipation_epsilon * (
        + div_diss(evo_shift[ui], l0)
        + div_diss(evo_shift[ui], l1)
        + div_diss(evo_shift[ui], l2)
    )
)

shift_B_rhs_diss = cottonmouth_z4.overwrite(shift_B_rhs)
fun_z4_diss.add_eqn(
    shift_B_rhs_diss[ui],
    shift_B_rhs[ui] + dissipation_epsilon * (
        + div_diss(shift_B[ui], l0)
        + div_diss(shift_B[ui], l1)
        + div_diss(shift_B[ui], l2)
    )
)

###
# Bake the cake
###
cottonmouth_z4.bake(
    do_cse=True,
    temporary_promotion_strategy=promote_none(),
    do_madd=False,
    do_recycle_temporaries=True,
    do_split_output_eqns=False,  # NOTE: This is broken, never turn on
    cse_optimization_level=CseOptimizationLevel.Fast
)

###
# Thorn creation
###
CppCarpetXWizard(
    cottonmouth_z4,
    CppCarpetXGenerator(
        cottonmouth_z4,
        sync_mode=SyncMode.EmulatePresync,
        interior_sync_schedule_target=post_step_group,
        extra_schedule_blocks=[
            initial_group,
            post_step_group,
            rhs_group,
            analysis_group,
        ],
        explicit_syncs=[
            sync_monitored_constraints
        ]
    )
).generate_thorn()

# References
# [1] https://arxiv.org/pdf/1810.12346
# [2] https://arxiv.org/pdf/0912.2920
# [3] https://arxiv.org/pdf/1212.2901 (typo in constraints)
