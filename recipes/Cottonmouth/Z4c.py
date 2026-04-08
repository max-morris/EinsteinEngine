#  Copyright (C) 2024-2026 Lucas T. Sanches, Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
#
#  This file is part of the Einstein Engine (EinsteinEngine).
#
#  EinsteinEngine is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  EinsteinEngine is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import functools

from sympy import Rational

from EinsteinEngine import *

###
# Thorn definitions
###
cottonmouth_Z4c = ThornDef("Cottonmouth", "CottonmouthZ4c")

###
# Some more indices
###
ul, ll = mk_pair("l")
um, lm = mk_pair("m")

###
# Finite difference stencils
###

# Fourth order centered
cottonmouth_Z4c.set_derivative_stencil(5)

# Fifth order Kreiss-Oliger disspation stencil
div_diss = cottonmouth_Z4c.mk_stencil(
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
def_max = cottonmouth_Z4c.decl_fun("max", args=2, is_stencil=False)

###
# Thorn parameters
###
chi_floor = cottonmouth_Z4c.add_param(
    "chi_floor",
    default=1.0e-6,
    desc="Chi will never be smaller than this value"
)

evolved_lapse_floor = cottonmouth_Z4c.add_param(
    "evolved_lapse_floor",
    default=1.0e-8,
    desc="The evolved lapse will never be smaller than this value"
)

dissipation_epsilon = cottonmouth_Z4c.add_param(
    "dissipation_epsilon",
    default=0.32,
    desc="The ammount of dissipation to add."
)

eta_beta = cottonmouth_Z4c.add_param(
    "eta_beta",
    default=2.0,
    desc="Standard Gamma driver eta gauge parameter. Must be of order 2 / M_ADM"
)

# See Refs. [1,2] for the default
kappa_1 = cottonmouth_Z4c.add_param(
    "kappa_1",
    default=0.02,
    desc="Constraint damping parameter kappa_1. Must be of order 1 / L wehre L is the typical simulation scale."
)

# See Refs. [1,2] for the default
kappa_2 = cottonmouth_Z4c.add_param(
    "kappa_2",
    default=0.0,
    desc="Constraint damping parameter kappa_2."
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
g = cottonmouth_Z4c.decl(
    "g",
    [li, lj],
    symmetries=[(li, lj)],
    from_thorn="ADMBaseX"
)

k = cottonmouth_Z4c.decl(
    "k",
    [li, lj],
    symmetries=[(li, lj)],
    from_thorn="ADMBaseX"
)

alp = cottonmouth_Z4c.decl(
    "alp",
    [],
    from_thorn="ADMBaseX"
)

beta = cottonmouth_Z4c.decl(
    "beta",
    [ui],
    from_thorn="ADMBaseX"
)

###
# TmunuBaseX vars.
###
eTtt = cottonmouth_Z4c.decl(
    "eTtt",
    [],
    from_thorn="TmunuBaseX"
)

eTti = cottonmouth_Z4c.decl(
    "eTt",
    [li],
    from_thorn="TmunuBaseX"
)

eTij = cottonmouth_Z4c.decl(
    "eT",
    [li, lj],
    symmetries=[(li, lj)],
    from_thorn="TmunuBaseX"
)

###
# Evolved Gauge Vars.
###
evo_lapse_rhs = cottonmouth_Z4c.decl(
    "evo_lapse_rhs",
    [],
    parity=parity_scalar
)

evo_lapse = cottonmouth_Z4c.decl(
    "evo_lapse",
    [],
    rhs=evo_lapse_rhs,
    parity=parity_scalar
)

evo_shift_rhs = cottonmouth_Z4c.decl(
    "evo_shift_rhs",
    [ui],
    parity=parity_vector
)

evo_shift = cottonmouth_Z4c.decl(
    "evo_shift",
    [ui],
    rhs=evo_shift_rhs,
    parity=parity_vector
)

###
# Evolved Z4 vars.
###
# \Theta
Theta_rhs = cottonmouth_Z4c.decl(
    "Theta_rhs",
    [],
    parity=parity_scalar
)

Theta = cottonmouth_Z4c.decl(
    "Theta",
    [],
    rhs=Theta_rhs,
    parity=parity_scalar
)

# \chi
chi_rhs = cottonmouth_Z4c.decl(
    "chi_rhs",
    [],
    parity=parity_scalar
)

chi = cottonmouth_Z4c.decl(
    "chi",
    [],
    rhs=chi_rhs,
    parity=parity_scalar
)

# K (trace of Extrinsic Curvature)
trK_rhs = cottonmouth_Z4c.decl(
    "trK_rhs",
    [],
    parity=parity_scalar
)

trK = cottonmouth_Z4c.decl(
    "trK",
    [],
    rhs=trK_rhs,
    parity=parity_scalar
)

# Evolved \tilde{\Gamma}^i
evo_Gammat_rhs = cottonmouth_Z4c.decl(
    "evo_Gammat_rhs",
    [ui],
    parity=parity_vector
)

evo_Gammat = cottonmouth_Z4c.decl(
    "evo_Gammat",
    [ui],
    rhs=evo_Gammat_rhs,
    parity=parity_vector
)

# \tilde{\gamma_{i j}}
gt_rhs = cottonmouth_Z4c.decl(
    "gt_rhs",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

gt = cottonmouth_Z4c.decl(
    "gt",
    [li, lj],
    symmetries=[(li, lj)],
    rhs=gt_rhs,
    parity=parity_sym2ten
)

# \tilde{A}_{ij}
At_rhs = cottonmouth_Z4c.decl(
    "At_rhs",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

At = cottonmouth_Z4c.decl(
    "At",
    [li, lj],
    symmetries=[(li, lj)],
    rhs=At_rhs,
    parity=parity_sym2ten
)

###
# Monitored constraint Vars.
###
HamCons = cottonmouth_Z4c.decl(
    "HamCons",
    [],
    parity=parity_scalar
)

MomCons = cottonmouth_Z4c.decl(
    "MomCons",
    [ui],
    parity=parity_vector
)

ZtCons = cottonmouth_Z4c.decl(
    "ZtCons",
    [ui], parity=parity_vector
)

###
# Ricci tensor.
###
R = cottonmouth_Z4c.decl(
    "R",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

Rchi = cottonmouth_Z4c.decl(
    "Rchi",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

Rt = cottonmouth_Z4c.decl(
    "Rt",
    [li, lj],
    symmetries=[(li, lj)],
    parity=parity_sym2ten
)

###
# Matter terms.
###

#  n_a n_b T^{ab}
rho = cottonmouth_Z4c.decl(
    "rho",
    []
)

# -\gamma_{ia} n_b T^{ab}
Svec = cottonmouth_Z4c.decl(
    "Svec",
    [li]
)

# \gamma^{ia} \gamma_{jb} T^{ab}
S = cottonmouth_Z4c.decl(
    "S",
    [li, lj],
    symmetries=[(li, lj)],
)

trS = cottonmouth_Z4c.decl(
    "trS",
    [],
)

###
# Aux. Vars.
###
# \tilde{\Gamma}_{ijk}
Gammat = cottonmouth_Z4c.decl(
    "Gammat",
    [li, lj, lk],
    symmetries=[(lj, lk)]
)

# Gammatd = \tilde{\gamma}^{jk}\tilde{Gamma}^i_{jk}
Gammatd = cottonmouth_Z4c.decl("Gammatd", [ui])

# covd2_alpha = D_i D_j \alpha
covd2_alpha = cottonmouth_Z4c.decl(
    "covd2_alpha",
    [li, lj],
    symmetries=[(li, lj)]
)

#  covdt2_chi = \tilde{D}_i \tilde{D}_j \chi
covdt2_chi = cottonmouth_Z4c.decl(
    "covdt2_chi",
    [li, lj],
    symmetries=[(li, lj)]
)

# Trace free symbol in the evolution for \tilde{A}_{ij}
AtTF = cottonmouth_Z4c.decl(
    "AtTF",
    [li, lj],
    symmetries=[(li, lj)]
)

###
# Substitution rules
###
# Physical metric and its inverse
g_mat = cottonmouth_Z4c.get_matrix(g[li, lj])
g_imat = inv(g_mat)
detg = det(g_mat)
cottonmouth_Z4c.add_substitution_rule(g[ui, uj], g_imat)

# Conformal metric and its inverse
gt_mat = cottonmouth_Z4c.get_matrix(gt[li, lj])
detgt = det(gt_mat)

# Use the fact that det(gt) = 1 to simplify the inverse expression
# Note that det(gt) = 1 is an *enforced* constraint
gt_imat = inv(gt_mat) * detgt
cottonmouth_Z4c.add_substitution_rule(gt[ui, uj], gt_imat)

# \tilde{\Gamma}_{ijk}. Eq. (2.16) of [1]
cottonmouth_Z4c.add_substitution_rule(
    Gammat[lk, li, lj],
    Rational(1, 2) * (
        D(gt[lj, lk], li) + D(gt[li, lk], lj) - D(gt[li, lj], lk)
    )
)

# \tilde{\Gamma}^i_{jk}. Eq. (2.14) of [1]
cottonmouth_Z4c.add_substitution_rule(
    Gammat[uk, li, lj], gt[uk, ul] * Gammat[ll, li, lj]
)

cottonmouth_Z4c.add_substitution_rule(
    Gammatd[ui], gt[uj, uk] * Gammat[ui, lj, lk]
)

# At
cottonmouth_Z4c.add_substitution_rule(At[ui, lj], gt[ui, uk] * At[lk, lj])
cottonmouth_Z4c.add_substitution_rule(At[ui, uj], gt[uj, uk] * At[ui, lk])

# Matter term definitions.
cottonmouth_Z4c.add_substitution_rule(
    rho,
    1 / evo_lapse**2 * (
        eTtt - 2 * evo_shift[ui] * eTti[li]
        + evo_shift[ui] * evo_shift[uj] * eTij[li, lj]
    )
)

cottonmouth_Z4c.add_substitution_rule(
    Svec[li],
    -1/evo_lapse * (eTti[li] - evo_shift[uj] * eTij[li, lj])
)

cottonmouth_Z4c.add_substitution_rule(
    S[li, lj],
    eTij[li, lj]
)

cottonmouth_Z4c.add_substitution_rule(
    trS,
    chi * gt[ua, ub] * eTij[la, lb]
)

# Covariant derivatives with respect to the physical metric
# TODO: Check
cottonmouth_Z4c.add_substitution_rule(
    covd2_alpha[lj, lk],
    D(evo_lapse, lj, lk)
    - Gammat[ui, lj, lk] * D(evo_lapse, li)
    + Rational(1, 2) * (1 / chi) * (
        + D(evo_lapse, lj) * D(chi, lk)
        + D(evo_lapse, lk) * D(chi, lj)
        - gt[ul, um] * gt[lj, lk] * D(evo_lapse, ll) * D(chi, lm)
    )
)

cottonmouth_Z4c.add_substitution_rule(
    covdt2_chi[li, lj],
    D(chi, lj, li)
    - Gammat[uk, li, lj] * D(chi, lk)
)

###
# Aux. groups
###

# Initialization
initial_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthZ4c_InitialGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_Initial"),
    after=[Identifier("ADMBaseX_PostInitial")],
    description=String("Z4 initialization routines")
)

# Post-step
post_step_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthZ4c_PostStepGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_PostStep"),
    before=[Identifier("ADMBaseX_SetADMVars")],
    description=String("Z4 post-step routines")
)

# RHS
rhs_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthZ4c_RHSGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_RHS"),
    description=String("Z4 equations RHS computation"),
)

# Analysis
analysis_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthZ4c_AnalysisGroup"),
    at_or_in=AtOrIn.At,
    schedule_bin=Identifier("analysis"),
    description=String("Z4 analysis routines"),
)

###
# Convert ADM to Z4 variables
#
###
fun_adm_to_z4c_pt1 = cottonmouth_Z4c.create_function(
    "adm_to_z4c_pt1",
    initial_group,
    intent_override=IntentOverride.WriteInterior
)

fun_adm_to_z4c_pt1.add_eqn(
    Theta,
    sympify(0)
)

# Eq. (11) of [2], right
fun_adm_to_z4c_pt1.add_eqn(
    chi,
    1 / (cbrt(detg))
)

# Eq. (12) of [2], left
fun_adm_to_z4c_pt1.add_eqn(
    trK,
    g[ui, uj] * k[li, lj]
)

# Eq. (11) of [2], left
fun_adm_to_z4c_pt1.add_eqn(
    gt[li, lj],
    (1 / cbrt(detg)) * g[li, lj]
)

# Eq. (12) of [2], right
fun_adm_to_z4c_pt1.add_eqn(
    At[li, lj],
    (1 / cbrt(detg)) * (
        k[li, lj]
        - Rational(1, 3) * g[li, lj] * g[uk, ul] * k[lk, ll]
    )
)

fun_adm_to_z4c_pt1.add_eqn(
    evo_lapse,
    alp
)

fun_adm_to_z4c_pt1.add_eqn(
    evo_shift[ui],
    beta[ui]
)

fun_adm_to_z4c_pt2 = cottonmouth_Z4c.create_function(
    "adm_to_z4c_pt2",
    initial_group,
    schedule_after=["adm_to_z4c_pt1"]
)

# Eq. (2.6) of [1]
fun_adm_to_z4c_pt2.add_eqn(
    evo_Gammat[ui],
    Gammatd[ui]
)

###
# Enforce algebraic constraints
###
fun_z4c_enforce_pt1 = cottonmouth_Z4c.create_function(
    "z4c_enforce_pt1",
    post_step_group
)

# Enforce chi floor
chi_enforce = cottonmouth_Z4c.overwrite(chi)

fun_z4c_enforce_pt1.add_eqn(
    chi_enforce,
    def_max(chi, chi_floor)
)

# Enforce lapse floor
evo_lapse_enforce = cottonmouth_Z4c.overwrite(evo_lapse)

fun_z4c_enforce_pt1.add_eqn(
    evo_lapse_enforce,
    def_max(evo_lapse, evolved_lapse_floor)
)

# Enforce \det(\tilde{\gamma}) = 1
gt_enforce = cottonmouth_Z4c.overwrite(gt)

fun_z4c_enforce_pt1.add_eqn(
    gt_enforce[li, lj],
    gt[li, lj] / (cbrt(detgt))
)

fun_z4c_enforce_pt2 = cottonmouth_Z4c.create_function(
    "z4c_enforce_pt2",
    post_step_group,
    schedule_after=["z4c_enforce_pt1"]
)

# Enforce \tilde{\gamma}^{i j} \tilde{A}_{ij} = 0 (A)
At_enforce = cottonmouth_Z4c.overwrite(At)

fun_z4c_enforce_pt2.add_eqn(
    At_enforce[li, lj],
    At[li, lj] - Rational(1, 3) * gt[li, lj] * gt[uk, ul] * At[lk, ll]
)

###
# Convert Z4 to ADM variables
###
fun_z4c_to_adm = cottonmouth_Z4c.create_function(
    "z4c_to_adm",
    post_step_group,
    schedule_after=["z4c_enforce_pt2"],
    intent_override=IntentOverride.E2E
)

# Eq. (2.4) of [1]
fun_z4c_to_adm.add_eqn(
    g[li, lj],
    (1 / chi) * gt[li, lj]
)

# Eq. (2.5) of [1]
fun_z4c_to_adm.add_eqn(
    k[li, lj],
    (1 / chi) * (
        At[li, lj]
        + Rational(1, 3) * gt[li, lj] * trK
    )
)

fun_z4c_to_adm.add_eqn(
    alp,
    evo_lapse
)

fun_z4c_to_adm.add_eqn(
    beta[ui],
    evo_shift[ui]
)

###
# Compute monitored constraints
###
fun_z4c_constraints = cottonmouth_Z4c.create_function(
    "z4c_constraints",
    analysis_group
)

# Eq (8) of [1]
fun_z4c_constraints.add_eqn(
    Rchi[li, lj],
    + Rational(1, 2) * (1 / chi) * covdt2_chi[li, lj]
    + Rational(1, 2) * (1 / chi) * gt[li, lj] * gt[uk, ul] * covdt2_chi[lk, ll]
    - Rational(1, 4) * (1 / (chi**2)) * D(chi, li) * D(chi, lj)
    - Rational(3, 4) * (1 / (chi**2)) *
    gt[li, lj] * gt[ul, uk] * D(chi, lk) * D(chi, ll)
)

fun_z4c_constraints.split_loop()

# Eq (9) of [1]
fun_z4c_constraints.add_eqn(
    Rt[li, lj],
    - Rational(1, 2) * gt[ul, um] * D(gt[li, lj], ll, lm)
    + Rational(1, 2) * (
        + gt[lk, li] * D(evo_Gammat[uk], lj)
        + gt[lk, lj] * D(evo_Gammat[uk], li)
    )
    + Rational(1, 2) * (
        + Gammatd[uk] * Gammat[li, lj, lk]
        + Gammatd[uk] * Gammat[lj, li, lk]
    )
    + gt[ul, um] * Gammat[uk, ll, li] * Gammat[lj, lk, lm]
    + gt[ul, um] * Gammat[uk, ll, lj] * Gammat[li, lk, lm]
    + gt[ul, um] * Gammat[uk, li, lm] * Gammat[lk, ll, lj]
)

fun_z4c_constraints.split_loop()

fun_z4c_constraints.add_eqn(
    R[li, lj],
    Rchi[li, lj] + Rt[li, lj]
)

# Eq. (13) of [1]
fun_z4c_constraints.add_eqn(
    ZtCons[ui],
    + Rational(1, 2) * (
        + evo_Gammat[ui]
        - Gammatd[ui]
    )
)

# Eq. (14) of [1] with corrections from Eq. (23) of [2]
fun_z4c_constraints.add_eqn(
    HamCons,
    chi * gt[ui, uj] * R[li, lj]
    - At[li, lj] * At[ui, uj]
    + Rational(2, 3) * (trK + 2 * Theta)**2
    # Matter.
    - 16 * pi * rho
)

# Eq. (15) of [1] with corrections from Eq. (24) of [2]
fun_z4c_constraints.add_eqn(
    MomCons[ui],
    D(At[ui, uj], lj)
    + Gammat[ui, lj, lk] * At[uj, uk]
    - Rational(2, 3) * gt[ui, uj] * D(trK, lj)
    - Rational(4, 3) * gt[ui, uj] * D(Theta, lj)
    - Rational(3, 2) * At[ui, uj] * (1 / chi) * D(chi, lj)
    # Matter.
    - 8 * pi * chi * gt[ui, uj] * Svec[lj]
)

# We will explicitly sync the monitored constraints, because they are
# written on the interior only, and It would be nice to have them available
# everywhere, for monitoring, debuging, etc
sync_monitored_constraints = ExplicitSyncBatch(
    [HamCons, MomCons, ZtCons],
    analysis_group,
    schedule_after=["z4c_constraints"],
    name="sync_z4_monitored_constraints"
)

###
# Z4 Evolution equations
###
fun_z4c_rhs = cottonmouth_Z4c.create_function(
    "z4c_rhs",
    rhs_group
)

# Eq (8) of [1]
fun_z4c_rhs.add_eqn(
    Rchi[li, lj],
    + Rational(1, 2) * (1 / chi) * covdt2_chi[li, lj]
    + Rational(1, 2) * (1 / chi) * gt[li, lj] * gt[uk, ul] * covdt2_chi[lk, ll]
    - Rational(1, 4) * (1 / (chi**2)) * D(chi, li) * D(chi, lj)
    - Rational(3, 4) * (1 / (chi**2)) *
    gt[li, lj] * gt[ul, uk] * D(chi, lk) * D(chi, ll)
)

fun_z4c_rhs.split_loop()

# Eq (9) of [1]
fun_z4c_rhs.add_eqn(
    Rt[li, lj],
    - Rational(1, 2) * gt[ul, um] * D(gt[li, lj], ll, lm)
    + Rational(1, 2) * (
        + gt[lk, li] * D(evo_Gammat[uk], lj)
        + gt[lk, lj] * D(evo_Gammat[uk], li)
    )
    + Rational(1, 2) * (
        + Gammatd[uk] * Gammat[li, lj, lk]
        + Gammatd[uk] * Gammat[lj, li, lk]
    )
    + gt[ul, um] * Gammat[uk, ll, li] * Gammat[lj, lk, lm]
    + gt[ul, um] * Gammat[uk, ll, lj] * Gammat[li, lk, lm]
    + gt[ul, um] * Gammat[uk, li, lm] * Gammat[lk, ll, lj]
)

fun_z4c_rhs.split_loop()

fun_z4c_rhs.add_eqn(
    R[li, lj],
    Rchi[li, lj] + Rt[li, lj]
)

# Eq. (6) of [1]
fun_z4c_rhs.add_eqn(
    Theta_rhs,
    Rational(1, 2) * evo_lapse * (
        + chi * gt[ui, uj] * R[li, lj]
        - At[li, lj] * At[ui, uj]
        + Rational(2, 3) * (trK + 2 * Theta)**2
    )
    # Damping
    - evo_lapse * kappa_1 * (2 + kappa_2) * Theta
    # Advection
    + evo_shift[ui] * D(Theta, li)
    # Matter
    - evo_lapse * 8 * pi * rho
)

# Eq. (1) of [1]
fun_z4c_rhs.add_eqn(
    chi_rhs,
    + Rational(2, 3) * chi * (
        + evo_lapse * (
            trK + 2 * Theta
        )
        - D(evo_shift[ui], li)
    )
    # Advection
    + evo_shift[ui] * D(chi, li)
)

# Eq. (3) of [1]
fun_z4c_rhs.add_eqn(
    trK_rhs,
    - chi * gt[ui, uj] * covd2_alpha[li, lj]
    + evo_lapse * (
        At[li, lj] * At[ui, uj]
        + Rational(1, 3) * (trK + 2 * Theta)**2
    )
    # Damping
    + evo_lapse * kappa_1 * (1 - kappa_2) * Theta
    # Advection
    + evo_shift[ui] * D(trK, li)
    # Matter
    + 4 * pi * evo_lapse * (trS + rho)
)

# Eq. (5) of [1]
fun_z4c_rhs.add_eqn(
    evo_Gammat_rhs[ui],
    - 2 * At[ui, uj] * D(evo_lapse, lj)
    + 2 * evo_lapse * (
        + Gammat[ui, lj, lk] * At[uj, uk]
        - Rational(3, 2) * At[ui, uj] * (1 / chi) * D(chi, lj)
        - Rational(2, 3) * gt[ui, uj] * D(trK, lj)
        - Rational(4, 3) * gt[ui, uj] * D(Theta, lj)
    )
    + gt[uj, uk] * D(evo_shift[ui], lj, lk)
    + Rational(1, 3) * gt[ui, uj] * D(evo_shift[uk], lj, lk)
    - Gammatd[uj] * D(evo_shift[ui], lj)
    + Rational(2, 3) * Gammatd[ui] * D(evo_shift[uj], lj)
    # Damping
    - evo_lapse * kappa_1 * (evo_Gammat[ui] - Gammatd[ui])
    # Advection
    + evo_shift[uj] * D(evo_Gammat[ui], lj)
    # Matter
    - 16 * pi * evo_lapse * chi * gt[ui, uj] * Svec[lj]
)

# Eq. (2) of [1]
fun_z4c_rhs.add_eqn(
    gt_rhs[li, lj],
    - 2 * evo_lapse * At[li, lj]
    + gt[lk, li] * D(evo_shift[uk], lj)
    + gt[lk, lj] * D(evo_shift[uk], li)
    - Rational(2, 3) * gt[li, lj] * D(evo_shift[uk], lk)
    # Advection
    + evo_shift[uk] * D(gt[li, lj], lk)
)

# Eq. (4) of [1]
fun_z4c_rhs.add_eqn(
    AtTF[li, lj],
    - covd2_alpha[li, lj]
    + evo_lapse * (
        + R[li, lj]
        # Matter
        - 8 * pi * S[li, lj]
    )
)

fun_z4c_rhs.add_eqn(
    At_rhs[li, lj],
    chi * (
        + AtTF[li, lj]
        - Rational(1, 3) * gt[li, lj] * gt[uk, ul] * AtTF[lk, ll]
    )
    + evo_lapse * (
        + (trK + 2 * Theta) * At[li, lj]
        - 2 * At[uk, li] * At[lk, lj]
    )
    + At[lk, li] * D(evo_shift[uk], lj)
    + At[lk, lj] * D(evo_shift[uk], li)
    - Rational(2, 3) * At[li, lj] * D(evo_shift[uk], lk)
    # Advection
    + evo_shift[uk] * D(At[li, lj], lk)
)

# Eq. (11) of [1]
fun_z4c_rhs.add_eqn(
    evo_lapse_rhs,
    - 2 * evo_lapse * trK
    # Advection
    + evo_shift[ui] * D(evo_lapse, li)
)

# Eq. (12) of [1]
fun_z4c_rhs.add_eqn(
    evo_shift_rhs[ui],
    + evo_Gammat[ui]
    - eta_beta * evo_shift[ui]
    # Advection
    + evo_shift[uj] * D(evo_shift[ui], lj)
)

# Dissipation
fun_z4c_diss = cottonmouth_Z4c.create_function(
    "z4c_apply_dissipation",
    rhs_group,
    schedule_after=["z4c_rhs"]
)

Theta_rhs_diss = cottonmouth_Z4c.overwrite(Theta_rhs)
fun_z4c_diss.add_eqn(
    Theta_rhs_diss,
    Theta_rhs + dissipation_epsilon * (
        + div_diss(Theta, l0)
        + div_diss(Theta, l1)
        + div_diss(Theta, l2)
    )
)

chi_rhs_diss = cottonmouth_Z4c.overwrite(chi_rhs)
fun_z4c_diss.add_eqn(
    chi_rhs_diss,
    chi_rhs + dissipation_epsilon * (
        + div_diss(chi, l0)
        + div_diss(chi, l1)
        + div_diss(chi, l2)
    )
)

trK_rhs_diss = cottonmouth_Z4c.overwrite(trK_rhs)
fun_z4c_diss.add_eqn(
    trK_rhs_diss,
    trK_rhs + dissipation_epsilon * (
        + div_diss(trK, l0)
        + div_diss(trK, l1)
        + div_diss(trK, l2)
    )
)

GammaHat_rhs_diss = cottonmouth_Z4c.overwrite(evo_Gammat_rhs)
fun_z4c_diss.add_eqn(
    GammaHat_rhs_diss[ui],
    evo_Gammat_rhs[ui] + dissipation_epsilon * (
        + div_diss(evo_Gammat[ui], l0)
        + div_diss(evo_Gammat[ui], l1)
        + div_diss(evo_Gammat[ui], l2)
    )
)

gt_rhs_diss = cottonmouth_Z4c.overwrite(gt_rhs)
fun_z4c_diss.add_eqn(
    gt_rhs_diss[li, lj],
    gt_rhs[li, lj] + dissipation_epsilon * (
        + div_diss(gt[li, lj], l0)
        + div_diss(gt[li, lj], l1)
        + div_diss(gt[li, lj], l2)
    )
)

At_rhs_diss = cottonmouth_Z4c.overwrite(At_rhs)
fun_z4c_diss.add_eqn(
    At_rhs_diss[li, lj],
    At_rhs[li, lj] + dissipation_epsilon * (
        + div_diss(At[li, lj], l0)
        + div_diss(At[li, lj], l1)
        + div_diss(At[li, lj], l2)
    )
)

evo_lapse_rhs_diss = cottonmouth_Z4c.overwrite(evo_lapse_rhs)
fun_z4c_diss.add_eqn(
    evo_lapse_rhs_diss,
    evo_lapse_rhs + dissipation_epsilon * (
        + div_diss(evo_lapse, l0)
        + div_diss(evo_lapse, l1)
        + div_diss(evo_lapse, l2)
    )
)

evo_shift_rhs_diss = cottonmouth_Z4c.overwrite(evo_shift_rhs)
fun_z4c_diss.add_eqn(
    evo_shift_rhs_diss[ui],
    evo_shift_rhs[ui] + dissipation_epsilon * (
        + div_diss(evo_shift[ui], l0)
        + div_diss(evo_shift[ui], l1)
        + div_diss(evo_shift[ui], l2)
    )
)

###
# Bake the cake
###
cottonmouth_Z4c.bake(
    do_cse=True,
    temporary_promotion_strategy=promote_none(),
    do_madd=False,
    do_recycle_temporaries=True,
    do_split_output_eqns=False,  # NOTE: This is broken, never turn on
    cse_optimization_level=CseOptimizationLevel.Fast,
    ordering_fn=functools.partial(prioritize_rare_symbols, consider_frequency=True, complexity_factor=0.0)
)

###
# Thorn creation
###
CppCarpetXWizard(
    cottonmouth_Z4c,
    CppCarpetXGenerator(
        cottonmouth_Z4c,
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
# [1] https://arxiv.org/pdf/1212.2901 (typo in constraints, refer to [2])
# [2] https://arxiv.org/pdf/0912.2920
