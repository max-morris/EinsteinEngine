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

# Modified copy of Z4c.py: the 8 Lie-derivative ("Advection") terms (shift
# vector dotted into a spatial derivative -- one each for Theta, chi, trK,
# evo_Gammat, gt, At, evo_lapse, and evo_shift itself) use an upwinded
# derivative (Dupwind, defined below) instead of the centered stencil `D`
# used for every other derivative in this recipe.
#
# This matches STvAR's own advection scheme: STvAR selects between a
# forward-biased stencil (its dupD* variables, offset=+1) and a
# backward-biased stencil (ddnD*, offset=-1) based on the sign of the shift
# component in that direction (`dupDfoo*(beta_U_i > 0) + ddnDfoo*(beta_U_i
# < 0)` in STvAR's ET_Integration_Rhs_K.H) -- done here via the same
# `h_step`-based one-sided-stencil-selection recipe already used in
# recipes/osdiv/osdiv.py, rather than STvAR's boolean-multiply form; the
# two are mathematically equivalent.
#
# Produces a distinctly-named thorn (CottonmouthZ4cUpwind<suffix> instead
# of CottonmouthZ4c<suffix>) so it can be checked out/built alongside the
# original for a side-by-side comparison.

import argparse
import functools
import sys
from pathlib import Path

from sympy import Rational

from EinsteinEngine import *

################################
# BEGIN Generate Options
###


###
# Finite difference stencils
###

parser = argparse.ArgumentParser(prog='Cottonmouth Z4c (upwinded)', description='A code generator for the Z4c equations, using upwinded advection derivatives')
parser.add_argument('--vacuum', action='store_true', default=False, help='Whether to generate matter terms.')
parser.add_argument('--fd-order', type=int, default=4, help='Order of the finite difference equations to use.')
parser.add_argument(
    '--no-atij-backreaction', action='store_true', default=False,
    help="Drop the matter (-8*pi*S_ij) term from AtTF/At_rhs specifically, while keeping "
         "every other matter term (rho in Theta_rhs/HamCons, trS+rho in trK_rhs, Svec in "
         "evo_Gammat_rhs/MomCons) active. Matches Simflowny/MHDuet's mhduetBS-2.0 CCZ4 "
         "implementation exactly -- confirmed by direct inspection of its Atd_ij evolution "
         "equation, which has no scalar-field backreaction term at all, unlike this recipe's "
         "normal (Z4c.py-inherited) AtTF, which does include -8*pi*S_ij whenever matter terms "
         "are on. See boson_star.py's reproduction test against mhduetBS-2.0's example_BS: "
         "feeding MHDuet's own solved equilibrium profile into the *normal* (S_ij-including) "
         "variant produces genuine, fast growth in |phi|^2/trK, because that profile is only "
         "a valid equilibrium for equations that omit this term -- this flag lets that "
         "hypothesis be tested directly."
)
parser.add_argument(
    '--mhduet-theta-rhs-convention', action='store_true', default=False,
    help="Theta_rhs's quadratic trK/Theta source term: 1 selects "
         "mhduetBS-2.0's own CCZ4 form (2/3)*trK^2 + (2/3)*Theta*(trK - "
         "2*Theta); 0 (default) selects the literature-standard Z4c form "
         "(2/3)*(trK + 2*Theta)^2 this recipe cited as 'Eq (6) of [1]'. "
         "These are NOT algebraically equivalent (confirmed by direct "
         "MathML-equation extraction from mhduetBS-2.0's own "
         "documentation/PDEModel-*.xml): expanding MHDuet's form gives a "
         "(2/3) trK*Theta cross term and a -(4/3) Theta^2 term, versus the "
         "standard form's (8/3) trK*Theta and +(8/3) Theta^2 -- differing "
         "in both magnitude and sign. At_rhs's trK*At_ij term (which uses "
         "bare trK, not trK+2*Theta, always -- see mhduetBS-2.0's own "
         "Atd_ij RHS, which has no 2*Theta enhancement at all) is fixed "
         "unconditionally to match MHDuet regardless of this flag, since "
         "that specific discrepancy was unambiguous."
)
pres=parser.parse_args(sys.argv[1:])

stencil_order = pres.fd_order
use_matter_terms = 0 if pres.vacuum else 1
use_atij_matter = 0 if (pres.vacuum or pres.no_atij_backreaction) else 1
mhduet_theta_rhs_convention = 1 if pres.mhduet_theta_rhs_convention else 0

suffix = (
    f"{stencil_order}{'v' if pres.vacuum else 'm'}"
    + ("NoBR" if pres.no_atij_backreaction else "")
    # NOTE: --mhduet-theta-rhs-convention deliberately does NOT get a name
    # suffix (unlike --no-atij-backreaction's "NoBR") -- it's a pure
    # generation-time codegen switch on an existing thorn, regenerated in
    # place like every other fix this session, not a new coexisting
    # variant. Giving it a distinct name would also require regenerating
    # boson_star.py with a matching --metric-thorn and updating every
    # parfile's ActiveThorns/thorn-prefixed params, well beyond the scope
    # of this specific convention test.
)

###
# END Generate Options
################################

###
# Thorn definitions
#
# Named distinctly from Z4c.py's CottonmouthZ4c<suffix> so both thorns can
# coexist in the same Cactus tree/ThornList without colliding.
###
cottonmouth_Z4c = ThornDef("Cottonmouth", f"CottonmouthZ4cUpwind{suffix}", derivative_stencil_width=stencil_order + 1)

###
# Some more indices
###
ul, ll = cottonmouth_Z4c.mk_pair("l")
um, lm = cottonmouth_Z4c.mk_pair("m")

# Fifth order Kreiss-Oliger disspation stencil
div_diss = cottonmouth_Z4c.mk_stencil(
    "div_diss",
    li,
    kreiss_oliger_stencil(stencil_order+1, li)
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

# CCZ4-style continuous algebraic-constraint damping (Alic et al. 2011,
# "Conformal and covariant formulation of the Z4 system with
# constraint-violation damping"). Plain Z4c enforces det(gt)=1 and
# tr(At)=0 only via a HARD post-step algebraic reprojection
# (z4c_enforce_pt1/pt2, above); CCZ4 additionally adds a continuous
# exponential-relaxation force directly into gt_rhs/At_rhs that actively
# pulls det(gt)/tr(At) back toward their constraint values every RHS
# evaluation, not just once per step after the fact. Confirmed missing
# here via direct comparison against mhduetBS-2.0's own compiled RHS
# (AdvanceLevel.cpp: d_gtd_xx_o2_..._l0 includes a
# "-1/3*kappa_cc*alpha*gtd_xx*log(det(gtd))" term absent from this
# thorn's gt_rhs, and d_Atd_xx_o2_..._l0 includes an analogous
# "-1/3*kappa_cc*alpha*gtd_xx*tr(Atd)" term absent from At_rhs) --
# mhduetBS-2.0's own example_BS run uses problem.p_kappa_cc = 1.0.
kappa_cc = cottonmouth_Z4c.add_param(
    "kappa_cc",
    default=1.0,
    desc="CCZ4-style continuous constraint-damping parameter kappa_cc, relaxing "
         "det(gt)=1 and tr(At)=0 continuously within gt_rhs/At_rhs (in addition to "
         "the existing hard post-step algebraic enforcement). Matches mhduetBS-2.0's "
         "problem.p_kappa_cc; default 1.0 per its example_BS run."
)

# Runtime on/off for the -8*pi*S_ij term in AtTF/At_rhs. The generation-time
# --no-atij-backreaction flag (which also renames the thorn to *NoBR) cannot
# coexist in the same executable as CottonmouthZ4cUpwind4m -- EinsteinEngine
# emits identical C++ symbol names -- so a parfile switch is the only way to
# drop this term in the live 4m thorn used for the MHDuet reproduction.
# mhduetBS-2.0's own Atd_ij equation has no scalar-field backreaction at all
# (confirmed via MathML extraction of atd_xx_op.xml); its solved example_BS
# profile is an equilibrium only of that S_ij-free system. Leaving this at
# the default 1.0 keeps the flop-matched 4m kernel; reproduction parfiles
# must set it to 0.0 (boson.md: do not knowingly differ from MHDuet).
include_atij_backreaction = cottonmouth_Z4c.add_param(
    "include_atij_backreaction",
    default=1.0,
    desc="Multiplier on the -8*pi*S_ij term in AtTF/At_rhs. 1 (default) keeps "
         "the standard Z4c matter backreaction; 0 drops it to match mhduetBS-2.0's "
         "own Atd_ij equation, which has no scalar-field S_ij source."
)

# Controls if NewRadX should be applied
cottonmouth_Z4c.add_param(
    "apply_NewRadX",
    default=False,
    desc="Apply NewRadX boundary conditions"
)

# NewRadX powers
radpower_Theta = cottonmouth_Z4c.add_param(
    "radpower_Theta",
    default=1.0,
    desc="NewRadX radpower for Theta"
)

radpower_chi = cottonmouth_Z4c.add_param(
    "radpower_chi",
    default=1.0,
    desc="NewRadX radpower for chi"
)

radpower_trK = cottonmouth_Z4c.add_param(
    "radpower_trK",
    default=1.0,
    desc="NewRadX radpower the trace of K_ij"
)

radpower_evo_Gammat = cottonmouth_Z4c.add_param(
    "radpower_evo_Gammat",
    default=1.0,
    desc="NewRadX radpower Gamma^i"
)

radpower_gt = cottonmouth_Z4c.add_param(
    "radpower_gt",
    default=1.0,
    desc="NewRadX radpower for gt_ij"
)

radpower_At = cottonmouth_Z4c.add_param(
    "radpower_At",
    default=1.0,
    desc="NewRadX radpower for At_ij"
)

radpower_evo_lapse = cottonmouth_Z4c.add_param(
    "radpower_evo_lapse",
    default=1.0,
    desc="NewRadX radpower for the lapse, alpha"
)

radpower_evo_shift = cottonmouth_Z4c.add_param(
    "radpower_evo_shift",
    default=1.0,
    desc="NewRadX radpower for the shift, beta^i"
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
# Upwinded derivative for the Lie-derivative ("Advection") terms below.
#
# Matches STvAR's dupD/ddnD scheme: for the direction picked out by the
# stencil's free index, use the forward-biased stencil (offset=+1) when the
# shift component in that direction is positive, and the backward-biased
# stencil (offset=-1) when it is negative -- instead of the centered
# stencil `D` used for every other derivative in this recipe. Same recipe
# as recipes/osdiv/osdiv.py, using evo_shift as the direction field.
###
kdelta = cottonmouth_Z4c.mk_kdelta()

Dupwind = cottonmouth_Z4c.mk_stencil(
    "Dupwind",
    la,
    h_step(evo_shift[ub] * kdelta[la, lb]) * finite_difference_stencil(stencil_order, 1, 1, la) +
    h_step(-evo_shift[ub] * kdelta[la, lb]) * finite_difference_stencil(stencil_order, 1, -1, la)
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

# Diagnostic-only quantities (not physics, purely for debugging the x=0
# spike seen in Theta_rhs/trK_rhs/HamCons -- see context.md). Isolates the
# two suspect "Op" terms so their raw values can be inspected directly via
# out_tsv, since R[li,lj]/covd2_alpha[li,lj] are otherwise pure local
# temporaries with no persistent storage.
RicciTermDiag = cottonmouth_Z4c.decl(
    "RicciTermDiag",
    [],
    parity=parity_scalar
)
D2AlphaTermDiag = cottonmouth_Z4c.decl(
    "D2AlphaTermDiag",
    [],
    parity=parity_scalar
)

# trK_rhs term-by-term diagnostics (see context.md 2026-08-26: an ingoing
# trK wave is ~11x stronger than MHDuet's at t~3). Each is exactly one
# summand of trK_rhs, so they add to the pre-dissipation RHS.
# Each piece is a standalone stored GF (do NOT sum them into another
# named decl -- CSE would demote the pieces to tile-local temps, the
# same bug that ate phiR_rhs when Tmunu was merged into boson_star_rhs).
TrkD2aDiag = cottonmouth_Z4c.decl("TrkD2aDiag", [], parity=parity_scalar)
TrkZgradDiag = cottonmouth_Z4c.decl("TrkZgradDiag", [], parity=parity_scalar)
TrkAtAtDiag = cottonmouth_Z4c.decl("TrkAtAtDiag", [], parity=parity_scalar)
TrkKsqDiag = cottonmouth_Z4c.decl("TrkKsqDiag", [], parity=parity_scalar)
TrkKappaDiag = cottonmouth_Z4c.decl("TrkKappaDiag", [], parity=parity_scalar)
TrkMatterDiag = cottonmouth_Z4c.decl("TrkMatterDiag", [], parity=parity_scalar)
TrkAdvDiag = cottonmouth_Z4c.decl("TrkAdvDiag", [], parity=parity_scalar)

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

Rt_tmp = cottonmouth_Z4c.decl(
    "Rt_tmp",
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

# MHDuet's Zu^i (AdvanceLevel.cpp: Zu_x = 0.5 * chi_max * (Gamh_x - Gamt_x)).
# Previously this was the literature Z^i = (1/2)(Gamh^i - Gamt^i) without χ,
# which made trK's 2*Zu·∇α and Theta's -Zu·∇α too small by χ, and (after
# substituting) put an extra 1/χ on Gamh's Z-damping. ZtCons below stays
# the undensitized literature Z^i; only the RHS source uses this Zu.
Zvec = cottonmouth_Z4c.decl("Zvec", [ui])

cottonmouth_Z4c.add_substitution_rule(
    Zvec[ui], Rational(1, 2) * chi * (evo_Gammat[ui] - Gammatd[ui])
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
    Rt_tmp[li, lj],
    - Rational(1, 2) * gt[ul, um] * D(gt[li, lj], ll, lm)
    + Rational(1, 2) * (
        + gt[lk, li] * D(evo_Gammat[uk], lj)
        + gt[lk, lj] * D(evo_Gammat[uk], li)
    )
    + Rational(1, 2) * (
        + Gammatd[uk] * Gammat[li, lj, lk]
        + Gammatd[uk] * Gammat[lj, li, lk]
    )
)

fun_z4c_constraints.split_loop()

fun_z4c_constraints.add_eqn(
    Rt[li, lj],
    Rt_tmp[li, lj]
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
    + use_matter_terms * (-16) * pi * rho
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
    + use_matter_terms * (-8) * pi * chi * gt[ui, uj] * Svec[lj]
)

# Recompute trK_rhs pieces on the completed post-step state (analysis),
# so out_norm sees the physical terms at y_{n+1} rather than RK4 stage k4.
fun_z4c_constraints.split_loop()
fun_z4c_constraints.add_eqn(
    TrkD2aDiag, -chi * gt[ui, uj] * covd2_alpha[li, lj]
)
fun_z4c_constraints.add_eqn(
    TrkZgradDiag, 2 * Zvec[ui] * D(evo_lapse, li)
)
fun_z4c_constraints.add_eqn(
    TrkAtAtDiag, evo_lapse * At[li, lj] * At[ui, uj]
)
fun_z4c_constraints.add_eqn(
    TrkKsqDiag, evo_lapse * Rational(1, 3) * (trK + 2 * Theta)**2
)
fun_z4c_constraints.add_eqn(
    TrkKappaDiag, kappa_1 * (1 - kappa_2) * Theta
)
fun_z4c_constraints.add_eqn(
    TrkMatterDiag, use_matter_terms * 4 * pi * evo_lapse * (trS + rho)
)
fun_z4c_constraints.add_eqn(
    TrkAdvDiag, evo_shift[ui] * Dupwind(trK, li)
)

sync_state = ExplicitSyncBatch(
    cottonmouth_Z4c.get_state(),
    ScheduleBin.PostSubStep,
    schedule_before=["ADMBaseX_SetADMVars"],
    name="sync_state",
)
sync_z4c = ExplicitSyncBatch(
    [At, gt, chi, evo_lapse, evo_shift, trK],
    "z4c_to_adm_group",
    schedule_before=["z4c_to_adm"],
    name="sync_z4c",
)
sync_z4c_pt2 = ExplicitSyncBatch(
    [gt],
    "adm_to_z4c_pt2_group",
    schedule_before=["adm_to_z4c_pt2"],
    name="sync_z4c_pt2",
    # IN ODESolvers_PostStep BEFORE ADMBaseX_SetADMVars
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
    Rt_tmp[li, lj],
    - Rational(1, 2) * gt[ul, um] * D(gt[li, lj], ll, lm)
    + Rational(1, 2) * pull_out(
        + gt[lk, li] * D(evo_Gammat[uk], lj)
        + gt[lk, lj] * D(evo_Gammat[uk], li)
    )
    + Rational(1, 2) * pull_out(
        + Gammatd[uk] * Gammat[li, lj, lk]
        + Gammatd[uk] * Gammat[lj, li, lk]
    )
)

fun_z4c_rhs.split_loop()

fun_z4c_rhs.add_eqn(
    Rt[li, lj],
    Rt_tmp[li, lj]
    + gt[ul, um] * Gammat[uk, ll, li] * Gammat[lj, lk, lm]
    + gt[ul, um] * Gammat[uk, ll, lj] * Gammat[li, lk, lm]
    + gt[ul, um] * Gammat[uk, li, lm] * Gammat[lk, ll, lj]
)

fun_z4c_rhs.add_eqn(
    R[li, lj],
    Rchi[li, lj] + Rt[li, lj]
)

# Diagnostic-only (see declaration above): the exact Ricci-scalar term as
# it appears in Theta_rhs's/HamCons's "Op" part, isolated for direct
# out_tsv inspection of the x=0 spike.
fun_z4c_rhs.add_eqn(
    RicciTermDiag,
    chi * gt[ui, uj] * R[li, lj]
)

# Diagnostic-only: the exact D^2(alpha) term as it appears in trK_rhs's
# "Op" part, isolated for direct out_tsv inspection of the x=0 spike.
fun_z4c_rhs.add_eqn(
    D2AlphaTermDiag,
    chi * gt[ui, uj] * covd2_alpha[li, lj]
)

fun_z4c_rhs.soft_split()

# Eq. (6) of [1]. Damping term deliberately has NO evo_lapse factor --
# MHDuet/mhduetBS-2.0's own compiled Theta_rhs (AdvanceLevel.cpp:
# d_theta_o2_t3_m0_l0) computes this same damping as
# "-alpha*kappa_z1_p*(2+kappa_z2_p)*theta" with kappa_z1_p = p_kappa_z1/
# alpha, so its alpha factor cancels EXACTLY, leaving a lapse-INDEPENDENT
# damping rate of p_kappa_z1*(2+kappa_z2). Multiplying kappa_1 by
# evo_lapse here (as this thorn previously did) makes the damping rate
# alpha-DEPENDENT instead, so a constant kappa_1 parfile value can only
# ever match MHDuet's rate at one particular alpha -- not throughout an
# evolution where alpha itself is drifting (which is exactly the
# phenomenon under investigation). Dropping evo_lapse here reproduces
# MHDuet's cancellation exactly (rather than introducing a redundant
# kappa_1/evo_lapse division that immediately cancels back out, which
# would only add floating-point noise for no benefit), so kappa_1 can
# now be set to EXACTLY match MHDuet's p_kappa_z1.
# Theta_rhs's quadratic trK/Theta source term is NOT the same between
# this recipe's literature form and mhduetBS-2.0's own CCZ4 form --
# confirmed by direct MathML-equation extraction from mhduetBS-2.0's own
# documentation/PDEModel-*.xml (not just its compiled RHS): expanding
# MHDuet's ".6666666666666667*trK^2 + .6666666666666667*Theta*(trK -
# 2*Theta)" gives (2/3)*trK^2 + (2/3)*trK*Theta - (4/3)*Theta^2, versus
# the standard Z4c form's (2/3)*(trK+2*Theta)^2 = (2/3)*trK^2 +
# (8/3)*trK*Theta + (8/3)*Theta^2 -- these differ in both the cross-term
# coefficient AND the sign/magnitude of the Theta^2 term, so they are not
# a relabeling of the same expression. See --mhduet-theta-rhs-convention.
theta_quadratic_term = (
    mhduet_theta_rhs_convention * (
        Rational(2, 3) * trK**2 + Rational(2, 3) * Theta * (trK - 2 * Theta)
    )
    + (1 - mhduet_theta_rhs_convention) * (
        Rational(2, 3) * (trK + 2 * Theta)**2
    )
)

fun_z4c_rhs.add_eqn(
    Theta_rhs,
    Rational(1, 2) * evo_lapse * (
        + chi * gt[ui, uj] * R[li, lj]
        - At[li, lj] * At[ui, uj]
        + theta_quadratic_term
    )
    # Z-vector gradient source, previously missing entirely -- confirmed
    # via MathML extraction (theta_op.xml terms 1-3: "-Zu_i*d_i(Alpha)").
    # Ablation-tested (2025-08-26): removing this term alone (with all
    # other fixes active) did not restore the kappa-only-fixes baseline --
    # ruled out as the sole cause of the regression. Restored per boson.md
    # ("doing something you know is different from MHDuet is the wrong
    # path") -- pivoting to single-Euler-step diagnosis instead of further
    # full-evolution ablation.
    - Zvec[ui] * D(evo_lapse, li)
    # Damping (lapse-independent, see comment above)
    - kappa_1 * (2 + kappa_2) * Theta
    # Advection (upwinded)
    + evo_shift[ui] * Dupwind(Theta, li)
    # Matter
    + use_matter_terms * (-evo_lapse) * 8 * pi * rho
)

# Eq. (4) of [1]
fun_z4c_rhs.add_eqn(
    AtTF[li, lj],
    - covd2_alpha[li, lj]
    + evo_lapse * (
        + R[li, lj]
        # Matter. use_atij_matter drops this at codegen for --vacuum /
        # --no-atij-backreaction; include_atij_backreaction is the runtime
        # switch for the live 4m thorn (see that parameter's docstring).
        + use_atij_matter * include_atij_backreaction * (-8) * pi * S[li, lj]
    )
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
    # Advection (upwinded)
    + evo_shift[ui] * Dupwind(chi, li)
)

# Eq. (3) of [1]
fun_z4c_rhs.add_eqn(
    trK_rhs,
    - chi * gt[ui, uj] * covd2_alpha[li, lj]
    # Z-vector gradient source, previously missing entirely -- confirmed
    # via MathML extraction (trk_op.xml terms 20-22: "2.0*Zu_i*d_i(Alpha)").
    # NOTE: trK_rhs's own quadratic term below, unlike Theta_rhs's, DOES
    # match MHDuet's (2/3)*alpha*(trK+2*Theta)^2 exactly as-is (trk_op.xml
    # term 23) -- no convention flag needed here. Ablation-tested
    # (2025-08-26): removing this term alone did not restore the
    # kappa-only-fixes baseline -- restored, see Theta_rhs's matching note.
    + 2 * Zvec[ui] * D(evo_lapse, li)
    + evo_lapse * (
        At[li, lj] * At[ui, uj]
        + Rational(1, 3) * (trK + 2 * Theta)**2
    )
    # Damping (lapse-independent -- see Theta_rhs's kappa_1 comment above;
    # MHDuet's matching term is d_trK_o2_t21_m0_l0's
    # "alpha*kappa_z1_p*(1-kappa_z2_p)*theta" with the same kappa_z1_p =
    # p_kappa_z1/alpha cancellation)
    + kappa_1 * (1 - kappa_2) * Theta
    # Advection (upwinded)
    + evo_shift[ui] * Dupwind(trK, li)
    # Matter
    + use_matter_terms * 4 * pi * evo_lapse * (trS + rho)
)

# trK_rhs pieces, each exactly one summand (pre-dissipation).
fun_z4c_rhs.add_eqn(
    TrkD2aDiag, -chi * gt[ui, uj] * covd2_alpha[li, lj]
)
fun_z4c_rhs.add_eqn(
    TrkZgradDiag, 2 * Zvec[ui] * D(evo_lapse, li)
)
fun_z4c_rhs.add_eqn(
    TrkAtAtDiag, evo_lapse * At[li, lj] * At[ui, uj]
)
fun_z4c_rhs.add_eqn(
    TrkKsqDiag, evo_lapse * Rational(1, 3) * (trK + 2 * Theta)**2
)
fun_z4c_rhs.add_eqn(
    TrkKappaDiag, kappa_1 * (1 - kappa_2) * Theta
)
fun_z4c_rhs.add_eqn(
    TrkMatterDiag, use_matter_terms * 4 * pi * evo_lapse * (trS + rho)
)
fun_z4c_rhs.add_eqn(
    TrkAdvDiag, evo_shift[ui] * Dupwind(trK, li)
)

fun_z4c_rhs.split_loop()

# Eq. (5) of [1]
fun_z4c_rhs.add_eqn(
    evo_Gammat_rhs[ui],
    - 2 * At[ui, uj] * D(evo_lapse, lj)
    # Missing Theta-gradient source term -- confirmed via MathML extraction
    # (gamh_x_op.xml terms 23-25: "-(2.0*theta*gtu_xj)*d/dj[Alpha]").
    # Ablation-tested (2025-08-26): confirmed NOT a contributor to the
    # regression (restoring this term alone, with the 1/chi matter-term
    # factor below still disabled, kept the good result) -- see
    # context.md. Permanently active.
    - 2 * Theta * gt[ui, uj] * D(evo_lapse, lj)
    + 2 * evo_lapse * (
        + Gammat[ui, lj, lk] * At[uj, uk]
        - Rational(3, 2) * At[ui, uj] * (1 / chi) * D(chi, lj)
        - Rational(2, 3) * gt[ui, uj] * D(trK, lj)
        - Rational(1, 3) * gt[ui, uj] * D(Theta, lj)
    )
    + gt[uj, uk] * D(evo_shift[ui], lj, lk)
    + Rational(1, 3) * gt[ui, uj] * D(evo_shift[uk], lj, lk)
    # These two shift-divergence terms use the EVOLVED evo_Gammat^j, not
    # the algebraic Gammatd^j -- confirmed via MathML extraction
    # (gamh_x_op.xml terms 0-2, 15: "-Gamh_j*d_j(Betau_i)" and
    # "(2/3)*Gamh_i*div_Beta", using "Gamh" = this thorn's evo_Gammat, the
    # EVOLVED state variable, not a freshly-recomputed algebraic
    # connection). Since evo_Gammat^j - Gammatd^j = 2*Z^j, this differs
    # from the previous Gammatd-based form by -2*Z^j*d_j(beta^i) +
    # (4/3)*Z^i*d_j(beta^j), which was previously missing entirely.
    # Ablation-tested (2025-08-26): reverting THIS substitution alone (with
    # all other fixes active) reproduced the same regression as the full
    # fix set, ruling it out as the cause.
    - evo_Gammat[uj] * D(evo_shift[ui], lj)
    + Rational(2, 3) * evo_Gammat[ui] * D(evo_shift[uj], lj)
    # Damping: MHDuet d_Gamh_*_o2_t24_m0_l0
    #   -2*alpha*inv_chi*Zu_i*(kappa_z1_p + (4/3)*theta + (2/3)*trK)
    # with kappa_z1_p = p_kappa_z1/alpha and Zu = (chi/2)*(Gamh - Gamt).
    # Alpha cancels against kappa_z1_p, leaving
    #   -(2/chi)*Zu*(kappa_1 + alpha*((4/3)*Theta + (2/3)*trK))
    # which is -(Gamh - Gamt)*(...) -- no leftover 1/chi on (Gamh-Gamt).
    - (2 / chi) * Zvec[ui] * (
        + kappa_1
        + evo_lapse * (Rational(4, 3) * Theta + Rational(2, 3) * trK)
    )
    # Advection (upwinded)
    + evo_shift[uj] * Dupwind(evo_Gammat[ui], lj)
    # Matter term: kept WITHOUT the 1/chi factor, deliberately, despite
    # MHDuet's own MathML formula having one (gamh_x_op.xml term 26:
    # "-16*pi*alpha*inv_chi*gtu_xj*Jtd_ADM_j"). Ablation-tested
    # (2025-08-26): adding this factor alone (isolated from every other
    # fix this segment) reproduces, in full, a genuine regression versus
    # the pre-fix behavior -- see context.md's "ROOT CAUSE OF THE
    # REGRESSION" section. Leading hypothesis, NOT YET VERIFIED: MHDuet's
    # "Jtd_ADM" and this thorn's own "Svec" likely carry different
    # implicit chi-normalization conventions, making a literal 1/chi
    # multiplication onto Svec an over-correction rather than a true
    # match, even though it looks like a direct symbol-for-symbol
    # correspondence in the extracted MathML term. Left off pending that
    # verification (would need comparing Svec's own defining substitution
    # rule against how MHDuet actually builds/normalizes Jtd_ADM_j in its
    # own source -- not yet traced).
    + use_matter_terms * (-16) * pi * evo_lapse * gt[ui, uj] * Svec[lj]
)

# Eq. (2) of [1], plus CCZ4's continuous det(gt)=1 constraint-damping term
# (see kappa_cc's docstring above) -- vanishes identically at exact
# det(gt)=1, so this is a pure restoring force, not a formula change to
# the vacuum/matter dynamics.
fun_z4c_rhs.add_eqn(
    gt_rhs[li, lj],
    # Trace-cleaned At_ij (At_ij - (1/3)*tr(At)*gt_ij), not raw At_ij --
    # confirmed via MathML extraction (gtd_xx_op.xml term 4: "-2*alpha*
    # (Atd_xx - (1/3)*trAt*lambda_0*gtd_xx)", with mhduetBS-2.0's own
    # problem.lambda_0 = 1.0). Ablation-tested (2025-08-26): confirmed NOT
    # a contributor to the regression (disabling this + At_rhs's bare-trK
    # fix together made no difference versus the already-good baseline) --
    # see context.md. Permanently active.
    - 2 * evo_lapse * (At[li, lj] - Rational(1, 3) * gt[li, lj] * gt[uk, ul] * At[lk, ll])
    + gt[lk, li] * D(evo_shift[uk], lj)
    + gt[lk, lj] * D(evo_shift[uk], li)
    - Rational(2, 3) * gt[li, lj] * D(evo_shift[uk], lk)
    # Advection (upwinded)
    + evo_shift[uk] * Dupwind(gt[li, lj], lk)
    # CCZ4 continuous constraint damping (det(gt) = 1), lapse-independent
    # -- MHDuet's kappa_cc_p = p_kappa_cc/alpha, multiplied by alpha again
    # in gt_rhs (AdvanceLevel.cpp: "kappa_cc_p*alpha*gtd_xx*log(detgtd)"),
    # cancels exactly to a bare p_kappa_cc rate, same pattern as kappa_1
    # above (this term previously carried a bare evo_lapse factor with no
    # such cancellation, so it inherited the same lapse-dependence bug).
    - Rational(1, 3) * kappa_cc * gt[li, lj] * log(detgt)
)

fun_z4c_rhs.split_loop()

fun_z4c_rhs.add_eqn(
    At_rhs[li, lj],
    chi * (
        + AtTF[li, lj]
        - Rational(1, 3) * gt[li, lj] * gt[uk, ul] * AtTF[lk, ll]
    )
    # Bare trK, not (trK + 2*Theta) -- confirmed via MathML extraction
    # (atd_xx_op.xml term 5: "alpha*(trK*Atd_xx - 2*Atd_xx*Atud_xx - ...)",
    # no Theta term at all). Ablation-tested (2025-08-26): confirmed NOT a
    # contributor to the regression (disabling this + gt_rhs's
    # trace-cleaning fix together made no difference versus the
    # already-good baseline) -- see context.md. Permanently active.
    + evo_lapse * (
        + trK * At[li, lj]
        - 2 * At[uk, li] * At[lk, lj]
    )
    + At[lk, li] * D(evo_shift[uk], lj)
    + At[lk, lj] * D(evo_shift[uk], li)
    - Rational(2, 3) * At[li, lj] * D(evo_shift[uk], lk)
    # Advection (upwinded)
    + evo_shift[uk] * Dupwind(At[li, lj], lk)
    # CCZ4 continuous constraint damping (tr(At) = 0), lapse-independent
    # -- same kappa_cc_p*alpha = p_kappa_cc cancellation as gt_rhs above.
    - Rational(1, 3) * kappa_cc * gt[li, lj] * gt[uk, ul] * At[lk, ll]
)

# Eq. (11) of [1]
fun_z4c_rhs.add_eqn(
    evo_lapse_rhs,
    - 2 * evo_lapse * trK
    # Advection (upwinded)
    + evo_shift[ui] * Dupwind(evo_lapse, li)
)

# Eq. (12) of [1], with the standard Gamma-driver's 3/4 coefficient on the
# Gammat^i source term (previously missing -- this thorn used a bare
# coefficient of 1, i.e. a 33% stronger drive toward Gammat^i than the
# literature-standard 3/4 * Gammat^i - eta * beta^i form). Confirmed via
# direct comparison against MHDuet/mhduetBS-2.0's own compiled shift RHS
# (AdvanceLevel.cpp: d_Betau_x_o0_t0_m0_l0 = -feta*Betau_x + 0.75*(alpha*
# lambda_f1 + lambda_f0)*Gamh_x, with lambda_f0=1.0/lambda_f1=0.0 in
# mhduetBS-2.0's example_BS -- i.e. exactly 0.75*Gamh_x - eta*Betau_x),
# which uses the literature 3/4 coefficient while this thorn did not. eta
# itself was already correct (both effectively 1.0 in the comparison
# region; MHDuet's feta only decays for r > R_0 = 20, far outside any star
# studied here) -- only the missing 3/4 factor is being fixed here.
fun_z4c_rhs.add_eqn(
    evo_shift_rhs[ui],
    + Rational(3, 4) * evo_Gammat[ui]
    - eta_beta * evo_shift[ui]
    # Advection (upwinded)
    + evo_shift[uj] * Dupwind(evo_shift[ui], lj)
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

fun_z4c_diss.split_loop()

chi_rhs_diss = cottonmouth_Z4c.overwrite(chi_rhs)
fun_z4c_diss.add_eqn(
    chi_rhs_diss,
    chi_rhs + dissipation_epsilon * (
        + div_diss(chi, l0)
        + div_diss(chi, l1)
        + div_diss(chi, l2)
    )
)

fun_z4c_diss.split_loop()

trK_rhs_diss = cottonmouth_Z4c.overwrite(trK_rhs)
fun_z4c_diss.add_eqn(
    trK_rhs_diss,
    trK_rhs + dissipation_epsilon * (
        + div_diss(trK, l0)
        + div_diss(trK, l1)
        + div_diss(trK, l2)
    )
)

fun_z4c_diss.split_loop()

GammaHat_rhs_diss = cottonmouth_Z4c.overwrite(evo_Gammat_rhs)
fun_z4c_diss.add_eqn(
    GammaHat_rhs_diss[ui],
    evo_Gammat_rhs[ui] + dissipation_epsilon * (
        + div_diss(evo_Gammat[ui], l0)
        + div_diss(evo_Gammat[ui], l1)
        + div_diss(evo_Gammat[ui], l2)
    )
)

fun_z4c_diss.split_loop()

gt_rhs_diss = cottonmouth_Z4c.overwrite(gt_rhs)
fun_z4c_diss.add_eqn(
    gt_rhs_diss[li, lj],
    gt_rhs[li, lj] + dissipation_epsilon * (
        + div_diss(gt[li, lj], l0)
        + div_diss(gt[li, lj], l1)
        + div_diss(gt[li, lj], l2)
    )
)

fun_z4c_diss.split_loop()

At_rhs_diss = cottonmouth_Z4c.overwrite(At_rhs)
fun_z4c_diss.add_eqn(
    At_rhs_diss[li, lj],
    At_rhs[li, lj] + dissipation_epsilon * (
        + div_diss(At[li, lj], l0)
        + div_diss(At[li, lj], l1)
        + div_diss(At[li, lj], l2)
    )
)

fun_z4c_diss.split_loop()

evo_lapse_rhs_diss = cottonmouth_Z4c.overwrite(evo_lapse_rhs)
fun_z4c_diss.add_eqn(
    evo_lapse_rhs_diss,
    evo_lapse_rhs + dissipation_epsilon * (
        + div_diss(evo_lapse, l0)
        + div_diss(evo_lapse, l1)
        + div_diss(evo_lapse, l2)
    )
)

fun_z4c_diss.split_loop()

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
# Apply NewRadX
###
nrx_Theta = NewRadXBoundaryBatch(
    Theta,
    sympify(0),
    sympify(1),
    radpower_Theta,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_Theta",
)

nrx_chi = NewRadXBoundaryBatch(
    chi,
    sympify(1),
    sympify(1),
    radpower_chi,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_chi",
)

nrx_trK = NewRadXBoundaryBatch(
    trK,
    sympify(0),
    sympify(1),
    radpower_trK,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_trK",
)

nrx_evo_Gammat = NewRadXBoundaryBatch(
    evo_Gammat,
    sympify(0),
    sympify(1),
    radpower_evo_Gammat,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_evo_Gammat",
)

nrx_gt_xx = NewRadXBoundaryBatch(
    gt[l0, l0],
    sympify(1),
    sympify(1),
    radpower_gt,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_gt_xx",
)

nrx_gt_xy = NewRadXBoundaryBatch(
    gt[l0, l1],
    sympify(0),
    sympify(1),
    radpower_gt,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_gt_xy",
)

nrx_gt_xz = NewRadXBoundaryBatch(
    gt[l0, l2],
    sympify(-0),
    sympify(1),
    radpower_gt,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_gt_xz",
)

nrx_gt_yy = NewRadXBoundaryBatch(
    gt[l1, l1],
    sympify(1),
    sympify(1),
    radpower_gt,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_gt_yy",
)

nrx_gt_yz = NewRadXBoundaryBatch(
    gt[l1, l2],
    sympify(0),
    sympify(1),
    radpower_gt,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_gt_yz",
)

nrx_gt_zz = NewRadXBoundaryBatch(
    gt[l2, l2],
    sympify(1),
    sympify(1),
    radpower_gt,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_gt_zz",
)

nrx_At = NewRadXBoundaryBatch(
    At,
    sympify(0),
    sympify(1),
    radpower_At,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_At",
)

nrx_evo_lapse = NewRadXBoundaryBatch(
    evo_lapse,
    sympify(1),
    sympify(1),
    radpower_evo_lapse,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_evo_lapse",
)

nrx_evo_shift = NewRadXBoundaryBatch(
    evo_shift,
    sympify(0),
    sympify(1),
    radpower_evo_shift,
    rhs_group,
    schedule_after=["z4c_apply_dissipation"],
    cond="apply_NewRadX",
    name="z4c_apply_NewRadX_evo_shift",
)

###
# Bake the cake
###
cottonmouth_Z4c.bake(
    do_cse=True,
    temporary_promotion_strategy=promote_none(),
    do_madd=False,
    do_recycle_temporaries=False,
    cse_optimization_level=CseOptimizationLevel.Optimal,
    soft_split_retainment_strategy=retain_rank(50),
    ordering_fn=functools.partial(
        prioritize_rare_symbols, consider_frequency=True, complexity_factor=0.0
    )
)

###
# Thorn creation
###
recipe_dir = Path(__file__).resolve().parent

with (recipe_dir / 'cottonmouth_agpl3.txt').open('r') as fd:
    license_file = fd.read()

with (recipe_dir / 'cottonmouth_agpl3_header.txt').open('r') as fd:
    license_header = fd.read()

CppCarpetXWizard(
    cottonmouth_Z4c,
    CppCarpetXGenerator(
        cottonmouth_Z4c,
        sync_mode=SyncMode.HandsOff,
        interior_sync_schedule_target=post_step_group,
        extra_schedule_blocks=[
            initial_group,
            post_step_group,
            rhs_group,
            analysis_group,
        ],
        explicit_syncs=[
            sync_state,
            sync_z4c,
            sync_z4c_pt2
        ],
        new_rad_x_boundary_fns=[
            nrx_Theta,
            nrx_chi,
            nrx_trK,
            nrx_evo_Gammat,
            nrx_gt_xx,
            nrx_gt_xy,
            nrx_gt_xz,
            nrx_gt_yy,
            nrx_gt_yz,
            nrx_gt_zz,
            nrx_At,
            nrx_evo_lapse,
            nrx_evo_shift,
        ]
    ),
    license_header=license_header,
    license_file=license_file
).generate_thorn()

# References
# [1] https://arxiv.org/pdf/1212.2901 (typo in constraints, refer to [2])
# [2] https://arxiv.org/pdf/0912.2920
