#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
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

"""
The CCZ4 right-hand side with the moving-puncture gauge, as GRTeclyn computes it, for AMReX's ParallelFor.

The generated class EE::CCZ4RHS<order> (CCZ4RHS<order>.hpp) replaces the CCZ4 right-hand side of GRTeclyn's
Examples/BinaryBH (BinaryBHLevel::specific_eval_rhs, spatial_derivative_order 4): the three ParallelFor launches of
CCZ4RHS<FourthOrderDerivatives>::compute_chi_and_h_ij, CCZ4RHS::compute_A_ij_and_Theta_and_Gamma, and
MovingPunctureGauge<FourthOrderDerivatives>::calculate_rhs followed by CCZ4RHS::apply_dissipation. It has
CCZ4RHS's constructor (dx and an optional cosmological constant), reads the same parameters, and computes the same
right-hand side, split into the same three kernels: compute(rhs, state) launches them in order.

Sources. The formulation is CCZ4 [1], with the conformal decomposition and modifications of [2] and [3] as implemented
in GRChombo [4] and ported to AMReX in GRTeclyn [5, 6]. Since the generated kernel must agree with GRTeclyn to
roundoff, GRTeclyn's implementation is the specification; each block below names the GRTeclyn source it transcribes
(paths in the GRTeclyn repository at commit 8d6de53e), and grteclyn/analysis/recipe-justification.md maps every term
to [1-3] and to the GRChombo recipe (recipes/foreign/grchombo/ccz4_grchombo.py), from which this one is adapted.

[1] D. Alic, C. Bona-Casas, C. Bona, L. Rezzolla, C. Palenzuela, Phys. Rev. D 85, 064040 (2012), arXiv:1106.2254.
[2] D. Alic, W. Kastaun, L. Rezzolla, Phys. Rev. D 88, 064049 (2013), arXiv:1307.7391.
[3] K. Clough et al., Class. Quantum Grav. 32, 245011 (2015), arXiv:1503.03436 (GRChombo).
[4] T. Andrade et al., J. Open Source Softw. 6, 3703 (2021) (GRChombo).
[5] Source/CCZ4/CCZ4RHS.hpp, CCZ4RHS.impl.hpp, CCZ4Geometry.hpp, MovingPunctureGauge.hpp; Source/Maths/TensorAlgebra.hpp;
    Source/Grids/Coordinates.hpp.
[6] Source/Grids/FourthOrderDerivatives.hpp.

Differences from the GRChombo recipe, all from GRTeclyn's source:
* The formulation is selected without a branch: bssn_coeff (ccz4.formulation: 0 CCZ4, 1 BSSN) and
  ccz4_coeff = 1 - bssn_coeff multiply Z, Theta's right-hand side and the two variants of K's. Both formulations are
  therefore generated, as in GRTeclyn.
* The Gamma-driver damping is eta(r) = eta (c_f R^2 / (r^2 + R^2) + 1 - c_f) (gauge.enable_eta_cutoff gives c_f,
  gauge.eta_cutoff_radius R), with r = max(|x - center|, 1e-6) at the cell center x = (i + 1/2) dx. It is computed
  everywhere, as in GRTeclyn, whether or not the cutoff is enabled.
* The work is split into GRTeclyn's three kernels; dissipation is added in the third, to every variable.

Usage: python ccz4_grteclyn.py [--fd-order 4] [--out-dir DIR]
"""

import argparse
import sys
from typing import Any, cast

from sympy import Rational, Idx, IndexedBase, Expr, Abs, Piecewise, Float

from EinsteinEngine import *

parser = argparse.ArgumentParser(description='GRTeclyn CCZ4 RHS slot-in kernel')
parser.add_argument('--fd-order', type=int, default=4, choices=[4], help='Finite-difference order (GRTeclyn: 4 or 6).')
parser.add_argument('--out-dir', default='.', help='Directory to write the header to.')
args = parser.parse_args(sys.argv[1:])

fd_order = args.fd_order
d = 3  # GR_SPACEDIM

cc = AMReXKernelClass(
    f"CCZ4RHS{fd_order}",
    namespace="EE",
    ctor_args=[
        AMReXCtorArg("amrex::Real", "a_dx"),
        AMReXCtorArg("amrex::Real", "a_cosmological_constant", "0"),
    ],
    # CCZ4RHS's constructor (CCZ4_params_t::fill_params, evolution.sigma) and MovingPunctureGauge's (params_t::fill_params).
    ctor_prelude=[
        "CCZ4_params_t ccz4_params{};",
        "ccz4_params.fill_params();",
        "MovingPunctureGauge<>::params_t gauge_params{};",
        "gauge_params.fill_params();",
        "amrex::Real a_sigma{};",
        "{",
        "    GRParmParse pp;",
        "    pp.get(\"evolution.sigma\", a_sigma);",
        "}",
    ],
    dx="a_dx",
    includes=["CCZ4RHS.hpp", "MovingPunctureGauge.hpp", "GRParmParse.hpp"],
    derivative_stencil_width=fd_order + 1,
)

ul, ll = cc.mk_pair("l")
um, lm = cc.mk_pair("m")
un, ln = cc.mk_pair("n")

###
# Parameters: CCZ4_params_t and MovingPunctureGauge::params_t fields, evolution.sigma, and the cosmological constant.
###
bssn_coeff = cc.add_param("bssn_coeff", init="ccz4_params.bssn_coeff")
kappa1 = cc.add_param("kappa1", init="ccz4_params.kappa1")
kappa2 = cc.add_param("kappa2", init="ccz4_params.kappa2")
kappa3 = cc.add_param("kappa3", init="ccz4_params.kappa3")
covariant_z4_coeff = cc.add_param("covariant_z4_coeff", init="ccz4_params.covariant_z4_coeff")
lapse_advec_coeff = cc.add_param("lapse_advec_coeff", init="gauge_params.lapse_advec_coeff")
lapse_power = cc.add_param("lapse_power", init="gauge_params.lapse_power")
lapse_coeff = cc.add_param("lapse_coeff", init="gauge_params.lapse_coeff")
shift_Gamma_coeff = cc.add_param("shift_Gamma_coeff", init="gauge_params.shift_Gamma_coeff")
shift_advec_coeff = cc.add_param("shift_advec_coeff", init="gauge_params.shift_advec_coeff")
eta = cc.add_param("eta", init="gauge_params.eta")
eta_cutoff_coeff = cc.add_param("eta_cutoff_coeff", init="gauge_params.eta_cutoff_coeff")
eta_cutoff_radius = cc.add_param("eta_cutoff_radius", init="gauge_params.eta_cutoff_radius")
center = [cc.add_param(f"center{a}", init=f"gauge_params.center[{a}]") for a in range(3)]
sigma = cc.add_param("sigma", init="a_sigma")
cosmological_constant = cc.add_param("cosmological_constant", init="a_cosmological_constant")
x, y, z = cc.mk_coords()

###
# State variables (Source/CCZ4/CCZ4StateVariables.hpp) and their right-hand sides.
###
def state(name: str, indices: list[Idx], symmetric: bool = False) -> tuple[IndexedBase, IndexedBase]:
    if symmetric:
        sym = [(indices[0], indices[1])]
        rhs = cc.decl(f"{name}_rhs", indices, symmetries=sym)
        return cc.decl(name, indices, rhs=rhs, symmetries=sym), rhs
    rhs = cc.decl(f"{name}_rhs", indices)
    return cc.decl(name, indices, rhs=rhs), rhs

chi, chi_rhs = state("chi", [])                  # conformal factor
h, h_rhs = state("h", [li, lj], symmetric=True)  # conformal metric
K, K_rhs = state("K", [])                        # trace of the extrinsic curvature
A, A_rhs = state("A", [li, lj], symmetric=True)  # trace-free conformal extrinsic curvature
Theta, Theta_rhs = state("Theta", [])            # CCZ4 Theta
Gamma, Gamma_rhs = state("Gamma", [ui])          # conformal connection functions \hat\Gamma^i
lapse, lapse_rhs = state("lapse", [])
shift, shift_rhs = state("shift", [ui])
B, B_rhs = state("B", [ui])                      # Gamma-driver auxiliary variable

###
# Derivative operators (Source/Grids/FourthOrderDerivatives.hpp). D(f, li) and D(f, li, lj) are EE's centered
# stencils of width fd_order + 1, which have FourthOrderDerivatives' diff1, diff2 and mixed_diff2 weights.
###
# advection_term: the advection of f along direction a, beta^a (D_a f), with GRTeclyn's lopsided stencils chosen by
# the sign of beta^a: the upwind one (points -1..3) where beta^a > 0, the downwind one (points -3..1) otherwise. It is
# written without a branch as beta+ D_up f + beta- D_down f, beta+- = (beta^a +- |beta^a|)/2: for beta^a > 0, beta+ =
# beta^a and beta- = 0 exactly, and vice versa, so each point gets exactly one of the two stencils, as in GRTeclyn.
adv_up = cc.mk_stencil("adv_up", la, finite_difference_stencil(fd_order, 1, 1, la))
adv_down = cc.mk_stencil("adv_down", la, finite_difference_stencil(fd_order, 1, -1, la))
# dissipation_term: (f_{-3} - 6 f_{-2} + 15 f_{-1} - 20 f_0 + 15 f_1 - 6 f_2 + f_3) / (64 dx) per direction.
diss = cc.mk_stencil("diss", la, kreiss_oliger_stencil(fd_order + 1, la))


def advec(f: Expr) -> Expr:
    """FourthOrderDerivatives::advection: sum over directions a of beta^a (upwinded D_a f)."""
    return cast(Expr, sum(
        (Rational(1, 2) * (shift[ua_] + Abs(shift[ua_])) * adv_up(f, la_)
         + Rational(1, 2) * (shift[ua_] - Abs(shift[ua_])) * adv_down(f, la_))
        for ua_, la_ in ((u0, l0), (u1, l1), (u2, l2))
    ))


def dissipation(f: Expr) -> Expr:
    """FourthOrderDerivatives::calculate_dissipation(..., m_sigma, ivar): sigma * sum_i KO_i f."""
    return cast(Expr, sigma * (diss(f, l0) + diss(f, l1) + diss(f, l2)))


divshift = cc.decl("divshift", [])
cc.add_substitution_rule(divshift, D(shift[ui], li))  # CCZ4Geometry::compute_divshift

###
# Kernel 1: CCZ4RHS::compute_chi_and_h_ij.
###
fn_chi_h = cc.create_function("compute_chi_and_h_ij")

# rhs.chi = advec.chi + (2/d) chi (lapse K - divshift)
fn_chi_h.add_eqn(chi_rhs, advec(chi) + Rational(2, d) * chi * (lapse * K - divshift))

# rhs.h_ij = advec.h_ij - 2 lapse A_ij - (2/d) h_ij divshift + h_ki d_j shift^k + h_kj d_i shift^k
fn_chi_h.add_eqn(
    h_rhs[li, lj],
    advec(h[li, lj]) - 2 * lapse * A[li, lj] - Rational(2, d) * h[li, lj] * divshift
    + h[lk, li] * D(shift[uk], lj) + h[lk, lj] * D(shift[uk], li)
)

###
# Kernel 2: CCZ4RHS::compute_A_ij_and_Theta_and_Gamma, with CCZ4Geometry::compute_inverse_metric,
# compute_christoffel and compute_ricci_Z.
###
# Kernel 2 is three quarters of the RHS time (grteclyn/analysis/hotloops.md), so it is the one whose splits and order
# are tuned: it reads the out-params of EE's split tuners (EinsteinEngine.tuning.tune_splitting). Without a tuner
# they are None and Early, which never split, and the kernel is GRTeclyn's.
fn_A = cc.create_function(
    "compute_A_ij_and_Theta_and_Gamma",
    auto_hard_split_predicate=get_optional_tuning_param('auto_hard_split_predicate', None),
    auto_soft_split_predicate=get_optional_tuning_param('auto_soft_split_predicate', None),
    auto_split_locus=get_optional_tuning_param('auto_split_locus', SplitLocus.Early),
)

# h^{ij}: the inverse of the evolved h_ij (compute_inverse_metric), not assuming det h = 1.
h_mat = cc.get_matrix(h[li, lj])
hUU = cc.decl("hUU", [ui, uj], symmetries=[(ui, uj)])
fn_A.add_eqn(hUU[ui, uj], h_mat.adjugate() * (1 / h_mat.det()))  # type: ignore[no-untyped-call]
cc.add_substitution_rule(h[ui, uj], hUU[ui, uj])

# chris.LLL[i][j][k] = 1/2 (d_k h_ji + d_j h_ki - d_i h_jk)
chris = cc.decl("chris", [li, lj, lk], symmetries=[(lj, lk)])
cc.add_substitution_rule(chris[li, lj, lk], Rational(1, 2) * (D(h[lj, li], lk) + D(h[lk, li], lj) - D(h[lj, lk], li)))
# chris.ULL[i][j][k] = h^{il} LLL[l][j][k]
chrisULL = cc.decl("chrisULL", [ui, lj, lk], symmetries=[(lj, lk)])
fn_A.add_eqn(chrisULL[ui, lj, lk], h[ui, ul] * chris[ll, lj, lk])
cc.add_substitution_rule(chris[ui, lj, lk], chrisULL[ui, lj, lk])
# chris_LLU[i][j][k] = h^{kl} LLL[i][j][l] (compute_ricci_hat)
cc.add_substitution_rule(chris[li, lj, uk], h[uk, ul] * chris[li, lj, ll])
# chris.contracted[i] = h^{jk} ULL[i][j][k]
chris_c = cc.decl("chris_c", [ui])
fn_A.add_eqn(chris_c[ui], h[uj, uk] * chris[ui, lj, lk])

# Z_over_chi[i] = ccz4_coeff 1/2 (Gamma^i - chris.contracted^i), ccz4_coeff = 1 - bssn_coeff; Z^i = chi Z_over_chi^i
ccz4_coeff = 1 - bssn_coeff
Z_over_chi = cc.decl("Z_over_chi", [ui])
cc.add_substitution_rule(Z_over_chi[ui], ccz4_coeff * Rational(1, 2) * (Gamma[ui] - chris_c[ui]))

# covdtilde2chi[k][l] = d_k d_l chi - ULL[m][k][l] d_m chi
covdtilde2chi = cc.decl("covdtilde2chi", [lk, ll], symmetries=[(lk, ll)])
cc.add_substitution_rule(covdtilde2chi[lk, ll], D(chi, lk, ll) - chris[um, lk, ll] * D(chi, lm))
boxtildechi = cc.decl("boxtildechi", [])
cc.add_substitution_rule(boxtildechi, h[ui, uj] * covdtilde2chi[li, lj])
dchi_dot_dchi = cc.decl("dchi_dot_dchi", [])
cc.add_substitution_rule(dchi_dot_dchi, h[ui, uj] * D(chi, li) * D(chi, lj))

# ricci_hat_ij = sum_k 1/2 (h_ki d_j Gamma^k + h_kj d_i Gamma^k) + 1/2 Gamma^k d_k h_ij
#              + sum_kl [-1/2 h^{kl} d_k d_l h_ij + ULL[k][l][i] LLU[j][k][l] + ULL[k][l][j] LLU[i][k][l]
#                        + ULL[k][i][l] LLU[k][j][l]]
ricci_hat = cc.decl("ricci_hat", [li, lj], symmetries=[(li, lj)])
cc.add_substitution_rule(
    ricci_hat[li, lj],
    Rational(1, 2) * (h[lk, li] * D(Gamma[uk], lj) + h[lk, lj] * D(Gamma[uk], li))
    + Rational(1, 2) * Gamma[uk] * D(h[li, lj], lk)
    - Rational(1, 2) * h[uk, ul] * D(h[li, lj], lk, ll)
    + chris[uk, ll, li] * chris[lj, lk, ul]
    + chris[uk, ll, lj] * chris[li, lk, ul]
    + chris[uk, li, ll] * chris[lk, lj, ul]
)

# ricci_chi_ij = 1/2 ((d-2) covdtilde2chi_ij + h_ij boxtildechi
#                     - ((d-2) d_i chi d_j chi + d h_ij dchi_dot_dchi) / (2 chi))
ricci_chi = cc.decl("ricci_chi", [li, lj], symmetries=[(li, lj)])
cc.add_substitution_rule(
    ricci_chi[li, lj],
    Rational(1, 2) * (
        (d - 2) * covdtilde2chi[li, lj] + h[li, lj] * boxtildechi
        - ((d - 2) * D(chi, li) * D(chi, lj) + d * h[li, lj] * dchi_dot_dchi) / (2 * chi)
    )
)

# compute_z_terms: sum_k Z_over_chi^k (h_ik d_j chi + h_jk d_i chi - h_ij d_k chi)
z_terms = cc.decl("z_terms", [li, lj], symmetries=[(li, lj)])
cc.add_substitution_rule(
    z_terms[li, lj],
    Z_over_chi[uk] * (h[li, lk] * D(chi, lj) + h[lj, lk] * D(chi, li) - h[li, lj] * D(chi, lk))
)

# ricci.LL_ij = (ricci_chi_ij + chi ricci_hat_ij + z_terms_ij) / chi; ricci.scalar = chi h^{ij} ricci.LL_ij
ricci = cc.decl("ricci", [li, lj], symmetries=[(li, lj)])
fn_A.add_eqn(ricci[li, lj], (ricci_chi[li, lj] + chi * ricci_hat[li, lj] + z_terms[li, lj]) / chi)
ricci_scalar = cc.decl("ricci_scalar", [])
fn_A.add_eqn(ricci_scalar, chi * h[ui, uj] * ricci[li, lj])

Z_dot_d1lapse = cc.decl("Z_dot_d1lapse", [])
cc.add_substitution_rule(Z_dot_d1lapse, chi * Z_over_chi[ui] * D(lapse, li))
dlapse_dot_dchi = cc.decl("dlapse_dot_dchi", [])
cc.add_substitution_rule(dlapse_dot_dchi, h[um, un] * D(lapse, lm) * D(chi, ln))

# covdtilde2lapse_kl = d_k d_l lapse - ULL[m][k][l] d_m lapse
# covd2lapse_kl = chi covdtilde2lapse_kl + 1/2 (d_k lapse d_l chi + d_k chi d_l lapse - h_kl dlapse_dot_dchi)
covd2lapse = cc.decl("covd2lapse", [lk, ll], symmetries=[(lk, ll)])
cc.add_substitution_rule(
    covd2lapse[lk, ll],
    chi * (D(lapse, lk, ll) - chris[um, lk, ll] * D(lapse, lm))
    + Rational(1, 2) * (D(lapse, lk) * D(chi, ll) + D(chi, lk) * D(lapse, ll) - h[lk, ll] * dlapse_dot_dchi)
)

# tr_covd2lapse = -(d/2) dlapse_dot_dchi - chi chris.contracted^i d_i lapse
#                 + h^{ij} (chi d_i d_j lapse + d_i lapse d_j chi)
tr_covd2lapse = cc.decl("tr_covd2lapse", [])
cc.add_substitution_rule(
    tr_covd2lapse,
    -Rational(d, 2) * dlapse_dot_dchi
    - chi * chris_c[ui] * D(lapse, li)
    + h[ui, uj] * (chi * D(lapse, li, lj) + D(lapse, li) * D(chi, lj))
)

# A_UU = compute_A_UU(vars, h_UU); Aij_squared = A_UU^{ij} A_ij
cc.add_substitution_rule(A[ui, uj], h[ui, uk] * h[uj, ul] * A[lk, ll])
tr_A2 = cc.decl("tr_A2", [])
cc.add_substitution_rule(tr_A2, A[li, lj] * A[ui, uj])

# Adot_TF = trace-free part of (-covd2lapse_ij + chi lapse ricci_ij) (make_trace_free)
Adot = cc.decl("Adot", [li, lj], symmetries=[(li, lj)])
cc.add_substitution_rule(Adot[li, lj], -covd2lapse[li, lj] + chi * lapse * ricci[li, lj])
Adot_TF = cc.decl("Adot_TF", [li, lj], symmetries=[(li, lj)])
cc.add_substitution_rule(Adot_TF[li, lj], Adot[li, lj] - Rational(1, d) * h[li, lj] * (h[uk, ul] * Adot[lk, ll]))

# kappa1_times_lapse = covariant_z4_coeff kappa1 + (1 - covariant_z4_coeff) kappa1 lapse
kappa1_times_lapse = cc.decl("kappa1_times_lapse", [])
cc.add_substitution_rule(
    kappa1_times_lapse, covariant_z4_coeff * kappa1 + (1 - covariant_z4_coeff) * kappa1 * lapse
)

# rhs.A_ij = advec.A_ij + Adot_TF_ij + A_ij (lapse (K - 2 Theta) - (2/d) divshift)
#            + A_ki d_j shift^k + A_kj d_i shift^k - 2 lapse h^{kl} A_ik A_lj
fn_A.add_eqn(
    A_rhs[li, lj],
    advec(A[li, lj]) + Adot_TF[li, lj]
    + A[li, lj] * (lapse * (K - 2 * Theta) - Rational(2, d) * divshift)
    + A[lk, li] * D(shift[uk], lj) + A[lk, lj] * D(shift[uk], li)
    - 2 * lapse * h[uk, ul] * A[li, lk] * A[ll, lj]
)

# rhs.Theta = ccz4_coeff [advec.Theta + 1/2 lapse (R - Aij_squared + ((d-1)/d) K^2 - 2 Theta K)
#             - 1/2 Theta kappa1_times_lapse ((d+1) + kappa2 (d-1)) - Z.d1lapse - lapse Lambda]
fn_A.add_eqn(
    Theta_rhs,
    ccz4_coeff * (
        advec(Theta)
        + Rational(1, 2) * lapse * (ricci_scalar - tr_A2 + Rational(d - 1, d) * K * K - 2 * Theta * K)
        - Rational(1, 2) * Theta * kappa1_times_lapse * ((d + 1) + kappa2 * (d - 1))
        - Z_dot_d1lapse
        - lapse * cosmological_constant
    )
)

# rhs.K = advec.K - tr_covd2lapse
#         + ccz4_coeff [lapse (R + K (K - 2 Theta)) - kappa1_times_lapse d (1 + kappa2) Theta - 2 lapse d / (d - 1) Lambda]
#         + bssn_coeff [lapse (Aij_squared + K^2 / d) - 2 lapse Lambda / (d - 1)]
fn_A.add_eqn(
    K_rhs,
    advec(K) - tr_covd2lapse
    + ccz4_coeff * (
        lapse * (ricci_scalar + K * (K - 2 * Theta))
        - kappa1_times_lapse * d * (1 + kappa2) * Theta
        - 2 * lapse * Rational(d, d - 1) * cosmological_constant
    )
    + bssn_coeff * (
        lapse * (tr_A2 + K * K / d)
        - 2 * lapse * cosmological_constant / (d - 1)
    )
)

# rhs.Gamma^i = advec.Gamma^i
#   + (2/d) (divshift (chris.contracted^i + 2 kappa3 Z_over_chi^i) - 2 lapse K Z_over_chi^i)
#   - 2 kappa1_times_lapse Z_over_chi^i
#   + sum_j [2 h^{ij} (lapse d_j Theta - Theta d_j lapse) - 2 A^{ij} d_j lapse
#            - lapse ((2 (d-1)/d) h^{ij} d_j K + d A^{ij} d_j chi / chi)
#            - (chris.contracted^j + 2 kappa3 Z_over_chi^j) d_j shift^i
#            + sum_k (2 lapse ULL[i][j][k] A^{jk} + h^{jk} d_j d_k shift^i + ((d-2)/d) h^{ij} d_j d_k shift^k)]
fn_A.add_eqn(
    Gamma_rhs[ui],
    advec(Gamma[ui])
    + Rational(2, d) * (divshift * (chris_c[ui] + 2 * kappa3 * Z_over_chi[ui]) - 2 * lapse * K * Z_over_chi[ui])
    - 2 * kappa1_times_lapse * Z_over_chi[ui]
    + 2 * h[ui, uj] * (lapse * D(Theta, lj) - Theta * D(lapse, lj))
    - 2 * A[ui, uj] * D(lapse, lj)
    - lapse * (Rational(2 * (d - 1), d) * h[ui, uj] * D(K, lj) + d * A[ui, uj] * D(chi, lj) / chi)
    - (chris_c[uj] + 2 * kappa3 * Z_over_chi[uj]) * D(shift[ui], lj)
    + 2 * lapse * chris[ui, lj, lk] * A[uj, uk]
    + h[uj, uk] * D(shift[ui], lj, lk)
    + Rational(d - 2, d) * h[ui, uj] * D(shift[uk], lj, lk)
)

###
# Kernel 3: MovingPunctureGauge::calculate_rhs, then CCZ4RHS::apply_dissipation (add_dissipation over all
# NUM_CCZ4_VARS = 25 variables). The B equation reads rhs.Gamma as kernel 2 wrote it, before dissipation.
###
fn_gauge = cc.create_function("gauge_and_dissipation")

# MovingPunctureGauge::compute_eta with Coordinates(IntVect(i, j, k), dx, center):
# x_a = (i_a + 1/2) dx - center_a, r = max(|x|, 1e-6) (Coordinates::get_radius),
# eta(r) = eta (eta_cutoff_coeff R^2 / (r^2 + R^2) + (1 - eta_cutoff_coeff)), R = eta_cutoff_radius.
r_raw = sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
# A temporary, so that r is squared as GRTeclyn squares it (sympy would fold max(sqrt(s), 1e-6)^2 into s).
radius = cc.decl("radius", [])
fn_gauge.add_eqn(radius, Piecewise((Float('1.0e-6'), r_raw < Float('1.0e-6')),  # type: ignore[no-untyped-call]
                                   (r_raw, True)))  # std::max(r, 1.0e-6_rt)
eta_cutoff_radius_squared = eta_cutoff_radius * eta_cutoff_radius
eta_of_x = eta * (eta_cutoff_coeff * (eta_cutoff_radius_squared / (radius * radius + eta_cutoff_radius_squared))
                  + (1 - eta_cutoff_coeff))

# rhs.lapse = lapse_advec_coeff advec.lapse - lapse_coeff lapse^lapse_power (K - 2 Theta), plus dissipation
fn_gauge.add_eqn(
    lapse_rhs,
    lapse_advec_coeff * advec(lapse) - lapse_coeff * lapse ** lapse_power * (K - 2 * Theta) + dissipation(lapse)
)
# rhs.shift^i = shift_advec_coeff advec.shift^i + shift_Gamma_coeff B^i, plus dissipation
fn_gauge.add_eqn(
    shift_rhs[ui],
    shift_advec_coeff * advec(shift[ui]) + shift_Gamma_coeff * B[ui] + dissipation(shift[ui])
)
# rhs.B^i = shift_advec_coeff advec.B^i - shift_advec_coeff advec.Gamma^i + rhs.Gamma^i - eta(r) B^i, plus dissipation
fn_gauge.add_eqn(
    B_rhs[ui],
    shift_advec_coeff * advec(B[ui]) - shift_advec_coeff * advec(Gamma[ui]) + Gamma_rhs[ui] - eta_of_x * B[ui]
    + dissipation(B[ui])
)

# Dissipation of the variables kernels 1 and 2 wrote: rhs.X += sigma sum_i KO_i X.
for var, var_rhs, idx in ((chi, chi_rhs, []), (h, h_rhs, [li, lj]), (K, K_rhs, []), (A, A_rhs, [li, lj]),
                          (Theta, Theta_rhs, []), (Gamma, Gamma_rhs, [ui])):
    var_rhs_diss = cc.overwrite(var_rhs)
    if idx:
        fn_gauge.add_eqn(var_rhs_diss[*idx], var_rhs[*idx] + dissipation(var[*idx]))
    else:
        fn_gauge.add_eqn(var_rhs_diss, var_rhs + dissipation(var))

# Bake options (e.g. CSE settings, a global ordering_fn) may come from a tuner; kernel 2's ordering functions are the
# split tuners' out-params, passed for that function only (an add_eqn_order refers to its add_eqn calls).
bake_options: dict[str, Any] = dict(get_optional_tuning_param('bake_options', dict()))
kernel2_orders: dict[str, Any] = {k: v for k in ('early_ordering_fn', 'pre_population_ordering_fn', 'ordering_fn')
                  if (v := get_optional_tuning_param(k, None)) is not None}
if kernel2_orders:
    bake_options['functions'] = {**bake_options.get('functions', dict()),
                                 "compute_A_ij_and_Theta_and_Gamma": kernel2_orders}
cc.bake(**bake_options)

AMReXWizard(cc, out_dir=args.out_dir).generate_header(
    banner=f"// Generated by EinsteinEngine from recipes/foreign/grteclyn/ccz4_grteclyn.py --fd-order {fd_order}.\n"
           f"// Drop-in replacement for the CCZ4 right-hand side of GRTeclyn's BinaryBH (CCZ4RHS and\n"
           f"// MovingPunctureGauge with FourthOrderDerivatives): EE::CCZ4RHS{fd_order}(dx).compute(rhs, state)."
)
