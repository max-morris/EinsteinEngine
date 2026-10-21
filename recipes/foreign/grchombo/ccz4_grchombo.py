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
The CCZ4 right-hand side with the moving-puncture gauge, as GRChombo computes it, for GRChombo's BoxLoops.

The generated class EE::CCZ4RHS<order> (CCZ4RHS<order>.hpp) replaces
`CCZ4RHS<MovingPunctureGauge, FourthOrderDerivatives>` in a GRChombo level's specificEvalRHS: it has the same
constructor and computes the same right-hand side, including advection and Kreiss-Oliger dissipation.

Sources. The formulation is CCZ4 [1], with the conformal decomposition and modifications of [2] and [3] as implemented
in GRChombo [4, 5]. Since the generated kernel must agree with GRChombo to roundoff, GRChombo's implementation is the
specification; each block below names the GRChombo source it transcribes (paths in the GRChombo repository at commit
37e65952), and grchombo/analysis/recipe-justification.md maps every term to [1-3].

[1] D. Alic, C. Bona-Casas, C. Bona, L. Rezzolla, C. Palenzuela, Phys. Rev. D 85, 064040 (2012), arXiv:1106.2254.
[2] D. Alic, W. Kastaun, L. Rezzolla, Phys. Rev. D 88, 064049 (2013), arXiv:1307.7391.
[3] K. Clough et al., Class. Quantum Grav. 32, 245011 (2015), arXiv:1503.03436 (GRChombo).
[4] Source/CCZ4/CCZ4RHS.impl.hpp, Source/CCZ4/CCZ4Geometry.hpp, Source/CCZ4/MovingPunctureGauge.hpp,
    Source/utils/TensorAlgebra.hpp.
[5] Source/BoxUtils/FourthOrderDerivatives.hpp.

Scope: the CCZ4 formulation (GRChombo's formulation = USE_CCZ4, which the BinaryBH and KerrBH examples use); the
constructor rejects USE_BSSN. Both covariantZ4 settings and a cosmological constant are supported, as in GRChombo.

Usage: python ccz4_grchombo.py [--fd-order 4] [--out-dir DIR]
"""

import argparse
import sys
from typing import cast

from sympy import Add, Rational, Idx, IndexedBase, Expr, Abs, Integer

from EinsteinEngine import *

parser = argparse.ArgumentParser(description='GRChombo CCZ4 RHS slot-in kernel')
parser.add_argument('--fd-order', type=int, default=4, choices=[4], help='Finite-difference order (GRChombo: 4).')
parser.add_argument('--out-dir', default='.', help='Directory to write the header to.')
parser.add_argument('--mixed-d2', choices=['ee', 'nested'], default='ee',
                    help="Mixed second derivatives: EE's built-in grouping (default; the validated kernel), or the "
                         "differences across one direction first (a diagnostic variant).")
parser.add_argument('--split', choices=['none', 'chris', 'ricci', 'halves'], default='none',
                    help="PROBE (branch probe/grchombo-split): hard split the function into passes, class CCZ4RHS4Split "
                         "(chris: after h^ij and the Christoffels; ricci: after the Ricci block; halves: after the "
                         "chi, h, K and A right-hand sides).")
args = parser.parse_args(sys.argv[1:])

fd_order = args.fd_order
d = 3  # GR_SPACEDIM

cc = GRChomboComputeClass(
    f"CCZ4RHS{fd_order}" + ("Split" if args.split != 'none' else ""),
    namespace="EE",
    ctor_args=[
        GRChomboCtorArg("CCZ4_params_t<MovingPunctureGauge::params_t>", "a_params"),
        GRChomboCtorArg("double", "a_dx"),
        GRChomboCtorArg("double", "a_sigma"),
        GRChomboCtorArg("int", "a_formulation", "0"),
        GRChomboCtorArg("double", "a_cosmological_constant", "0"),
    ],
    dx="a_dx",
    ctor_checks=[
        GRChomboCtorCheck("a_formulation != 0", "only the CCZ4 formulation (USE_CCZ4) is implemented"),
    ],
    includes=["CCZ4RHS.hpp", "MovingPunctureGauge.hpp"],
    derivative_stencil_width=fd_order + 1,
)

ul, ll = cc.mk_pair("l")
um, lm = cc.mk_pair("m")
un, ln = cc.mk_pair("n")

###
# Parameters: CCZ4_params_t and MovingPunctureGauge::params_t fields, and the constructor's sigma and
# cosmological constant.
###
kappa1 = cc.add_param("kappa1", init="a_params.kappa1")
kappa2 = cc.add_param("kappa2", init="a_params.kappa2")
kappa3 = cc.add_param("kappa3", init="a_params.kappa3")
# 1 if covariantZ4 (kappa1 -> kappa1 / lapse), else 0.
covariantZ4 = cc.add_param("covariantZ4", init="a_params.covariantZ4 ? 1.0 : 0.0")
lapse_advec_coeff = cc.add_param("lapse_advec_coeff", init="a_params.lapse_advec_coeff")
lapse_power = cc.add_param("lapse_power", init="a_params.lapse_power")
lapse_coeff = cc.add_param("lapse_coeff", init="a_params.lapse_coeff")
shift_Gamma_coeff = cc.add_param("shift_Gamma_coeff", init="a_params.shift_Gamma_coeff")
shift_advec_coeff = cc.add_param("shift_advec_coeff", init="a_params.shift_advec_coeff")
eta = cc.add_param("eta", init="a_params.eta")
sigma = cc.add_param("sigma", init="a_sigma")
cosmological_constant = cc.add_param("cosmological_constant", init="a_cosmological_constant")

###
# State variables (Source/CCZ4/CCZ4UserVariables.hpp) and their right-hand sides.
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

kdelta = cc.decl("kdelta", [la, lb], symmetries=[(la, lb)],
                 substitution_rule=mk_matrix([[1 if i == j else 0 for j in range(3)] for i in range(3)]))

###
# Derivative operators (Source/BoxUtils/FourthOrderDerivatives.hpp). D(f, li) and D(f, li, lj) are EE's centered
# stencils of width fd_order + 1, which have FourthOrderDerivatives' diff1, diff2 and mixed_diff2 weights.
###
# advection_term: the advection of f along direction a, beta^a (D_a f), with GRChombo's lopsided stencils chosen by
# the sign of beta^a: the upwind one (points -1..3) where beta^a > 0, the downwind one (points -3..1) otherwise. It is
# written without a branch as beta+ D_up f + beta- D_down f, beta+- = (beta^a +- |beta^a|)/2: for beta^a > 0, beta+ =
# beta^a and beta- = 0 exactly, and vice versa, so each point gets exactly one of the two stencils, as in GRChombo.
# (A Piecewise here would make the global CSE extract the condition beta^a > 0 as a Boolean temporary, which EE does
# not support.)
adv_up = cc.mk_stencil("adv_up", la, finite_difference_stencil(fd_order, 1, 1, la))
if args.mixed_d2 == 'nested':
    # Diagnostic variant (Max, 2026-10-20): mixed second derivatives with the differences across the second direction
    # first, ((f(A,B) - f(A,-B)) - (f(-A,B) - f(-A,-B))), then the weighted sum over the offset pairs (A, B), as AthenaK's
    # Dxy. The weights are the same, (8, -1)/12 per direction (GRChombo's mixed_diff2: 64, 8, 1 over 144). EE's built-in
    # D(f, la, lb) adds equal-|weight| corners first. The diagonal keeps EE's built-in stencil: finite_difference_stencil
    # returns the Taylor coefficient f''/2 (it omits the factor ndivs!), so each term is doubled here, as D does.
    _den, _w = {4: (12, [8, -1])}[fd_order]
    _dxy = Rational(1, _den ** 2) * noop(sum(
        Integer(na * nb) * noop(noop(stencil(la * a + lb * b) - stencil(la * a - lb * b))
                                - noop(stencil(-la * a + lb * b) - stencil(-la * a - lb * b)))
        for a, na in enumerate(_w, 1) for b, nb in enumerate(_w, 1))) * DDI(la) * DDI(lb)
    _dxx = finite_difference_stencil(fd_order, 2, 0, la) / DDI(la) ** 2
    _dxx = Add(*(2 * t for t in _dxx.args)) * DDI(la) ** 2
    d2 = cc.mk_stencil("d2", la, la, cast(Expr, _dxx), la, lb, cast(Expr, _dxy))


def D2(f: Expr, i: Idx, j: Idx) -> Expr:
    """Second derivative: EE's D(f, i, j), or with --mixed-d2 nested the recipe's d2 (mixed terms grouped
    one direction first)."""
    return cast(Expr, d2(f, i, j) if args.mixed_d2 == 'nested' else D(f, i, j))


adv_down = cc.mk_stencil("adv_down", la, finite_difference_stencil(fd_order, 1, -1, la))
# dissipation_term: sigma / 64 * (f_{-3} - 6 f_{-2} + 15 f_{-1} - 20 f_0 + 15 f_1 - 6 f_2 + f_3) / dx per direction.
diss = cc.mk_stencil("diss", la, kreiss_oliger_stencil(fd_order + 1, la))


def advec(f: Expr) -> Expr:
    """FourthOrderDerivatives::advection: sum over directions a of beta^a (upwinded D_a f)."""
    return cast(Expr, sum(
        (Rational(1, 2) * (shift[ua_] + Abs(shift[ua_])) * adv_up(f, la_)
         + Rational(1, 2) * (shift[ua_] - Abs(shift[ua_])) * adv_down(f, la_))
        for ua_, la_ in ((u0, l0), (u1, l1), (u2, l2))
    ))


def dissipation(f: Expr) -> Expr:
    """FourthOrderDerivatives::add_dissipation(rhs, cell, m_sigma): sigma * sum_i KO_i f."""
    return cast(Expr, sigma * (diss(f, l0) + diss(f, l1) + diss(f, l2)))


###
# Geometry (TensorAlgebra.hpp: compute_inverse_sym, compute_christoffel; CCZ4Geometry::compute_ricci_Z).
###
# The compute function. Like GRChombo, it first computes h^{ij}, the Christoffel symbols and the Ricci tensor once
# (as local temporaries: variables that are both written and read here), then the right-hand sides.
fn = cc.create_function("compute")

# h^{ij}: the inverse of the evolved h_ij (compute_inverse_sym), not assuming det h = 1.
h_mat = cc.get_matrix(h[li, lj])
hUU = cc.decl("hUU", [ui, uj], symmetries=[(ui, uj)])
fn.add_eqn(hUU[ui, uj], h_mat.adjugate() * (1 / h_mat.det()))  # type: ignore[no-untyped-call]
cc.add_substitution_rule(h[ui, uj], hUU[ui, uj])

# chris.LLL[i][j][k] = 1/2 (d_k h_ji + d_j h_ki - d_i h_jk)
chris = cc.decl("chris", [li, lj, lk], symmetries=[(lj, lk)])
cc.add_substitution_rule(chris[li, lj, lk], Rational(1, 2) * (D(h[lj, li], lk) + D(h[lk, li], lj) - D(h[lj, lk], li)))
# chris.ULL[i][j][k] = h^{il} LLL[l][j][k]
chrisULL = cc.decl("chrisULL", [ui, lj, lk], symmetries=[(lj, lk)])
fn.add_eqn(chrisULL[ui, lj, lk], h[ui, ul] * chris[ll, lj, lk])
cc.add_substitution_rule(chris[ui, lj, lk], chrisULL[ui, lj, lk])
# chris_LLU[i][j][k] = h^{kl} LLL[i][j][l]
cc.add_substitution_rule(chris[li, lj, uk], h[uk, ul] * chris[li, lj, ll])
# chris.contracted[i] = h^{jk} ULL[i][j][k]
chris_c = cc.decl("chris_c", [ui])
fn.add_eqn(chris_c[ui], h[uj, uk] * chris[ui, lj, lk])
if args.split == 'chris':
    fn.split_loop()

# Z_over_chi[i] = 1/2 (Gamma^i - chris.contracted^i); Z^i = chi Z_over_chi^i
Z_over_chi = cc.decl("Z_over_chi", [ui])
cc.add_substitution_rule(Z_over_chi[ui], Rational(1, 2) * (Gamma[ui] - chris_c[ui]))

# covdtilde2chi[k][l] = d_k d_l chi - ULL[m][k][l] d_m chi
covdtilde2chi = cc.decl("covdtilde2chi", [lk, ll], symmetries=[(lk, ll)])
cc.add_substitution_rule(covdtilde2chi[lk, ll], D2(chi, lk, ll) - chris[um, lk, ll] * D(chi, lm))
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
    - Rational(1, 2) * h[uk, ul] * D2(h[li, lj], lk, ll)
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
fn.add_eqn(ricci[li, lj], (ricci_chi[li, lj] + chi * ricci_hat[li, lj] + z_terms[li, lj]) / chi)
ricci_scalar = cc.decl("ricci_scalar", [])
fn.add_eqn(ricci_scalar, chi * h[ui, uj] * ricci[li, lj])
if args.split == 'ricci':
    fn.split_loop()

###
# CCZ4RHS::rhs_equation (Source/CCZ4/CCZ4RHS.impl.hpp).
###
divshift = cc.decl("divshift", [])
cc.add_substitution_rule(divshift, D(shift[ui], li))
Z_dot_d1lapse = cc.decl("Z_dot_d1lapse", [])
cc.add_substitution_rule(Z_dot_d1lapse, chi * Z_over_chi[ui] * D(lapse, li))
dlapse_dot_dchi = cc.decl("dlapse_dot_dchi", [])
cc.add_substitution_rule(dlapse_dot_dchi, h[um, un] * D(lapse, lm) * D(chi, ln))

# covdtilde2lapse_kl = d_k d_l lapse - ULL[m][k][l] d_m lapse
# covd2lapse_kl = chi covdtilde2lapse_kl + 1/2 (d_k lapse d_l chi + d_k chi d_l lapse - h_kl dlapse_dot_dchi)
covd2lapse = cc.decl("covd2lapse", [lk, ll], symmetries=[(lk, ll)])
cc.add_substitution_rule(
    covd2lapse[lk, ll],
    chi * (D2(lapse, lk, ll) - chris[um, lk, ll] * D(lapse, lm))
    + Rational(1, 2) * (D(lapse, lk) * D(chi, ll) + D(chi, lk) * D(lapse, ll) - h[lk, ll] * dlapse_dot_dchi)
)

# tr_covd2lapse = -(d/2) dlapse_dot_dchi - chi chris.contracted^i d_i lapse
#                 + h^{ij} (chi d_i d_j lapse + d_i lapse d_j chi)
tr_covd2lapse = cc.decl("tr_covd2lapse", [])
cc.add_substitution_rule(
    tr_covd2lapse,
    -Rational(d, 2) * dlapse_dot_dchi
    - chi * chris_c[ui] * D(lapse, li)
    + h[ui, uj] * (chi * D2(lapse, li, lj) + D(lapse, li) * D(chi, lj))
)

# A_UU = raise_all(A, h_UU); tr_A2 = A_ij A^ij
cc.add_substitution_rule(A[ui, uj], h[ui, uk] * h[uj, ul] * A[lk, ll])
tr_A2 = cc.decl("tr_A2", [])
cc.add_substitution_rule(tr_A2, A[li, lj] * A[ui, uj])

# Adot_TF = trace-free part of (-covd2lapse_ij + chi lapse ricci_ij) (make_trace_free)
Adot = cc.decl("Adot", [li, lj], symmetries=[(li, lj)])
cc.add_substitution_rule(Adot[li, lj], -covd2lapse[li, lj] + chi * lapse * ricci[li, lj])
Adot_TF = cc.decl("Adot_TF", [li, lj], symmetries=[(li, lj)])
cc.add_substitution_rule(Adot_TF[li, lj], Adot[li, lj] - Rational(1, d) * h[li, lj] * (h[uk, ul] * Adot[lk, ll]))

# kappa1_times_lapse = kappa1 if covariantZ4 else kappa1 * lapse
kappa1_times_lapse = cc.decl("kappa1_times_lapse", [])
cc.add_substitution_rule(kappa1_times_lapse, kappa1 * (covariantZ4 + (1 - covariantZ4) * lapse))

# Gammadot^i (without advection), and rhs.Gamma^i = advec.Gamma^i + Gammadot^i, which the B equation uses before
# dissipation is added.
Gammadot = cc.decl("Gammadot", [ui])
cc.add_substitution_rule(
    Gammadot[ui],
    Rational(2, d) * (divshift * (chris_c[ui] + 2 * kappa3 * Z_over_chi[ui]) - 2 * lapse * K * Z_over_chi[ui])
    - 2 * kappa1_times_lapse * Z_over_chi[ui]
    + 2 * h[ui, uj] * (lapse * D(Theta, lj) - Theta * D(lapse, lj))
    - 2 * A[ui, uj] * D(lapse, lj)
    - lapse * (Rational(2 * (d - 1), d) * h[ui, uj] * D(K, lj) + d * A[ui, uj] * D(chi, lj) / chi)
    - (chris_c[uj] + 2 * kappa3 * Z_over_chi[uj]) * D(shift[ui], lj)
    + 2 * lapse * chris[ui, lj, lk] * A[uj, uk]
    + h[uj, uk] * D2(shift[ui], lj, lk)
    + Rational(d - 2, d) * h[ui, uj] * D2(shift[uk], lj, lk)
)
Gamma_rhs_nodiss = cc.decl("Gamma_rhs_nodiss", [ui])
cc.add_substitution_rule(Gamma_rhs_nodiss[ui], advec(Gamma[ui]) + Gammadot[ui])

###
# The right-hand sides: rhs_equation, MovingPunctureGauge::rhs_gauge, then add_dissipation over all 25 variables.
###

# rhs.chi = advec.chi + (2/d) chi (lapse K - divshift)
fn.add_eqn(chi_rhs, advec(chi) + Rational(2, d) * chi * (lapse * K - divshift) + dissipation(chi))

# rhs.h_ij = advec.h_ij - 2 lapse A_ij - (2/d) h_ij divshift + h_ki d_j shift^k + h_kj d_i shift^k
fn.add_eqn(
    h_rhs[li, lj],
    advec(h[li, lj]) - 2 * lapse * A[li, lj] - Rational(2, d) * h[li, lj] * divshift
    + h[lk, li] * D(shift[uk], lj) + h[lk, lj] * D(shift[uk], li)
    + dissipation(h[li, lj])
)

# rhs.K = advec.K + lapse (R + K (K - 2 Theta)) - kappa1_times_lapse d (1 + kappa2) Theta - tr_covd2lapse
#         - 2 lapse d / (d - 1) Lambda
fn.add_eqn(
    K_rhs,
    advec(K) + lapse * (ricci_scalar + K * (K - 2 * Theta))
    - kappa1_times_lapse * d * (1 + kappa2) * Theta - tr_covd2lapse
    - 2 * lapse * Rational(d, d - 1) * cosmological_constant
    + dissipation(K)
)

# rhs.A_ij = advec.A_ij + Adot_TF_ij + A_ij (lapse (K - 2 Theta) - (2/d) divshift)
#            + A_ki d_j shift^k + A_kj d_i shift^k - 2 lapse h^{kl} A_ik A_lj
fn.add_eqn(
    A_rhs[li, lj],
    advec(A[li, lj]) + Adot_TF[li, lj]
    + A[li, lj] * (lapse * (K - 2 * Theta) - Rational(2, d) * divshift)
    + A[lk, li] * D(shift[uk], lj) + A[lk, lj] * D(shift[uk], li)
    - 2 * lapse * h[uk, ul] * A[li, lk] * A[ll, lj]
    + dissipation(A[li, lj])
)
if args.split == 'halves':
    fn.split_loop()

# rhs.Theta = advec.Theta + 1/2 lapse (R - tr_A2 + ((d-1)/d) K^2 - 2 Theta K)
#             - 1/2 Theta kappa1_times_lapse ((d+1) + kappa2 (d-1)) - Z.d1lapse - lapse Lambda
fn.add_eqn(
    Theta_rhs,
    advec(Theta)
    + Rational(1, 2) * lapse * (ricci_scalar - tr_A2 + Rational(d - 1, d) * K * K - 2 * Theta * K)
    - Rational(1, 2) * Theta * kappa1_times_lapse * ((d + 1) + kappa2 * (d - 1))
    - Z_dot_d1lapse
    - lapse * cosmological_constant
    + dissipation(Theta)
)

# rhs.Gamma^i = advec.Gamma^i + Gammadot^i
fn.add_eqn(Gamma_rhs[ui], Gamma_rhs_nodiss[ui] + dissipation(Gamma[ui]))

# MovingPunctureGauge::rhs_gauge:
# rhs.lapse = lapse_advec_coeff advec.lapse - lapse_coeff lapse^lapse_power (K - 2 Theta)
fn.add_eqn(
    lapse_rhs,
    lapse_advec_coeff * advec(lapse) - lapse_coeff * lapse ** lapse_power * (K - 2 * Theta) + dissipation(lapse)
)
# rhs.shift^i = shift_advec_coeff advec.shift^i + shift_Gamma_coeff B^i
fn.add_eqn(
    shift_rhs[ui],
    shift_advec_coeff * advec(shift[ui]) + shift_Gamma_coeff * B[ui] + dissipation(shift[ui])
)
# rhs.B^i = shift_advec_coeff advec.B^i - shift_advec_coeff advec.Gamma^i + rhs.Gamma^i - eta B^i
fn.add_eqn(
    B_rhs[ui],
    shift_advec_coeff * advec(B[ui]) - shift_advec_coeff * advec(Gamma[ui]) + Gamma_rhs_nodiss[ui] - eta * B[ui]
    + dissipation(B[ui])
)

cc.bake(**get_optional_tuning_param('bake_options', dict()))

if args.split != 'none':
    from EinsteinEngine.wizards.grchombo_split_wizard import GRChomboSplitWizard
    GRChomboSplitWizard(cc, out_dir=args.out_dir).generate_header(
        banner=f"// Generated by EinsteinEngine from recipes/foreign/grchombo/ccz4_grchombo.py --fd-order {fd_order}"
               + (" --mixed-d2 nested" if args.mixed_d2 == 'nested' else "") + f" --split {args.split}.")
    sys.exit(0)
GRChomboWizard(cc, out_dir=args.out_dir).generate_header(
    banner=f"// Generated by EinsteinEngine from recipes/foreign/grchombo/ccz4_grchombo.py --fd-order {fd_order}"
           + (" --mixed-d2 nested" if args.mixed_d2 == 'nested' else "") + ".\n"
           "// Drop-in replacement for CCZ4RHS<MovingPunctureGauge, FourthOrderDerivatives> (CCZ4 formulation)."
)
