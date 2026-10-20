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
The CCZ4 right-hand side of MHDuet 2.0 (Simflowny "Einstein + MHD + tabEoS", AMReX), as a recipe library.

`add_ccz4_mhduet(cc, fn, param)` declares MHDuet's spacetime state, the matter fields the spacetime right-hand side
reads, and its parameters on any EE DSL frontend `cc`, and adds to the function `fn` the interior right-hand side of
the 22 spacetime fields: the RHS that MHDuet's fused kernel computes at a point before its outer-boundary override,
its Kreiss-Oliger dissipation and its Runge-Kutta update; with `ko=True` it adds the dissipation too (MHDuet adds
it after the boundary override). A backend entry point (the slot-in) is separate.

Sources. The formulation is CCZ4 [1, 2] coupled to GRMHD, as in MHDuet [3]. Since the kernel must agree with MHDuet
to roundoff, MHDuet's generated code is the specification: `AdvanceLevel.cpp` (mhduet-2.0.tgz, md5
306d7cf12aba2078641cea92ed0750d8), the stencils in `Functions.H`, and the Simflowny model documentation shipped with
it (`documentation/PDEModel-*.xml`, `problem.pdf`). Each block names the Simflowny auxiliary variables it
transcribes (suffix `_p` in the code); mhduet/analysis/recipe-justification.md maps every term to [1-3].

[1] D. Alic, C. Bona-Casas, C. Bona, L. Rezzolla, C. Palenzuela, Phys. Rev. D 85, 064040 (2012), arXiv:1106.2254.
[2] D. Alic, W. Kastaun, L. Rezzolla, Phys. Rev. D 88, 064049 (2013), arXiv:1307.7391.
[3] C. Palenzuela et al., MHDuet, arXiv:2510.13965.

Names follow the code: chi, gtd (conformal metric), Atd (trace-free conformal extrinsic curvature), trK, theta,
Gamh (evolved conformal connection functions), Alpha, Betau. Matter inputs: the conserved Df, tauf, Sfd_i, Bfu^i (the
densitized fields of the evolved state) and the primitives vfd_i, pf (MHDuet's unSyncAuxFieldGroup).

Simflowny writes rational coefficients as decimal literals. Two of them are not the double nearest to the rational:
0.6666666666666667 is fl(2/3) + 1 ulp and 1.333333333333333 is fl(4/3) - 1 ulp. This recipe uses the exact
rationals; see mhduet/analysis/recipe-justification.md, "Decimal coefficients".
"""

from typing import Callable

from sympy import Abs, Expr, IndexedBase, Piecewise, Rational, exp, log, pi, sqrt

from EinsteinEngine import *

STATE = ["chi", "gtd", "Atd", "trK", "theta", "Gamh", "Alpha", "Betau"]
PARAMS = ["chi_floor", "p_kappa_z1", "p_kappa_z2", "p_kappa_cc", "p_kappa_p", "p_alpha_0", "p_damping_exp",
          "p_feta", "R_0", "eta_damping_exp", "lambda_0", "lambda_1", "lambda_2", "lambda_f0", "lambda_f1",
          "lambda_f2", "lambda_f3", "trK0", "Betau_x_0", "Betau_y_0", "Betau_z_0"]

DIRS = ((u0, l0), (u1, l1), (u2, l2))


def cmax(a: Expr, b: Expr) -> Expr:
    """std::max(a, b) = (a < b) ? b : a. (EE's index machinery takes Piecewise, not sympy's Max.)"""
    return Piecewise((b, a < b), (a, True))


def cmin(a: Expr, b: Expr) -> Expr:
    """std::min(a, b) = (b < a) ? b : a."""
    return Piecewise((b, b < a), (a, True))


def add_ccz4_mhduet(cc, fn, param: Callable[[str], Expr], ko: bool = False) -> dict[str, IndexedBase]:
    """Declare the fields and add the 22 interior right-hand sides to `fn`. `param(name)` returns the symbol of
    MHDuet parameter `name` (its frontend-specific declaration). With `ko`, each RHS also gets MHDuet's
    Kreiss-Oliger dissipation, dissipation_factor_<field> sum_a meshDissipation_a(<field>) (MHDuet adds it only when
    the factor is positive, and so does this; a factor <= 0 adds an exact zero). MHDuet adds it after its outer-boundary
    override, so whether it belongs in the generated function is the slot-in's choice.

    For the slot-in: MHDuet's coordinates are grid nodes, x = prob_lo + i dx (its `xcoord(i)`), not cell centers;
    the radial damping (D1 of the justification) depends on them. Returns the declared state and RHS symbols."""
    # D(f, ...) must be MHDuet's 4th-order centered stencils (D1CDO4, D2CDO4, D1CDO4crossed): width 5.
    assert cc.apply_div.n == 5, f"derivative stencil width {cc.apply_div.n}, MHDuet needs 5"
    ul, ll = cc.mk_pair("l")
    um, lm = cc.mk_pair("m")
    x, y, z = cc.mk_coords()
    p = {n: param(n) for n in PARAMS}

    def state(name: str, indices, symmetric: bool = False) -> tuple[IndexedBase, IndexedBase]:
        syms = dict(symmetries=[(indices[0], indices[1])]) if symmetric else dict()
        rhs = cc.decl(f"RHS_{name}", indices, **syms)
        return cc.decl(name, indices, rhs=rhs, **syms), rhs

    chi, chi_rhs = state("chi", [])
    gtd, gtd_rhs = state("gtd", [li, lj], symmetric=True)
    Atd, Atd_rhs = state("Atd", [li, lj], symmetric=True)
    trK, trK_rhs = state("trK", [])
    theta, theta_rhs = state("theta", [])
    Gamh, Gamh_rhs = state("Gamh", [ui])
    Alpha, Alpha_rhs = state("Alpha", [])
    Betau, Betau_rhs = state("Betau", [ui])

    # Matter inputs (read only).
    Df = cc.decl("Df", [])
    tauf = cc.decl("tauf", [])
    Sfd = cc.decl("Sfd", [li])
    Bfu = cc.decl("Bfu", [ui])
    vfd = cc.decl("vfd", [li])
    pf = cc.decl("pf", [])

    kdelta = cc.decl("kdelta", [la, lb], symmetries=[(la, lb)],
                     substitution_rule=mk_matrix([[1 if i == j else 0 for j in range(3)] for i in range(3)]))

    # Stencils (Functions.H). D(f, li) is D1CDO4, D(f, li, li) D2CDO4, D(f, li, lj) for i != j D1CDO4crossed
    # (D1CDO4 applied twice). lieforward/liebackward are the 4th-order lopsided stencils on points -1..3 and -3..1.
    adv_up = cc.mk_stencil("adv_up", la, finite_difference_stencil(4, 1, 1, la))
    adv_dn = cc.mk_stencil("adv_dn", la, finite_difference_stencil(4, 1, -1, la))

    def advec(f: Expr) -> Expr:
        """sum_a max(0, Betau^a) lieforward_a(f) + min(0, Betau^a) liebackward_a(f). max(0, b) and min(0, b) are
        written (b + |b|)/2 and (b - |b|)/2, which equal them exactly in floating point (b + |b| is 2b or 0, and
        halving is exact), and avoid one branch per direction."""
        return sum(Rational(1, 2) * (Betau[ua_] + Abs(Betau[ua_])) * adv_up(f, la_)
                   + Rational(1, 2) * (Betau[ua_] - Abs(Betau[ua_])) * adv_dn(f, la_) for ua_, la_ in DIRS)

    if ko:
        diss = cc.mk_stencil("diss", la, kreiss_oliger_stencil(5, la))

    def KO(f: Expr, factor: Expr) -> Expr:
        """MHDuet's `if (factor > 0.0) RHS += factor * sum_a meshDissipation_a(f)`, meshDissipation the 7-point sixth
        difference / (64 dx) per direction. The guard is kept: a factor <= 0 adds an exact zero."""
        return Piecewise((factor, 0 < factor), (0, True)) * (diss(f, l0) + diss(f, l1) + diss(f, l2)) if ko else 0

    def dfac(name: str) -> Expr:
        return param(f"dissipation_factor_{name}")

    # the six independent components of a symmetric tensor, with MHDuet's component suffixes
    SYM = [((l0, l0), "xx"), ((l0, l1), "xy"), ((l0, l2), "xz"), ((l1, l1), "yy"), ((l1, l2), "yz"), ((l2, l2), "zz")]

    # chi_max, inv_chi, sdetg
    chi_max = cc.decl("chi_max", [])
    fn.add_eqn(chi_max, cmax(p["chi_floor"], chi))
    inv_chi = cc.decl("inv_chi", [])
    fn.add_eqn(inv_chi, 1 / Abs(chi_max))
    sdetg = cc.decl("sdetg", [])
    fn.add_eqn(sdetg, chi_max ** Rational(-3, 2))

    # detgtd = |det gtd|; gtu = cofactor(gtd) / detgtd
    g_mat = cc.get_matrix(gtd[li, lj])
    detgtd = cc.decl("detgtd", [])
    fn.add_eqn(detgtd, Abs(g_mat.det()))
    gtu = cc.decl("gtu", [ui, uj], symmetries=[(ui, uj)])
    fn.add_eqn(gtu[ui, uj], g_mat.adjugate() * (1 / detgtd))
    cc.add_substitution_rule(gtd[ui, uj], gtu[ui, uj])

    # Ctd_kij = 1/2 (d_i gtd_kj + d_j gtd_ki - d_k gtd_ij); Ct^k_ij = gtu^kl Ctd_lij; Gamt^k = gtu^ij Ct^k_ij
    Ctd = cc.decl("Ctd", [lk, li, lj], symmetries=[(li, lj)])
    cc.add_substitution_rule(Ctd[lk, li, lj],
                             Rational(1, 2) * (D(gtd[lk, lj], li) + D(gtd[lk, li], lj) - D(gtd[li, lj], lk)))
    Ct = cc.decl("Ct", [uk, li, lj], symmetries=[(li, lj)])
    fn.add_eqn(Ct[uk, li, lj], gtu[uk, ul] * Ctd[ll, li, lj])
    Gamt = cc.decl("Gamt", [uk])
    fn.add_eqn(Gamt[uk], gtu[ui, uj] * Ct[uk, li, lj])

    # Zu^i = 1/2 chi_max (Gamh^i - Gamt^i)
    Zu = cc.decl("Zu", [ui])
    fn.add_eqn(Zu[ui], Rational(1, 2) * chi_max * (Gamh[ui] - Gamt[ui]))

    # Atud^i_j = gtu^ik Atd_kj; Atu^ij = Atud^i_k gtu^kj; trAt = Atud^i_i
    Atud = cc.decl("Atud", [ui, lj])
    fn.add_eqn(Atud[ui, lj], gtu[ui, uk] * Atd[lk, lj])
    Atu = cc.decl("Atu", [ui, uj], symmetries=[(ui, uj)])
    fn.add_eqn(Atu[ui, uj], Atud[ui, lk] * gtu[uk, uj])
    trAt = cc.decl("trAt", [])
    fn.add_eqn(trAt, Atud[ui, li])

    # div_Beta = d_k Betau^k; d_div_Beta_i = d_i d_k Betau^k
    div_Beta = cc.decl("div_Beta", [])
    fn.add_eqn(div_Beta, D(Betau[uk], lk))
    d_div_Beta = cc.decl("d_div_Beta", [li])
    cc.add_substitution_rule(d_div_Beta[li], D(Betau[uk], lk, li))

    # Rtd_ij = -1/2 gtu^lm d_l d_m gtd_ij + 1/2 (gtd_ki d_j Gamh^k + gtd_kj d_i Gamh^k)
    #          + 1/2 Gamh^k (Ctd_ijk + Ctd_jik) + gtu^lm (Ct^k_li Ctd_jkm + Ct^k_lj Ctd_ikm + Ct^k_im Ctd_klj)
    Rtd = cc.decl("Rtd", [li, lj], symmetries=[(li, lj)])
    fn.add_eqn(
        Rtd[li, lj],
        -Rational(1, 2) * gtu[ul, um] * D(gtd[li, lj], ll, lm)
        + Rational(1, 2) * (gtd[lk, li] * D(Gamh[uk], lj) + gtd[lk, lj] * D(Gamh[uk], li))
        + Rational(1, 2) * Gamh[uk] * (Ctd[li, lj, lk] + Ctd[lj, li, lk])
        + gtu[ul, um] * (Ct[uk, ll, li] * Ctd[lj, lk, lm] + Ct[uk, ll, lj] * Ctd[li, lk, lm]
                         + Ct[uk, li, lm] * Ctd[lk, ll, lj])
    )

    # Rpd_ij (the conformal-factor and Z part of chi R_ij):
    #   1/2 d_i d_j chi - 1/2 Ct^k_ij d_k chi + 1/2 gtd_ij (gtu^kl d_k d_l chi - Gamh^k d_k chi)
    #   - 1/4 inv_chi d_i chi d_j chi - 3/4 inv_chi gtd_ij gtu^kl d_k chi d_l chi
    #   + inv_chi (Zu^k gtd_ki d_j chi + Zu^k gtd_kj d_i chi)
    Rpd = cc.decl("Rpd", [li, lj], symmetries=[(li, lj)])
    fn.add_eqn(
        Rpd[li, lj],
        Rational(1, 2) * D(chi, li, lj) - Rational(1, 2) * Ct[uk, li, lj] * D(chi, lk)
        + Rational(1, 2) * gtd[li, lj] * (gtu[uk, ul] * D(chi, lk, ll) - Gamh[uk] * D(chi, lk))
        - Rational(1, 4) * inv_chi * D(chi, li) * D(chi, lj)
        - Rational(3, 4) * inv_chi * gtd[li, lj] * gtu[uk, ul] * D(chi, lk) * D(chi, ll)
        + inv_chi * Zu[uk] * (gtd[lk, li] * D(chi, lj) + gtd[lk, lj] * D(chi, li))
    )

    # Rscalar = gtu^ij (chi_max Rtd_ij + Rpd_ij)
    Rscalar = cc.decl("Rscalar", [])
    fn.add_eqn(Rscalar, gtu[ui, uj] * (chi_max * Rtd[li, lj] + Rpd[li, lj]))

    # Matter: T^{mu nu} from the conserved and primitive fields (Tf4u_*, Sfuu_*).
    Bfd = cc.decl("Bfd", [li])
    fn.add_eqn(Bfd[li], inv_chi * gtd[li, lj] * Bfu[uj])
    sqBf = cc.decl("sqBf", [])
    fn.add_eqn(sqBf, Bfu[ui] * Bfd[li])
    Sfu = cc.decl("Sfu", [ui])
    fn.add_eqn(Sfu[ui], chi_max * gtu[ui, uj] * Sfd[lj])
    vfu = cc.decl("vfu", [ui])
    fn.add_eqn(vfu[ui], chi_max * gtu[ui, uj] * vfd[lj])
    sqvf = cc.decl("sqvf", [])
    fn.add_eqn(sqvf, cmin(vfu[ui] * vfd[li], Rational(9999999999, 10000000000)))
    sqW = cc.decl("sqW", [])
    fn.add_eqn(sqW, 1 / (1 - sqvf))
    Bfvf = cc.decl("Bfvf", [])
    fn.add_eqn(Bfvf, Bfu[ui] * vfd[li])
    Sfuu = cc.decl("Sfuu", [ui, uj], symmetries=[(ui, uj)])
    fn.add_eqn(
        Sfuu[ui, uj],
        Rational(1, 2) * (Sfu[ui] * vfu[uj] + Sfu[uj] * vfu[ui]) + chi_max * gtu[ui, uj] * sdetg * pf
        - (-Rational(1, 2) * sqBf * chi_max * gtu[ui, uj] + Bfu[ui] * Bfu[uj]) / (sdetg * sqW)
        - Rational(1, 2) * Bfvf * (-Bfvf * chi_max * gtu[ui, uj] + Bfu[ui] * vfu[uj] + Bfu[uj] * vfu[ui]) / sdetg
    )
    Tutt = cc.decl("Tutt", [])
    fn.add_eqn(Tutt, (Df + tauf) / (sdetg * Alpha ** 2))
    Tut = cc.decl("Tut", [ui])
    fn.add_eqn(Tut[ui], Sfu[ui] / (sdetg * Alpha) - Betau[ui] * Tutt)
    Tu = cc.decl("Tu", [ui, uj], symmetries=[(ui, uj)])
    fn.add_eqn(Tu[ui, uj], Sfuu[ui, uj] / sdetg - Sfu[ui] * Betau[uj] / (sdetg * Alpha) - Betau[ui] * Tut[uj])

    # ADM sources: rho_ADM = Alpha^2 Tutt; Jtd_ADM_i = Alpha gtd_ij (Betau^j Tutt + Tut^j);
    # pTtd_ADM_ij = Betatd_i Betatd_j Tutt + Betatd_i gtd_jk Tut^k + Betatd_j gtd_ik Tut^k + gtd_ik gtd_jl Tu^kl;
    # tr_pT = inv_chi gtu^ij pTtd_ADM_ij
    Betatd = cc.decl("Betatd", [li])
    fn.add_eqn(Betatd[li], gtd[li, lj] * Betau[uj])
    rho_ADM = cc.decl("rho_ADM", [])
    fn.add_eqn(rho_ADM, Alpha ** 2 * Tutt)
    Jtd_ADM = cc.decl("Jtd_ADM", [li])
    fn.add_eqn(Jtd_ADM[li], Alpha * gtd[li, lj] * (Betau[uj] * Tutt + Tut[uj]))
    pTtd_ADM = cc.decl("pTtd_ADM", [li, lj], symmetries=[(li, lj)])
    fn.add_eqn(
        pTtd_ADM[li, lj],
        Betatd[li] * Betatd[lj] * Tutt + Betatd[li] * gtd[lj, lk] * Tut[uk] + Betatd[lj] * gtd[li, lk] * Tut[uk]
        + gtd[li, lk] * gtd[lj, ll] * Tu[uk, ul]
    )
    tr_pT = cc.decl("tr_pT", [])
    fn.add_eqn(tr_pT, inv_chi * gtu[ui, uj] * pTtd_ADM[li, lj])

    # Damping coefficients: decay_factor = min(1, (R_0 / max(1e-10, r))^eta_damping_exp);
    # kappa_X = min(p_kappa_X, p_kappa_X decay_factor) / Alpha (X = z1, p, cc); feta = min(p_feta, p_feta decay).
    r_floor = cc.decl("r_floor", [])  # a separate temporary: sympy folds nested Piecewise conditions into ITE
    fn.add_eqn(r_floor, cmax(Rational(1, 10 ** 10), sqrt(x ** 2 + y ** 2 + z ** 2)))
    decay = cc.decl("decay_factor", [])
    fn.add_eqn(decay, cmin(1, (p["R_0"] / r_floor) ** p["eta_damping_exp"]))
    kappa_z1 = cc.decl("kappa_z1", [])
    fn.add_eqn(kappa_z1, cmin(p["p_kappa_z1"], p["p_kappa_z1"] * decay) / Alpha)
    kappa_p = cc.decl("kappa_p", [])
    fn.add_eqn(kappa_p, cmin(p["p_kappa_p"], p["p_kappa_p"] * decay) / Alpha)
    kappa_cc = cc.decl("kappa_cc", [])
    fn.add_eqn(kappa_cc, cmin(p["p_kappa_cc"], p["p_kappa_cc"] * decay) / Alpha)
    feta = cc.decl("feta", [])
    fn.add_eqn(feta, cmin(p["p_feta"], p["p_feta"] * decay))
    kappa_z2 = p["p_kappa_z2"]
    # Gp = exp(-(Alpha / p_alpha_0)^p_damping_exp)
    Gp = cc.decl("Gp", [])
    fn.add_eqn(Gp, exp(-(Alpha / p["p_alpha_0"]) ** p["p_damping_exp"]))

    # Psi1_ij = -chi_max (d_i d_j Alpha - Ct^k_ij d_k Alpha) - 1/2 chi_max inv_chi (d_i Alpha d_j chi + d_j Alpha d_i chi)
    #           - 8 pi inv_chi Alpha pTtd_ADM_ij + Alpha Rpd_ij + chi_max Alpha Rtd_ij;
    # Psi1TF_ij = Psi1_ij - 1/3 (gtu^kl Psi1_kl) gtd_ij
    Psi1 = cc.decl("Psi1", [li, lj], symmetries=[(li, lj)])
    fn.add_eqn(
        Psi1[li, lj],
        -chi_max * D(Alpha, li, lj) + chi_max * Ct[uk, li, lj] * D(Alpha, lk)
        - Rational(1, 2) * chi_max * inv_chi * (D(Alpha, li) * D(chi, lj) + D(Alpha, lj) * D(chi, li))
        - 8 * pi * inv_chi * Alpha * pTtd_ADM[li, lj] + Alpha * Rpd[li, lj] + chi_max * Alpha * Rtd[li, lj]
    )
    trPsi1 = cc.decl("trPsi1", [])
    fn.add_eqn(trPsi1, gtu[ui, uj] * Psi1[li, lj])
    Psi1TF = cc.decl("Psi1TF", [li, lj], symmetries=[(li, lj)])
    cc.add_substitution_rule(Psi1TF[li, lj], Psi1[li, lj] - Rational(1, 3) * trPsi1 * gtd[li, lj])

    AtdAtu = cc.decl("AtdAtu", [])
    cc.add_substitution_rule(AtdAtu, Atd[li, lj] * Atu[ui, uj])

    ###
    # The right-hand sides (RHS_<field> in AdvanceLevel.cpp, before the boundary override and dissipation).
    ###
    # gtd_ij: gtd_ik d_j Betau^k + gtd_jk d_i Betau^k - kappa_p Alpha Gp (gtd_ij - delta_ij)
    #         - 1/3 kappa_cc Alpha gtd_ij log(detgtd) - 2 Alpha (Atd_ij - 1/3 lambda_0 trAt gtd_ij)
    #         - 2/3 gtd_ij div_Beta + advection
    for (i_, j_), c in SYM:
        fn.add_eqn(
            gtd_rhs[i_, j_],
            gtd[i_, lk] * D(Betau[uk], j_) + gtd[j_, lk] * D(Betau[uk], i_)
            - kappa_p * Alpha * Gp * (gtd[i_, j_] - kdelta[i_, j_])
            - Rational(1, 3) * kappa_cc * Alpha * gtd[i_, j_] * log(detgtd)
            - 2 * Alpha * (Atd[i_, j_] - Rational(1, 3) * trAt * p["lambda_0"] * gtd[i_, j_])
            - Rational(2, 3) * gtd[i_, j_] * div_Beta
            + advec(gtd[i_, j_])
            + (KO(gtd[i_, j_], dfac(f"gtd_{c}")) if ko else 0)
        )
    # Atd_ij: Atd_ik d_j Betau^k + Atd_jk d_i Betau^k - 1/3 kappa_cc Alpha gtd_ij trAt
    #         + Alpha (trK Atd_ij - 2 Atd_ik Atud^k_j) + Psi1TF_ij - 2/3 Atd_ij div_Beta + advection
    for (i_, j_), c in SYM:
        fn.add_eqn(
            Atd_rhs[i_, j_],
            Atd[i_, lk] * D(Betau[uk], j_) + Atd[j_, lk] * D(Betau[uk], i_)
            - Rational(1, 3) * kappa_cc * Alpha * gtd[i_, j_] * trAt
            + Alpha * (trK * Atd[i_, j_] - 2 * Atd[i_, lk] * Atud[uk, j_])
            + Psi1TF[i_, j_]
            - Rational(2, 3) * Atd[i_, j_] * div_Beta
            + advec(Atd[i_, j_])
            + (KO(Atd[i_, j_], dfac(f"Atd_{c}")) if ko else 0)
        )
    # Gamh^i (per component, for the per-component dissipation factor)
    for n, (ui_, _) in enumerate(DIRS):
        fn.add_eqn(
            Gamh_rhs[ui_],
            -Rational(4, 3) * Alpha * gtu[ui_, uj] * D(trK, lj)
            - Rational(2, 3) * Alpha * gtu[ui_, uj] * D(theta, lj)
            - 2 * theta * gtu[ui_, uj] * D(Alpha, lj)
            - Gamh[uj] * D(Betau[ui_], lj)
            + gtu[uj, uk] * D(Betau[ui_], lj, lk)
            - 2 * Atu[ui_, uj] * D(Alpha, lj)
            - 3 * Alpha * inv_chi * Atu[ui_, uj] * D(chi, lj)
            - 2 * Alpha * inv_chi * Zu[ui_] * (kappa_z1 + Rational(4, 3) * theta + Rational(2, 3) * trK)
            + Rational(1, 3) * gtu[ui_, uj] * d_div_Beta[lj]
            - 16 * pi * Alpha * inv_chi * gtu[ui_, uj] * Jtd_ADM[lj]
            + 2 * Alpha * Ct[ui_, lj, lk] * Atu[uj, uk]
            + Rational(2, 3) * Gamh[ui_] * div_Beta
            + advec(Gamh[ui_])
            + (KO(Gamh[ui_], dfac(f"Gamh_{'xyz'[n]}")) if ko else 0)
        )
    # Betau^i: (Betau_0^i - Betau^i) feta + 3/4 (Alpha lambda_f1 + lambda_f0) Gamh^i + lambda_2 advection
    Betau_0 = (p["Betau_x_0"], p["Betau_y_0"], p["Betau_z_0"])
    for n, (ua_, _) in enumerate(DIRS):
        fn.add_eqn(
            Betau_rhs[ua_],
            (Betau_0[n] - Betau[ua_]) * feta
            + Rational(3, 4) * (Alpha * p["lambda_f1"] + p["lambda_f0"]) * Gamh[ua_]
            + p["lambda_2"] * advec(Betau[ua_])
            + (KO(Betau[ua_], dfac(f"Betau_{'xyz'[n]}")) if ko else 0)
        )
    # Alpha: -2 (Alpha lambda_f3 + lambda_f2) Alpha (trK - trK0) + lambda_1 advection
    fn.add_eqn(
        Alpha_rhs,
        -2 * (Alpha * p["lambda_f3"] + p["lambda_f2"]) * Alpha * (trK - p["trK0"]) + p["lambda_1"] * advec(Alpha)
        + (KO(Alpha, dfac("Alpha")) if ko else 0)
    )
    # chi: 2/3 chi_max (Alpha (2 theta + trK) - div_Beta) + advection
    fn.add_eqn(chi_rhs, Rational(2, 3) * chi_max * (Alpha * (2 * theta + trK) - div_Beta) + advec(chi)
               + (KO(chi, dfac("chi")) if ko else 0))
    # trK
    fn.add_eqn(
        trK_rhs,
        -chi_max * gtu[ui, uj] * D(Alpha, li, lj)
        + chi_max * Gamt[uk] * D(Alpha, lk)
        + Rational(1, 2) * gtu[ui, uj] * D(Alpha, li) * D(chi, lj)
        + 2 * Zu[uk] * D(Alpha, lk)
        + Alpha * kappa_z1 * (1 - kappa_z2) * theta
        + Rational(1, 3) * Alpha * (2 * theta + trK) ** 2
        + Alpha * AtdAtu
        + 4 * pi * Alpha * (rho_ADM + tr_pT)
        + advec(trK)
        + (KO(trK, dfac("trK")) if ko else 0)
    )
    # theta
    fn.add_eqn(
        theta_rhs,
        -Zu[uk] * D(Alpha, lk)
        - Alpha * kappa_z1 * (2 + kappa_z2) * theta
        - 8 * pi * Alpha * rho_ADM
        + Rational(1, 2) * Alpha * (Rscalar + Rational(2, 3) * trK ** 2 + Rational(2, 3) * theta * (trK - 2 * theta)
                                    - AtdAtu)
        + advec(theta)
        + (KO(theta, dfac("theta")) if ko else 0)
    )
    return dict(chi=chi, gtd=gtd, Atd=Atd, trK=trK, theta=theta, Gamh=Gamh, Alpha=Alpha, Betau=Betau,
                RHS_chi=chi_rhs, RHS_gtd=gtd_rhs, RHS_Atd=Atd_rhs, RHS_trK=trK_rhs, RHS_theta=theta_rhs,
                RHS_Gamh=Gamh_rhs, RHS_Alpha=Alpha_rhs, RHS_Betau=Betau_rhs)

