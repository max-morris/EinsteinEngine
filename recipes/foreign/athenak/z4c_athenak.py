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
The Z4c right-hand side with the puncture gauge, as AthenaK computes it, for AthenaK's Kokkos par_for.

The generated function z4c::ee::Z4cRHS<NGHOST>[SSL](MeshBlockPack *) replaces the two loops of
`Z4c::CalcRHS<NGHOST>` (src/z4c/z4c_calcrhs.cpp): the fused right-hand side ("z4c rhs loop") and the Kreiss-Oliger
dissipation ("K-O Dissipation"). It computes the same right-hand side in vacuum without drift control; CalcRHS keeps
its own loops for matter (ptmunu != nullptr) or drift control (the patch dispatches).

Sources. The formulation is Z4c [1, 2] as implemented in GR-Athena++ [3] and AthenaK [4]. Since the generated
kernel must agree with AthenaK to roundoff, AthenaK's implementation is the specification; each block below names
the AthenaK source it transcribes (paths in the AthenaK repository at commit c5a0d7f9), and
athenak/analysis/recipe-justification.md maps every term to [1-4]. The equations are those of Cottonmouth's Z4c.py
(recipes/Cottonmouth/Z4c.py) in AthenaK's variables (chi = psi^chi_psi_power, default -4, as Cottonmouth's chi), with
AthenaK's gauge options, its chi floor in divisions, its lopsided advection and its dissipation operator.

[1] S. Bernuzzi, D. Hilditch, Phys. Rev. D 81, 084003 (2010), arXiv:0912.2920.
[2] D. Hilditch et al., Phys. Rev. D 88, 084057 (2013), arXiv:1212.2901.
[3] B. Daszuta et al., Astrophys. J. Suppl. 257, 25 (2021), arXiv:2101.08289 (GR-Athena++).
[4] H. Zhu et al., arXiv:2409.10383 (2024) (AthenaK); src/z4c/z4c_calcrhs.cpp, src/z4c/z4c.cpp (options, diss),
    src/utils/finite_diff.hpp (stencils), src/coordinates/adm.hpp (SpatialDet, SpatialInv).

Usage: python z4c_athenak.py --nghost {2,3,4} [--ssl] [--diss {separate,fused,per-component}] [--out-dir DIR]
"""

import argparse
import sys
from math import comb

from typing import Any, Optional, cast

from sympy import Rational, Expr, Piecewise, Integer, Idx, IndexedBase
from nrpy.finite_difference import setup_FD_matrix__return_inverse_lowlevel

from EinsteinEngine import *

parser = argparse.ArgumentParser(description='AthenaK Z4c RHS slot-in kernel')
parser.add_argument('--nghost', type=int, default=4, choices=[2, 3, 4],
                    help='AthenaK ghost zones: 2, 3, 4 give 2nd, 4th, 6th order (CalcRHS<NGHOST>).')
parser.add_argument('--ssl', action='store_true',
                    help='Include the slow-start lapse term (z4c/slow_start_lapse = true).')
parser.add_argument('--diss', choices=['separate', 'fused', 'per-component'], default='separate',
                    help='separate: the dissipation is a second par_for over all 22 variables; fused: inside the '
                         'right-hand-side par_for; per-component: one par_for per variable component (22), each '
                         'thread doing one variable, as AthenaK\'s (m, n, k, j, i) loop does.')
parser.add_argument('--intermediates', choices=['temporaries', 'rules'], default='temporaries',
                    help='temporaries: the intermediate tensors AthenaK keeps as arrays (dphi, Ddphi, Rphi, Ddalpha, '
                         'AA_dd, DA, R, Ht and traces) are computed once per point (add_eqn), as AthenaK does; rules: '
                         'substituted into each use (sympy simplify()s each rule, which can reorder the arithmetic).')
parser.add_argument('--split', choices=['none', 'ricci', 'aux', 'ricci-aux'], default='none',
                    help='Hard splits of the right-hand-side loop (tuning): after the Ricci tensor (ricci), after all '
                         'intermediates (aux: R, Ddalpha, AA, DA, Ht), or both; values cross loops through a scratch view.')
parser.add_argument('--launch-bounds', default=None, metavar='THREADS,BLOCKS',
                    help='Launch the right-hand-side loops with Kokkos::LaunchBounds<THREADS, BLOCKS> (tuning).')
parser.add_argument('--cse-preset', choices=[p.value for p in CsePreset], default=None,
                    help="The global CSE's optimizations (CsePreset); default: EE's default (basic-collect-after).")
parser.add_argument('--ordering', choices=['reuse', 'rare', 'lexicographical', 'recipe'], default='reuse',
                    help='Post-population equation order: maximize_symbol_reuse (EE default), prioritize_rare_symbols, '
                         'lexicographical_order or recipe_order.')
parser.add_argument('--out-dir', default='.', help='Directory to write the header to.')
args = parser.parse_args(sys.argv[1:])

ng = args.nghost
width = 2 * ng - 1  # centered stencil width: order 2 * (ng - 1)
# The function name does not depend on --diss (the dispatch header calls Z4cRHS<ng>[SSL]); the par_for labels and
# the banner record the variant.
name = f"Z4cRHS{ng}{'SSL' if args.ssl else ''}"
variant = {'separate': '', 'fused': ' fused-diss', 'per-component': ' diss-per-component'}[args.diss]

kk = AthenaKKernel(
    name,
    namespace="z4c::ee",
    args=[("MeshBlockPack *", "pmy_pack")],
    preamble=[
        "auto &indcs = pmy_pack->pmesh->mb_indcs;",
        "auto &size = pmy_pack->pmb->mb_size;",
        "const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks, ke = indcs.ke;",
        "const int nmb = pmy_pack->nmb_thispack;",
        "auto &opt = pmy_pack->pz4c->opt;",
        "auto &u0 = pmy_pack->pz4c->u0;",
        "auto &u_rhs = pmy_pack->pz4c->u_rhs;",
    ],
    loop_bounds="0, nmb - 1, ks, ke, js, je, is, ie",
    inv_spacing=("1/size.d_view(m).dx1", "1/size.d_view(m).dx2", "1/size.d_view(m).dx3"),
    checks=[
        AthenaKCheck(f"indcs.ng != {ng}", f"generated for nghost = {ng}"),
        AthenaKCheck("pmy_pack->ptmunu != nullptr", "matter terms are not implemented"),
        AthenaKCheck("opt.enable_driftcontrol", "drift control is not implemented"),
        AthenaKCheck(f"opt.slow_start_lapse != {'true' if args.ssl else 'false'}",
                     f"generated for slow_start_lapse = {'true' if args.ssl else 'false'}"),
    ],
    includes=["athena.hpp", "mesh/mesh.hpp", "z4c/z4c.hpp"],
    scratch_extents=("nmb", "indcs.nx3 + 2 * indcs.ng", "indcs.nx2 + 2 * indcs.ng", "indcs.nx1 + 2 * indcs.ng"),
    derivative_stencil_width=width,
)

I = "z4c::Z4c::I_Z4C_"

###
# Parameters: Z4c::Options (src/z4c/z4c.cpp, Z4c::Z4c) and Z4c::diss.
###
chi_psi_power = kk.add_param("chi_psi_power", init="opt.chi_psi_power")
chi_div_floor = kk.add_param("chi_div_floor", init="opt.chi_div_floor")
damp_kappa1 = kk.add_param("damp_kappa1", init="opt.damp_kappa1")
damp_kappa2 = kk.add_param("damp_kappa2", init="opt.damp_kappa2")
lapse_oplog = kk.add_param("lapse_oplog", init="opt.lapse_oplog")
lapse_harmonicf = kk.add_param("lapse_harmonicf", init="opt.lapse_harmonicf")
lapse_harmonic = kk.add_param("lapse_harmonic", init="opt.lapse_harmonic")
lapse_advect = kk.add_param("lapse_advect", init="opt.lapse_advect")
shift_ggamma = kk.add_param("shift_ggamma", init="opt.shift_ggamma")
shift_advect = kk.add_param("shift_advect", init="opt.shift_advect")
shift_eta = kk.add_param("shift_eta", init="opt.shift_eta")
shift_alpha2ggamma = kk.add_param("shift_alpha2ggamma", init="opt.shift_alpha2ggamma")
shift_hh = kk.add_param("shift_hh", init="opt.shift_hh")
use_z4c = kk.add_param("use_z4c", init="opt.use_z4c ? 1.0 : 0.0")  # rhs.vTheta *= opt.use_z4c
# Z4c::diss = opt.diss * 2^(-2 ng) * (ng even ? -1 : 1) (z4c.cpp), the factor of the integer-weight stencils below.
diss = kk.add_param("diss", init="pmy_pack->pz4c->diss")
if args.ssl:
    chi_min_floor = kk.add_param("chi_min_floor", init="opt.chi_min_floor")
    ssl_damping_amp = kk.add_param("ssl_damping_amp", init="opt.ssl_damping_amp")
    ssl_damping_time = kk.add_param("ssl_damping_time", init="opt.ssl_damping_time")
    ssl_damping_index = kk.add_param("ssl_damping_index", init="opt.ssl_damping_index")
    time = kk.add_param("time", init="pmy_pack->pmesh->time")

###
# State (Z4c::Z4c_vars, views u0 and u_rhs, indices Z4c::I_Z4C_*).
###
def state(var: str, enum: str, indices: list[Idx], symmetric: bool = False) -> tuple[IndexedBase, IndexedBase]:
    if symmetric:
        sym = [(indices[0], indices[1])]
        return (kk.decl(var, indices, view="u0", enum=I + enum, symmetries=sym),
                kk.decl(f"{var}_rhs", indices, view="u_rhs", enum=I + enum, symmetries=sym))
    return (kk.decl(var, indices, view="u0", enum=I + enum),
            kk.decl(f"{var}_rhs", indices, view="u_rhs", enum=I + enum))


def pw(*pairs: tuple[object, object]) -> Expr:
    """A typed Piecewise."""
    return cast(Expr, Piecewise(*pairs))  # type: ignore[no-untyped-call]

chi, chi_rhs = state("chi", "CHI", [])                            # conformal factor, psi^chi_psi_power
g, g_rhs = state("g", "G", [li, lj], symmetric=True)              # conformal metric g_dd
Khat, Khat_rhs = state("Khat", "KHAT", [])                        # vKhat
A, A_rhs = state("A", "A", [li, lj], symmetric=True)              # vA_dd, trace-free conformal extrinsic curvature
Gam, Gam_rhs = state("Gam", "GAM", [ui])                          # vGam_u, conformal connection functions
Theta, Theta_rhs = state("Theta", "THETA", [])                    # vTheta
alpha, alpha_rhs = state("alpha", "ALPHA", [])                    # lapse
beta, beta_rhs = state("beta", "BETA", [ui])                      # shift

ue, le = kk.mk_pair("e")

###
# Stencils (src/utils/finite_diff.hpp, generated by AthenaK's py/write_FD.py).
###
# Dx is EE's centered stencil of width 2 ng - 1 (D(f, la)); Dxx and Dxy are d2 below. Check that
# their weights are AthenaK's.
_athenak_dx = {2: [Rational(1, 2)],
               3: [Rational(2, 3), Rational(-1, 12)],
               4: [Rational(3, 4), Rational(-3, 20), Rational(1, 60)]}  # weights at offsets +1, +2, +3
_athenak_dxx = {2: [-2, 1],
                3: [Rational(-5, 2), Rational(4, 3), Rational(-1, 12)],
                4: [Rational(-49, 18), Rational(3, 2), Rational(-3, 20), Rational(1, 90)]}  # offsets 0, 1, 2, 3
_fd = setup_FD_matrix__return_inverse_lowlevel(width, 0)
assert [_fd.col(1)[ng - 1 + k] for k in range(1, ng)] == _athenak_dx[ng], "EE's D(f, la) is not AthenaK's Dx"
assert [2 * _fd.col(2)[ng - 1 + k] for k in range(0, ng)] == _athenak_dxx[ng], "EE's D(f, la, la) is not AthenaK's Dxx"


def _stencil(den: int, weights: dict[int, int]) -> Expr:
    """1/den * sum_k w_k f(x + k h) * DDI(la), with integer weights w_k, grouped so the weights sum to zero exactly."""
    assert all(isinstance(w, int) for w in weights.values())
    total = sum(Integer(w) * stencil(la * k) for k, w in weights.items())
    return cast(Expr, Rational(1, den) * noop(total) * DDI(la))


# Second derivatives d2(f, la, lb), with AthenaK's grouping of the arithmetic (finite_diff.hpp: Dxx, Dxy), exact
# integer weights over one denominator. Dxx: the pairs f(+k) + f(-k) first. Dxy: for each pair of offsets (A, B) the
# differences across the second direction first, ((f(A,B) - f(A,-B)) - (f(-A,B) - f(-A,-B))), then the sum over (A, B).
# (EE's built-in D(f, la, lb) groups equal |weights|, f(A,B) + f(-A,-B) first: that sum of nearly equal values loses
# the cancellation's low bits; on level 1 it made the off-diagonal A_ij right-hand sides 10-20 times less accurate
# than AthenaK's, 2026-10-20.)
_dx_int = {2: (2, [1]), 3: (12, [8, -1]), 4: (60, [45, -9, 1])}  # Dx = (1/den) sum_k n_k (f(k) - f(-k)), k = 1..ng-1
_dxx_int = {2: (1, -2, [1]), 3: (12, -30, [16, -1]), 4: (180, -490, [270, -27, 2])}  # (1/den)(c0 f0 + sum n_k (f(k) + f(-k)))
_dden, _dn = _dx_int[ng]
assert [Rational(n, _dden) for n in _dn] == _athenak_dx[ng]
_xden, _x0, _xn = _dxx_int[ng]
assert [Rational(_x0, _xden)] + [Rational(n, _xden) for n in _xn] == [Rational(w) for w in _athenak_dxx[ng]]
_dxx_expr = Rational(1, _xden) * noop(
    sum(Integer(n) * noop(stencil(la * k) + stencil(-la * k)) for k, n in enumerate(_xn, 1))
    + Integer(_x0) * stencil(la * 0)) * DDI(la) ** 2
_dxy_expr = Rational(1, _dden ** 2) * noop(sum(
    Integer(na * nb) * noop(noop(stencil(la * a + lb * b) - stencil(la * a - lb * b))
                            - noop(stencil(-la * a + lb * b) - stencil(-la * a - lb * b)))
    for a, na in enumerate(_dn, 1) for b, nb in enumerate(_dn, 1))) * DDI(la) * DDI(lb)
d2 = kk.mk_stencil("d2", la, la, cast(Expr, _dxx_expr), la, lb, cast(Expr, _dxy_expr))

# Lx: lopsided first derivatives for the advection terms. dl (offsets -ng .. ng - 2) where the shift component is
# negative, dr (offsets -ng + 2 .. ng) otherwise. AthenaK's weights times den (finite_diff.hpp: Lx).
_lx_dl = {2: (2, {-2: 1, -1: -4, 0: 3}),
          3: (12, {-3: -1, -2: 6, -1: -18, 0: 10, 1: 3}),
          4: (60, {-4: 1, -3: -8, -2: 30, -1: -80, 0: 35, 1: 24, 2: -2})}
_dl_den, _dl_w = _lx_dl[ng]
# Check them against the one-point-lopsided stencils of the same width (nrpy, offsets -1 and +1).
for _off, _w in ((-1, _dl_w), (1, {-k: -w for k, w in _dl_w.items()})):
    _c = setup_FD_matrix__return_inverse_lowlevel(width, _off).col(1)
    assert {i - width // 2 + _off: _c[i] for i in range(width)} == {k: Rational(w, _dl_den) for k, w in _w.items()}, \
        "the Lx weights are not the lopsided first derivative"
adv_dl = kk.mk_stencil("adv_dl", la, _stencil(_dl_den, _dl_w))
adv_dr = kk.mk_stencil("adv_dr", la, _stencil(_dl_den, {-k: -w for k, w in _dl_w.items()}))

# Diss: the (2 ng)-th difference with binomial weights, times idx (finite_diff.hpp: Diss); z4c.cpp's diss carries
# opt.diss, the sign and 2^(-2 ng).
_diss_w = {k: (-1) ** (k + ng) * comb(2 * ng, ng + k) for k in range(-ng, ng + 1)}
diss_op = kk.mk_stencil("diss_op", la, _stencil(1, _diss_w))


def advect(f: Expr) -> Expr:
    """sum_a Lx(a, beta, f): beta^a dl_a f where beta^a < 0, else beta^a dr_a f (finite_diff.hpp: Lx)."""
    return sum(
        (pw((beta[ua_] * adv_dl(f, la_), beta[ua_] < 0), (beta[ua_] * adv_dr(f, la_), True))
         for ua_, la_ in ((u0, l0), (u1, l1), (u2, l2))),
        sympify(0)
    )


def dissipation(f: Expr) -> Expr:
    """diss * sum_a Diss(a, f) (z4c_calcrhs.cpp, the K-O Dissipation loop)."""
    return cast(Expr, diss * (diss_op(f, l0) + diss_op(f, l1) + diss_op(f, l2)))


###
# The right-hand side loop (z4c_calcrhs.cpp, "z4c rhs loop"). As in AthenaK, the inverse metric, the Christoffel
# symbols and the Ricci tensor are computed once per point (here as add_eqn temporaries), then the right-hand sides.
###
_lb = tuple(int(x) for x in args.launch_bounds.split(',')) if args.launch_bounds else None
if _lb is not None:
    variant += f' lb{_lb[0]}x{_lb[1]}'
if args.split != 'none':
    variant += f' split-{args.split}'
fn = kk.create_function("rhs", label=f"EE z4c rhs {name}{variant}",
                        launch_bounds=cast(Optional[tuple[int, int]], _lb))


def define(lhs: Any, rhs_: Any) -> None:
    """An intermediate AthenaK keeps in an array: a temporary (add_eqn) or a substitution rule (--intermediates)."""
    if args.intermediates == 'temporaries':
        fn.add_eqn(lhs, rhs_)
    else:
        kk.add_substitution_rule(lhs, rhs_)

# chi_guarded = max(chi, chi_div_floor), as chi > floor ? chi : floor. (A temporary: a Piecewise in a substitution
# rule would be folded into its neighbors by the rule's simplify().)
chi_g = kk.decl("chi_g", [])
fn.add_eqn(chi_g, pw((chi, chi > chi_div_floor), (chi_div_floor, True)))
p = chi_psi_power
# oopsi4 = pow(chi_guarded, -4 / chi_psi_power), once per point (a temporary: as a substitution rule, simplify()
# would merge it with the other powers of chi_guarded into several pow calls).
oopsi4 = kk.decl("oopsi4", [])
fn.add_eqn(oopsi4, chi_g ** (-4 / p))
# K = Khat + 2 Theta
K = kk.decl("K", [])
kk.add_substitution_rule(K, Khat + 2 * Theta)

# g_uu = SpatialInv(1/SpatialDet(g_dd), g_dd): the inverse of the evolved g_dd, not assuming det g = 1.
g_mat = kk.get_matrix(g[li, lj])
gu = kk.decl("gu", [ui, uj], symmetries=[(ui, uj)])
fn.add_eqn(gu[ui, uj], g_mat.adjugate() * (1 / g_mat.det()))  # type: ignore[no-untyped-call]
kk.add_substitution_rule(g[ui, uj], gu[ui, uj])

# Gamma_ddd(c,a,b) = 1/2 (d_a g_bc + d_b g_ac - d_c g_ab)   (dg_ddd(c,a,b) = d_c g_ab)
Gddd = kk.decl("Gddd", [lc, la, lb], symmetries=[(la, lb)])
kk.add_substitution_rule(Gddd[lc, la, lb], Rational(1, 2) * (D(g[lb, lc], la) + D(g[la, lc], lb) - D(g[la, lb], lc)))
# Gamma_udd(c,a,b) = g^cd Gamma_ddd(d,a,b)
Gudd = kk.decl("Gudd", [uc, la, lb], symmetries=[(la, lb)])
fn.add_eqn(Gudd[uc, la, lb], gu[uc, ud] * Gddd[ld, la, lb])
# Gamma_u(a) = g^bc Gamma_udd(a,b,c): the contracted Christoffel symbols of the metric (not the evolved Gam)
Gu = kk.decl("Gu", [ua])
fn.add_eqn(Gu[ua], gu[ub, uc] * Gudd[ua, lb, lc])

# R_dd(a,b) = 1/2 (g_ca d_b Gam^c + g_cb d_a Gam^c + Gamma_u(c) (Gamma_ddd(a,b,c) + Gamma_ddd(b,a,c)))
#           - 1/2 g^cd d_c d_d g_ab
#           + g^cd (Gamma_udd(e,c,a) Gamma_ddd(b,e,d) + Gamma_udd(e,c,b) Gamma_ddd(a,e,d)
#                   + Gamma_udd(e,a,d) Gamma_ddd(e,c,b))
Rdd = kk.decl("Rdd", [la, lb], symmetries=[(la, lb)])
fn.add_eqn(
    Rdd[la, lb],
    Rational(1, 2) * (g[lc, la] * D(Gam[uc], lb) + g[lc, lb] * D(Gam[uc], la)
                      + Gu[uc] * (Gddd[la, lb, lc] + Gddd[lb, la, lc]))
    - Rational(1, 2) * gu[uc, ud] * d2(g[la, lb], lc, ld)
    + gu[uc, ud] * (Gudd[ue, lc, la] * Gddd[lb, le, ld] + Gudd[ue, lc, lb] * Gddd[la, le, ld]
                    + Gudd[ue, la, ld] * Gddd[le, lc, lb])
)
if args.split in ('ricci', 'ricci-aux'):
    fn.split_loop()

# Derivatives of the conformal factor phi: dphi_d(a) = d_a chi / (chi_guarded chi_psi_power)
dphi = kk.decl("dphi", [la])
define(dphi[la], D(chi, la) / (chi_g * p))
# Ddphi_dd(a,b) = d_a d_b chi / (chi_guarded chi_psi_power) - chi_psi_power dphi_a dphi_b - Gamma_udd(c,a,b) dphi_c
Ddphi = kk.decl("Ddphi", [la, lb], symmetries=[(la, lb)])
define(Ddphi[la, lb], d2(chi, la, lb) / (chi_g * p) - p * dphi[la] * dphi[lb]
                         - Gudd[uc, la, lb] * dphi[lc])
# Rphi_dd(a,b) = 4 dphi_a dphi_b - 2 Ddphi_ab - 2 g_ab g^cd (Ddphi_cd + 2 dphi_c dphi_d)
# The contraction g^cd (...) is one scalar, as in AthenaK's loop, so that it is formed before it multiplies g_ab
# (expanded, its products with the small off-diagonal g_ab lose the cancellation's accuracy: level 1, 2026-10-20).
trRphi = kk.decl("trRphi", [])
define(trRphi, gu[uc, ud] * (Ddphi[lc, ld] + 2 * dphi[lc] * dphi[ld]))
Rphi = kk.decl("Rphi", [la, lb], symmetries=[(la, lb)])
define(Rphi[la, lb], 4 * dphi[la] * dphi[lb] - 2 * Ddphi[la, lb] - 2 * g[la, lb] * trRphi)

# Ddalpha_dd(a,b) = d_a d_b alpha - 2 (dphi_a d_b alpha + dphi_b d_a alpha) - Gamma_udd(c,a,b) d_c alpha
#                   + 2 g_ab g^cd dphi_c d_d alpha
dphi_dalpha = kk.decl("dphi_dalpha", [])  # g^cd dphi_c d_d alpha, one scalar as above
define(dphi_dalpha, gu[uc, ud] * dphi[lc] * D(alpha, ld))
Ddalpha = kk.decl("Ddalpha", [la, lb], symmetries=[(la, lb)])
define(
    Ddalpha[la, lb],
    d2(alpha, la, lb) - 2 * (dphi[la] * D(alpha, lb) + dphi[lb] * D(alpha, la)) - Gudd[uc, la, lb] * D(alpha, lc)
    + 2 * g[la, lb] * dphi_dalpha
)
# Ddalpha (trace) = oopsi4 g^ab Ddalpha_ab
trDdalpha = kk.decl("trDdalpha", [])
define(trDdalpha, oopsi4 * gu[ua, ub] * Ddalpha[la, lb])

# AA_dd(a,b) = g^cd A_ac A_db; AA = g^ab AA_ab; A_uu(a,b) = g^ac g^bd A_cd
AAdd = kk.decl("AAdd", [la, lb], symmetries=[(la, lb)])
define(AAdd[la, lb], gu[uc, ud] * A[la, lc] * A[ld, lb])
AA = kk.decl("AA", [])
define(AA, gu[ua, ub] * AAdd[la, lb])
kk.add_substitution_rule(A[ua, ub], gu[ua, uc] * gu[ub, ud] * A[lc, ld])

# DA_u(a) = -3/2 A_uu(a,b) d_b chi / chi_guarded - 1/3 g^ab (2 d_b Khat + d_b Theta) + Gamma_udd(a,b,c) A_uu(b,c)
DA = kk.decl("DA", [ua])
define(
    DA[ua],
    - Rational(3, 2) * A[ua, ub] * D(chi, lb) / chi_g
    - Rational(1, 3) * gu[ua, ub] * (2 * D(Khat, lb) + D(Theta, lb))
    + Gudd[ua, lb, lc] * A[ub, uc]
)

# R = oopsi4 g^ab (R_ab + Rphi_ab); Ht = R + 2/3 K^2 - AA
Rsc = kk.decl("Rsc", [])
define(Rsc, oopsi4 * gu[ua, ub] * (Rdd[la, lb] + Rphi[la, lb]))
Ht = kk.decl("Ht", [])
define(Ht, Rsc + Rational(2, 3) * K ** 2 - AA)
if args.split in ('aux', 'ricci-aux'):
    if args.intermediates != 'temporaries':
        raise SystemExit('--split aux needs --intermediates temporaries')
    fn.split_loop()

# dbeta = d_a beta^a; ddbeta_d(a) = 1/3 d_a d_b beta^b
dbeta = kk.decl("dbeta", [])
kk.add_substitution_rule(dbeta, D(beta[ua], la))
ddbeta = kk.decl("ddbeta", [la])
kk.add_substitution_rule(ddbeta[la], Rational(1, 3) * d2(beta[ub], la, lb))

# Lie derivatives along the shift (advection by Lx plus the finalization terms). These contain the Piecewise of the
# advection, so they are inlined (Python functions), not substitution rules.
def Lchi() -> Expr:
    return cast(Expr, advect(chi) + Rational(1, 6) * p * chi_g * dbeta)


def LGam(a: Idx, la_: Idx) -> Expr:
    """LGam_u(a) = Lx(Gam^a) + 2/3 Gamma_u(a) dbeta + g^ab ddbeta_b - Gamma_u(b) d_b beta^a + g^bc d_b d_c beta^a"""
    return cast(Expr, advect(Gam[a]) + Rational(2, 3) * Gu[a] * dbeta + gu[a, ub] * ddbeta[lb]
                - Gu[ub] * D(beta[a], lb) + gu[ub, uc] * d2(beta[a], lb, lc))


def Lg(a: Idx, b: Idx) -> Expr:
    """Lg_dd(a,b) = Lx(g_ab) - 2/3 g_ab dbeta + d_a beta^c g_bc + d_b beta^c g_ac   (dbeta_du(a,c) = d_a beta^c)"""
    return cast(Expr, advect(g[a, b]) - Rational(2, 3) * g[a, b] * dbeta + D(beta[uc], a) * g[b, lc]
                + D(beta[uc], b) * g[a, lc])


def LA(a: Idx, b: Idx) -> Expr:
    """LA_dd(a,b) = Lx(A_ab) - 2/3 A_ab dbeta + d_b beta^c A_ac + d_a beta^c A_bc"""
    return cast(Expr, advect(A[a, b]) - Rational(2, 3) * A[a, b] * dbeta + D(beta[uc], b) * A[a, lc]
                + D(beta[uc], a) * A[b, lc])


# With the dissipation fused into this loop, each right-hand side gets its dissipation term here.
fused = args.diss == 'fused'


def rhs(f: Expr, value: Expr) -> Expr:
    return value + dissipation(f) if fused else value


# rhs.vKhat = -Ddalpha + alpha (AA + 1/3 K^2) + LKhat + kappa1 (1 - kappa2) alpha Theta
fn.add_eqn(Khat_rhs, rhs(Khat, - trDdalpha + alpha * (AA + Rational(1, 3) * K ** 2) + advect(Khat)
                         + damp_kappa1 * (1 - damp_kappa2) * alpha * Theta))
# rhs.chi = Lchi - 1/6 chi_psi_power chi_guarded alpha K
fn.add_eqn(chi_rhs, rhs(chi, Lchi() - Rational(1, 6) * p * chi_g * alpha * K))
# rhs.vTheta = use_z4c (LTheta + alpha (1/2 Ht - (2 + kappa2) kappa1 Theta))
fn.add_eqn(Theta_rhs, rhs(Theta, use_z4c * (advect(Theta)
                                            + alpha * (Rational(1, 2) * Ht - (2 + damp_kappa2) * damp_kappa1 * Theta))))
# rhs.vGam_u(a) = 2 alpha DA_u(a) + LGam_u(a) - 2 alpha kappa1 (Gam^a - Gamma_u(a)) - 2 A_uu(a,b) d_b alpha
fn.add_eqn(Gam_rhs[ua], rhs(Gam[ua], 2 * alpha * DA[ua] + LGam(ua, la) - 2 * alpha * damp_kappa1 * (Gam[ua] - Gu[ua])
                            - 2 * A[ua, ub] * D(alpha, lb)))
# rhs.g_dd(a,b) = -2 alpha A_ab + Lg_ab
fn.add_eqn(g_rhs[la, lb], rhs(g[la, lb], - 2 * alpha * A[la, lb] + Lg(la, lb)))
# rhs.vA_dd(a,b) = oopsi4 (-Ddalpha_ab + alpha (R_ab + Rphi_ab)) - 1/3 g_ab (-Ddalpha + alpha R)
#                  + alpha (K A_ab - 2 AA_ab) + LA_ab
fn.add_eqn(
    A_rhs[la, lb],
    rhs(A[la, lb],
        oopsi4 * (- Ddalpha[la, lb] + alpha * (Rdd[la, lb] + Rphi[la, lb]))
        - Rational(1, 3) * g[la, lb] * (- trDdalpha + alpha * Rsc)
        + alpha * (K * A[la, lb] - 2 * AAdd[la, lb]) + LA(la, lb))
)
# rhs.alpha = lapse_advect Lalpha - f alpha Khat, f = lapse_oplog lapse_harmonicf + lapse_harmonic alpha
alpha_value = (lapse_advect * advect(alpha)
               - (lapse_oplog * lapse_harmonicf + lapse_harmonic * alpha) * alpha * Khat)
if args.ssl:
    # slow start lapse: += ssl_damping_amp (W - alpha) W^ssl_damping_index exp(-1/2 (time / ssl_damping_time)^2),
    # W = max(chi, chi_min_floor)^(1/2)
    W = pw((chi, chi > chi_min_floor), (chi_min_floor, True)) ** Rational(1, 2)
    alpha_value += (ssl_damping_amp * (W - alpha) * W ** ssl_damping_index
                    * exp(-Rational(1, 2) * (time / ssl_damping_time) ** 2))
fn.add_eqn(alpha_rhs, rhs(alpha, alpha_value))
# rhs.beta_u(a) = shift_ggamma Gam^a + shift_advect Lbeta^a - shift_eta beta^a + shift_alpha2ggamma alpha^2 Gam^a
#                 + sum_b shift_hh alpha chi_guarded (1/2 alpha d_b chi - d_b alpha) g^ab
fn.add_eqn(
    beta_rhs[ua],
    rhs(beta[ua],
        shift_ggamma * Gam[ua] + shift_advect * advect(beta[ua]) - shift_eta * beta[ua]
        + shift_alpha2ggamma * alpha ** 2 * Gam[ua]
        + shift_hh * alpha * chi_g * (Rational(1, 2) * alpha * D(chi, lb) - D(alpha, lb)) * gu[ua, ub])
)

###
# The dissipation loop (z4c_calcrhs.cpp, "K-O Dissipation"): u_rhs(n) += diss * sum_a Diss(a, u0(n)) for all 22
# variables.
###
_diss_vars = ((chi, chi_rhs, ()), (g, g_rhs, (li, lj)), (Khat, Khat_rhs, ()), (A, A_rhs, (li, lj)),
              (Gam, Gam_rhs, (ui,)), (Theta, Theta_rhs, ()), (alpha, alpha_rhs, ()), (beta, beta_rhs, (ui,)))
if args.diss == 'separate':
    fn_diss = kk.create_function("diss", label=f"EE z4c diss {name}")
    for var, var_rhs, idxs in _diss_vars:
        new = kk.overwrite(var_rhs)
        if idxs:
            fn_diss.add_eqn(new[idxs], var_rhs[idxs] + dissipation(var[idxs]))
        else:
            fn_diss.add_eqn(new, var_rhs + dissipation(var))
elif args.diss == 'per-component':
    _lo, _up = (l0, l1, l2), (u0, u1, u2)
    for var, var_rhs, idxs in _diss_vars:
        new = kk.overwrite(var_rhs)
        if not idxs:
            comps: list[tuple[Idx, ...]] = [()]
        elif len(idxs) == 1:
            comps = [(_up[a],) for a in range(3)]
        else:
            comps = [(_lo[a], _lo[b]) for a in range(3) for b in range(a, 3)]
        for c in comps:
            tag = str(var) + ''.join('xyz'[int(str(i)[-1])] for i in c)
            fc = kk.create_function(f"diss_{tag}", label=f"EE z4c diss {name} {tag}")
            if c:
                fc.add_eqn(new[c], var_rhs[c] + dissipation(var[c]))
            else:
                fc.add_eqn(new, var_rhs + dissipation(var))

_bake: dict[str, Any] = dict(get_optional_tuning_param('bake_options', dict()))
if args.cse_preset is not None:
    _bake['cse_optimizations'] = CsePreset(args.cse_preset)()
if args.ordering != 'reuse':
    _bake['ordering_fn'] = {'rare': prioritize_rare_symbols, 'lexicographical': lexicographical_order,
                            'recipe': recipe_order}[args.ordering]
kk.bake(**_bake)

AthenaKWizard(kk, out_dir=args.out_dir).generate_header(
    banner=f"// Generated by EinsteinEngine from recipes/foreign/athenak/z4c_athenak.py --nghost {ng}"
           f"{' --ssl' if args.ssl else ''} --diss {args.diss}"
           f"{'' if args.intermediates == 'temporaries' else ' --intermediates ' + args.intermediates}"
           f"{'' if args.split == 'none' else ' --split ' + args.split}"
           f"{'' if _lb is None else ' --launch-bounds ' + args.launch_bounds}"
           f"{'' if args.cse_preset is None else ' --cse-preset ' + args.cse_preset}"
           f"{'' if args.ordering == 'reuse' else ' --ordering ' + args.ordering}.\n"
           f"// Drop-in replacement for the loops of z4c::Z4c::CalcRHS<{ng}> (vacuum, no drift control)."
)
