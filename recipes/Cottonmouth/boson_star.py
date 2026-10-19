#  Copyright (C) 2026 Steven R. Brandt and other Einstein Engine contributors.
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

import argparse
import functools
import sys
from pathlib import Path

from sympy import Rational

from EinsteinEngine import *
from boson_star_profile import install_bs_profile_table

"""
Complex-scalar-field ("boson star") matter thorn, generated to be a
computationally-comparable counterpart of Simflowny/MHDuet's mhduetBS-2.0
CCZ4 + Klein-Gordon matter model, so that a CarpetX+Cottonmouth boson-star
benchmark can be compared against it on a similar physics/complexity basis
(mirroring the way Z4c_upwind.py was created to make CottonmouthZ4c
comparable to STvAR's mixed upwind/centered derivatives).

See fairness.md sections 6-9 (repo root) for the gap analysis this revision
addresses: field count (mhduetBS-2.0's default bench input runs *two*
independent complex scalars, --nstars lets this thorn match that), and
metric coupling (--metric-thorn must point at a *matter-enabled* Cottonmouth
Z4c variant, e.g. CottonmouthZ4cUpwind4m, or the Tmunu this thorn writes is
computed but never read -- see the WARNING below).

This is a MATTER thorn, not a metric thorn: it does not evolve the
spacetime. Instead it:

  * evolves one or two first-order-reduced complex Klein-Gordon fields
    -- star 1: (phiR, phiI, piR, piI); star 2 (--nstars 2): (pheR, pheI,
    peR, peI), matching mhduetBS-2.0's own field names for its two
    independent, non-interacting complex scalars,
  * reads the target Z4c metric thorn's evolved state (chi, gt_ij, trK,
    Theta, evo_Gammat^i, evo_lapse, evo_shift^i) via `from_thorn`, and
  * WRITES TmunuBaseX::eTtt/eTti/eTij, mirroring the way CottonmouthZ4c*
    only ever READS those groups (see Z4c.py's "Matter terms" section).

WARNING -- metric coupling: this thorn writes Tmunu; whether the spacetime
RHS actually *reads* it back depends entirely on --metric-thorn. Pointing
this at a --vacuum-generated Cottonmouth Z4c thorn (e.g. the default
CottonmouthZ4cUpwind4v) means the metric's own trK_rhs/Theta_rhs/
evo_Gammat_rhs/AtTF equations have their matter terms multiplied by
Z4c.py's `use_matter_terms = 0` and algebraically eliminated at generation
time -- the Tmunu this thorn computes is then genuine, timed, but *unread*
work (confirmed empirically: z4c_rhs's own wall-clock is identical with and
without this thorn linked in). For an actually-coupled boson star, generate
and use a matter-enabled metric thorn instead, e.g.:
    python3 Z4c_upwind.py            # (no --vacuum) -> CottonmouthZ4cUpwind4m
    python3 boson_star.py --metric-thorn CottonmouthZ4cUpwind4m
The vacuum default is kept here only because it is what the existing
Hopper1 benchmark (report.md Sec.D) was built and run against; it is a
valid *matter-kernel-cost* measurement, just not a coupled-evolution one.

Potential: the Friedberg-Lee-Sirlin "solitonic" boson-star potential used
by mhduetBS-2.0 (AdvanceLevel.cpp, dVdphi2_p/sfVi_p), with mass m, soliton
scale sigma, and optional quartic self-coupling lambda -- shared by both
stars, matching mhduetBS-2.0 (each star's potential depends only on its own
field amplitude, but both use the same sf_mass/sf_sigma/sf_lambda):

    V(|phi|^2)    = m^2 |phi|^2 (1 - 2|phi|^2/sigma^2)^2 + (lambda/2) |phi|^4
    dV/d|phi|^2   = m^2 (1 - 2|phi|^2/sigma^2)^2
                    - 4 m^2 |phi|^2 (1 - 2|phi|^2/sigma^2)/sigma^2
                    + lambda |phi|^2

Evolution equations, per star (CCZ4-style reduction, matching mhduetBS-2.0's
AdvanceLevel.cpp / PDEModel...xml term-for-term, translated into Z4c
field names: chi<->chi, gtd_ij<->gt_ij, trK<->trK, theta<->Theta,
Gamh^i<->evo_Gammat^i, alpha<->evo_lapse, beta^i<->evo_shift^i):

    d_t phi = beta^i d_i phi - alpha Pi                      (advection: upwinded)

    d_t Pi  = beta^i d_i Pi                                  (advection: upwinded)
              + alpha (trK + 2 Theta) Pi
              + alpha dV/d|phi|^2 phi
              - alpha chi gt^{ij} D~_i D~_j phi               (conformal Laplacian, centered)
              + (1/2) alpha gt^{ij} d_i(chi) d_j(phi)         (product-rule remainder, centered)
              - chi gt^{ij} d_i(alpha) d_j(phi)               (lapse-gradient, MHDuet m_piR_o2_t17-25)
              - 2 alpha Z^i d_i phi                           (Z4 constraint-vector term, centered)

where D~_i D~_j phi = d_i d_j phi - Gammat^k_{ij} d_k phi is the conformal
covariant Hessian and Zu^i = (chi/2)(evo_Gammat^i - Gammatd^i) is MHDuet's
Z-vector (AdvanceLevel.cpp: Zu_x = 0.5*chi_max*(Gamh_x - Gamt_x)), both
recomputed locally from gt/evo_Gammat. Combined with the conformal
Laplacian this yields MHDuet's first-derivative coefficient
alpha*(chi*Gamt - 2*Zu)·∇phi. Gammat/Gammatd/Zvec are computed once and
shared between stars (mhduetBS-2.0 likewise computes its CCZ4
connection/Z-vector once per RHS call, not once per scalar field).

Stress-energy tensor: built as a genuine 4D tensor contraction (as
mhduetBS-2.0 does, rather than the cheaper direct-ADM rho/S_i/S_ij route),
one star at a time, then linearly superposed -- exactly mhduetBS-2.0's own
"Tu_munu = Tsfu_munu (star 1) + Tsfe_munu (star 2)" -- before being handed
off through TmunuBaseX:

    T_ab(star) = 2 Re(d_a phi* d_b phi) - g_ab (g^{cd} d_c phi* d_d phi + V(|phi|^2))
    eTtt/eTti/eTij = sum over active stars of T_ab(star)

For real fields phiR/phiI, 2 Re(d_a phi* d_b phi) = 2(d_a phiR d_b phiR + d_a phiI d_b phiI),
so no complex-number support is needed -- everything reduces to real bilinears.

Kreiss-Oliger dissipation is added to all evolved scalar fields (both
stars), same as the geometric sector, to match mhduetBS-2.0's structure
(which applies dissipation to its scalar fields too, even where the
shipped .input files set those dissipation factors to zero).

Radiative (NewRadX) BCs are applied to the scalar sector after
dissipation, matching mhduetBS-2.0's Sommerfeld conditions
(phiR/phiI/pheR/pheI falloff 2, asymptotic 0; piR/piI/peR/peI falloff 3,
asymptotic 0; propagation speed 1). The Hessian uses the same 4th-order
split as MHDuet (D2CDO4 on the diagonal, D1CDO4crossed off-diagonal):
Cottonmouth D(phi,li,lj) is nrpy's 4th-order centered 2nd / mixed stencil,
identical to Functions.H.

Kernel-launch count: the scalar RHS and the Tmunu contraction are emitted
as ONE scheduled function (split_loop() between equation groups, like
Z4c.py's own z4c_rhs), not two, since mhduetBS-2.0 computes both in the
same fat ParallelFor. Kreiss-Oliger dissipation is kept as its own function
scheduled after it, matching every other Cottonmouth recipe's convention
(Z4c.py/Z4c_upwind.py/bssnok.py all split dissipation out from the main
RHS the same way) rather than mhduetBS-2.0's single-kernel style.

NOTE on thorn coupling: like CottonmouthZ4c4m's real matter source (an
externally-supplied Tmunu, e.g. from AsterX), this thorn is NOT vacuum-safe
to combine with a *different* Cottonmouth Z4c variant than the one named by
--metric-thorn in the SAME build: EinsteinEngine gives every generated
thorn identical, non-namespaced C++ symbol names for a given recipe, so two
Z4c variants (say CottonmouthZ4c4m and CottonmouthZ4cUpwind4v) still cannot
coexist in one link -- see the multiple-definition issue hit earlier when
building the upwind benchmark. This thorn only depends on --metric-thorn's
*public* ADM-style state (chi, gt, trK, Theta, evo_Gammat, evo_lapse,
evo_shift), which is thorn-private code, not shared, so that specific
collision does not apply here; but the thorn's *name* still bakes in
--metric-thorn (and --nstars) so that pointing this recipe at a different
metric thorn, or a different star count, produces a distinctly-named,
coexistable thorn.
"""

################################
# BEGIN Generate Options
###

parser = argparse.ArgumentParser(
    prog='Cottonmouth Boson Star',
    description='A code generator for a complex-scalar-field (boson star) matter thorn, '
                 'sized/structured to be comparable to Simflowny/MHDuet\'s mhduetBS-2.0.'
)
parser.add_argument('--fd-order', type=int, default=4, help='Order of the finite difference equations to use.')
parser.add_argument(
    '--metric-thorn',
    type=str,
    default='CottonmouthZ4cUpwind4v',
    help='Name of the (already-generated) Cottonmouth Z4c thorn implementation whose '
         'chi/gt/trK/Theta/evo_Gammat/evo_lapse/evo_shift state this matter thorn reads. '
         'Use a matter-enabled variant (e.g. CottonmouthZ4cUpwind4m, generated without '
         '--vacuum) for the Tmunu this thorn writes to actually feed back into the metric '
         'RHS -- a --vacuum metric thorn (the default here) has its matter terms multiplied '
         'by 0 at generation time, so Tmunu is computed but never read. See fairness.md Sec.6.'
)
parser.add_argument(
    '--nstars', type=int, default=1, choices=[1, 2],
    help='Number of independent, non-interacting complex scalar fields (mhduetBS-2.0\'s '
         '"nstars"). mhduetBS-2.0\'s own default bench input runs 2 (8 scalar components: '
         'phi*/phe*/pi*/pe*); this thorn defaults to 1 (4 components) for backward '
         'compatibility with the existing Hopper1 benchmark. Use --nstars 2 to match the '
         'default mhduetBS-2.0 field count exactly (fairness.md Sec.6).'
)
pres = parser.parse_args(sys.argv[1:])

stencil_order = pres.fd_order
metric_thorn = pres.metric_thorn
nstars = pres.nstars

# Derive a short, distinct tag from the metric thorn name for this thorn's
# own name, e.g. "CottonmouthZ4cUpwind4v" -> "Upwind4v", "CottonmouthZ4c4v" -> "4v".
tag = metric_thorn
for prefix in ("CottonmouthZ4c", "Cottonmouth"):
    if tag.startswith(prefix):
        tag = tag[len(prefix):]
        break
if not tag:
    tag = metric_thorn

###
# END Generate Options
################################

###
# Thorn definition
###
# NOTE: Cactus thorn names are capped at 27 characters, so "BosonStar" is
# abbreviated to "BS" here (matching this repo's own MHDuetBS/"bs"
# shorthand). --nstars 2 gets its own distinct thorn name ("BS2...") since
# its field set differs from the --nstars 1 (default) thorn.
star_infix = "" if nstars == 1 else "2"
boson_star = ThornDef("Cottonmouth", f"CottonmouthBS{star_infix}{tag}", derivative_stencil_width=stencil_order + 1)

###
# Indices
###
ul, ll = boson_star.mk_pair("l")

###
# Kreiss-Oliger dissipation stencil (matches Z4c.py/Z4c_upwind.py)
###
div_diss = boson_star.mk_stencil(
    "div_diss",
    li,
    kreiss_oliger_stencil(stencil_order + 1, li)
)

###
# Upwind/downwind advection stencil (matches Z4c_upwind.py's Dupwind).
# Built below, once evo_shift has been declared via from_thorn.
###
kdelta = boson_star.mk_kdelta()

###
# Tensor parities
###
# fmt: off
parity_scalar = parities(+1, +1, +1)
# fmt: on

###
# Thorn parameters
###
sf_mass = boson_star.add_param(
    "sf_mass",
    default=1.0,
    desc="Klein-Gordon mass parameter m, shared by all stars (mhduetBS-2.0: parameters.p_sfmass)."
)

sf_sigma = boson_star.add_param(
    "sf_sigma",
    default=1.0e6,
    desc="Solitonic-potential scale sigma, shared by all stars; large sigma recovers the "
         "plain mini-boson-star potential V=m^2|phi|^2 (mhduetBS-2.0: parameters.sfsigma). "
         "Must be > 0."
)

sf_lambda = boson_star.add_param(
    "sf_lambda",
    default=1.0,
    desc="Quartic self-interaction coupling lambda, shared by all stars (mhduetBS-2.0: parameters.sflambda)."
)

dissipation_epsilon = boson_star.add_param(
    "dissipation_epsilon",
    default=0.32,
    desc="The amount of Kreiss-Oliger dissipation to add to the scalar sector."
)

apply_NewRadX = boson_star.add_param(
    "apply_NewRadX",
    default=False,
    desc="Apply NewRadX radiative BCs to the scalar fields (mhduetBS-2.0 Sommerfeld)."
)

radpower_phi = boson_star.add_param(
    "radpower_phi",
    default=2.0,
    desc="NewRadX radpower for phiR/phiI (and pheR/pheI). MHDuet phiR_falloff=2."
)

radpower_pi = boson_star.add_param(
    "radpower_pi",
    default=3.0,
    desc="NewRadX radpower for piR/piI (and peR/peI). MHDuet piR_falloff=3."
)

initial_amplitude = boson_star.add_param(
    "initial_amplitude",
    default=0.01,
    desc="Unused for star 1 (tabulated equilibrium profile). Kept for recipe "
         "compatibility; star-2 Gaussian amplitude when --nstars 2."
)

initial_width = boson_star.add_param(
    "initial_width",
    default=1.0,
    desc="Unused for star 1 (tabulated equilibrium profile). Kept for recipe "
         "compatibility; star-2 Gaussian width when --nstars 2."
)

if nstars == 2:
    initial_amplitude2 = boson_star.add_param(
        "initial_amplitude2",
        default=0.01,
        desc="Amplitude of the trivial Gaussian-bump initial data used for benchmarking. Star 2."
    )

    initial_width2 = boson_star.add_param(
        "initial_width2",
        default=1.0,
        desc="Width of the trivial Gaussian-bump initial data used for benchmarking. Star 2."
    )

###
# Metric thorn state (read-only): chi, gt_ij, trK, Theta, evo_Gammat^i,
# evo_lapse, evo_shift^i, from the named Cottonmouth Z4c variant.
###
chi = boson_star.decl("chi", [], from_thorn=metric_thorn)

gt = boson_star.decl(
    "gt",
    [li, lj],
    symmetries=[(li, lj)],
    from_thorn=metric_thorn
)

trK = boson_star.decl("trK", [], from_thorn=metric_thorn)

Theta = boson_star.decl("Theta", [], from_thorn=metric_thorn)

evo_Gammat = boson_star.decl("evo_Gammat", [ui], from_thorn=metric_thorn)

evo_lapse = boson_star.decl("evo_lapse", [], from_thorn=metric_thorn)

evo_shift = boson_star.decl("evo_shift", [ui], from_thorn=metric_thorn)

###
# Now that evo_shift exists, build the real Dupwind stencil.
###
Dupwind = boson_star.mk_stencil(
    "Dupwind",
    la,
    h_step(evo_shift[ub] * kdelta[la, lb]) * finite_difference_stencil(stencil_order, 1, 1, la) +
    h_step(-evo_shift[ub] * kdelta[la, lb]) * finite_difference_stencil(stencil_order, 1, -1, la)
)

###
# TmunuBaseX vars (write side; basenames match Z4c.py's read side exactly
# so both land on the same TmunuBaseX::eTtt/eTti/eTij groups).
###
eTtt = boson_star.decl("eTtt", [], from_thorn="TmunuBaseX")

eTti = boson_star.decl("eTt", [li], from_thorn="TmunuBaseX")

eTij = boson_star.decl(
    "eT",
    [li, lj],
    symmetries=[(li, lj)],
    from_thorn="TmunuBaseX"
)

###
# Shared aux vars (local, non-persistent -- like Z4c.py's Gammat/Gammatd,
# these are recomputed fresh every RHS call, matching mhduetBS-2.0's own
# style). Computed once and reused by every star, same as mhduetBS-2.0
# computes its connection/Z-vector once per RHS call, not once per field.
###
Gammat = boson_star.decl("Gammat", [li, lj, lk], symmetries=[(lj, lk)])
Gammatd = boson_star.decl("Gammatd", [ui])
Zvec = boson_star.decl("Zvec", [ui])

###
# Substitution rules (metric-only, shared by all stars)
###
gt_mat = boson_star.get_matrix(gt[li, lj])
detgt = det(gt_mat)
gt_imat = inv(gt_mat) * detgt
boson_star.add_substitution_rule(gt[ui, uj], gt_imat)

# Conformal Christoffels, same construction as Z4c.py.
boson_star.add_substitution_rule(
    Gammat[lk, li, lj],
    Rational(1, 2) * (
        D(gt[lj, lk], li) + D(gt[li, lk], lj) - D(gt[li, lj], lk)
    )
)

boson_star.add_substitution_rule(
    Gammat[uk, li, lj], gt[uk, ul] * Gammat[ll, li, lj]
)

boson_star.add_substitution_rule(
    Gammatd[ui], gt[uj, uk] * Gammat[ui, lj, lk]
)

# MHDuet Zu^i = (chi/2)(Gamh^i - Gamt^i). Recomputed locally rather than
# read from Z4c (ZtCons is analysis-only and is the undensitized Z^i
# without chi). The -2*alpha*Zvec·∇phi term below then matches
# AdvanceLevel.cpp's alpha*(chi*Gamt - 2*Zu)·∇phi first-derivative piece.
boson_star.add_substitution_rule(
    Zvec[ui],
    Rational(1, 2) * chi * (evo_Gammat[ui] - Gammatd[ui])
)

###
# Aux. groups
###
initial_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthBosonStar_InitialGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_Initial"),
    after=[Identifier("ADMBaseX_PostInitial")],
    description=String("Boson star scalar-field initialization")
)

rhs_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthBosonStar_RHSGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("ODESolvers_RHS"),
    before=[Identifier("z4c_rhs")],
    description=String("Boson star scalar-field + Tmunu RHS computation"),
)

x, y, z = boson_star.mk_coords()

# NewRadX batches collected in add_star() and passed to the generator.
nrx_batches = []
r2 = x**2 + y**2 + z**2
r = sqrt(r2)

# Tabulated equilibrium profile (bs_profile_table.hxx). Undefined
# Function calls are emitted as C calls of these names.
bs_profile_phi = boson_star.decl_fun("bs_profile_phi", args=1, is_stencil=False)
bs_profile_alpha = boson_star.decl_fun("bs_profile_alpha", args=1, is_stencil=False)
# Must match BS_PROFILE_OMEGA in bs_profile_table.hxx.
bs_profile_omega = sympify(1.0666123628616333)

fun_initial = boson_star.create_function(
    "boson_star_initial",
    initial_group,
    intent_override=IntentOverride.WriteInterior
)

fun_bs_rhs = boson_star.create_function(
    "boson_star_rhs",
    rhs_group
)

fun_bs_diss = boson_star.create_function(
    "boson_star_apply_dissipation",
    rhs_group,
    schedule_after=["boson_star_rhs"]
)

###
# Per-star field declarations + equations. One call per active star; each
# star is a fully independent complex Klein-Gordon field (own phi/Pi, own
# |phi|^2/V), sharing only the metric-derived Gammat/Gammatd/Zvec above and
# the sf_mass/sf_sigma/sf_lambda potential parameters -- exactly
# mhduetBS-2.0's "linear superposition of independent stars" model.
# Returns (phiR, phiI, dtphiR, dtphiI, Vpot) for the Tmunu contraction below.
###
def add_star(label, name_phiR, name_phiI, name_piR, name_piI, amp_param, width_param):
    phiR_rhs = boson_star.decl(f"{name_phiR}_rhs", [], parity=parity_scalar)
    phiR = boson_star.decl(name_phiR, [], rhs=phiR_rhs, parity=parity_scalar)

    phiI_rhs = boson_star.decl(f"{name_phiI}_rhs", [], parity=parity_scalar)
    phiI = boson_star.decl(name_phiI, [], rhs=phiI_rhs, parity=parity_scalar)

    piR_rhs = boson_star.decl(f"{name_piR}_rhs", [], parity=parity_scalar)
    piR = boson_star.decl(name_piR, [], rhs=piR_rhs, parity=parity_scalar)

    piI_rhs = boson_star.decl(f"{name_piI}_rhs", [], parity=parity_scalar)
    piI = boson_star.decl(name_piI, [], rhs=piI_rhs, parity=parity_scalar)

    Xsq = boson_star.decl(f"Xsq_{label}", [])
    dVdX = boson_star.decl(f"dVdX_{label}", [])
    Vpot = boson_star.decl(f"Vpot_{label}", [])

    # |phi|^2 and the solitonic Friedberg-Lee-Sirlin potential (mhduetBS-2.0
    # AdvanceLevel.cpp: dVdphi2_p, sfVi_p), per-star.
    boson_star.add_substitution_rule(Xsq, phiR**2 + phiI**2)

    boson_star.add_substitution_rule(
        dVdX,
        sf_mass**2 * (1 - 2 * Xsq / sf_sigma**2)**2
        - 4 * sf_mass**2 * Xsq * (1 - 2 * Xsq / sf_sigma**2) / sf_sigma**2
        + sf_lambda * Xsq
    )

    boson_star.add_substitution_rule(
        Vpot,
        sf_mass**2 * Xsq * (1 - 2 * Xsq / sf_sigma**2)**2
        + Rational(1, 2) * sf_lambda * Xsq**2
    )

    # Star 1: tabulated static solitonic equilibrium (same table as
    # CottonmouthBosonStarID). Star 2, if present, stays a Gaussian bump
    # for the two-star kernel-cost benchmark.
    if label == "1":
        fun_initial.add_eqn(phiR, bs_profile_phi(r))
        fun_initial.add_eqn(phiI, sympify(0))
        fun_initial.add_eqn(piR, sympify(0))
        fun_initial.add_eqn(piI, bs_profile_omega * bs_profile_phi(r) / bs_profile_alpha(r))
    else:
        fun_initial.add_eqn(phiR, amp_param * exp(-r2 / width_param**2))
        fun_initial.add_eqn(phiI, sympify(0))
        fun_initial.add_eqn(piR, sympify(0))
        fun_initial.add_eqn(piI, sympify(0))

    # phi evolution (advection: upwinded, matching mhduetBS-2.0's shift-transport terms).
    fun_bs_rhs.add_eqn(
        phiR_rhs,
        evo_shift[ui] * Dupwind(phiR, li) - evo_lapse * piR
    )

    fun_bs_rhs.add_eqn(
        phiI_rhs,
        evo_shift[ui] * Dupwind(phiI, li) - evo_lapse * piI
    )

    # d_t phi again, as separate non-state aux vars, for the Tmunu
    # contraction further down in this SAME function. Do NOT reuse
    # phiR_rhs/phiI_rhs themselves for this: a state var (rhs=-linked) that
    # gets read again later in the function that writes it is demoted by
    # the generator to a tile-local temporary instead of a genuine
    # persistent grid-function write (observed as CarpetX "invalid grid
    # function" aborts on phiR_rhs/pheR_rhs -- piR_rhs/peR_rhs were fine,
    # since nothing reads them again within this function). CSE dedupes
    # the duplicated formula, so this costs nothing extra.
    dtphiR = boson_star.decl(f"dtphi{label}R", [])
    dtphiI = boson_star.decl(f"dtphi{label}I", [])
    fun_bs_rhs.add_eqn(dtphiR, evo_shift[ui] * Dupwind(phiR, li) - evo_lapse * piR)
    fun_bs_rhs.add_eqn(dtphiI, evo_shift[ui] * Dupwind(phiI, li) - evo_lapse * piI)

    fun_bs_rhs.split_loop()

    # Pi evolution: shift advection (upwinded) + CCZ4-style Klein-Gordon
    # source (conformal Laplacian + Christoffel/Z-vector terms +
    # product-rule remainder, all centered -- matches mhduetBS-2.0's
    # D1CDO4/D2CDO4 usage).
    fun_bs_rhs.add_eqn(
        piR_rhs,
        evo_shift[ui] * Dupwind(piR, li)
        + evo_lapse * (trK + 2 * Theta) * piR
        + evo_lapse * dVdX * phiR
        - evo_lapse * chi * gt[ui, uj] * (
            D(phiR, li, lj) - Gammat[uk, li, lj] * D(phiR, lk)
        )
        + Rational(1, 2) * evo_lapse * gt[ui, uj] * D(chi, li) * D(phiR, lj)
        # MHDuet m_piR_o2_t17-25: -chi * gtu^{ij} * d_i(Alpha) * d_j(phi).
        # From (1/alpha) D_i(alpha D^i phi); largest at the r~2 shell where
        # both grad(alpha) and grad(phi) are steep. Missing this left our
        # |phi|^2 spreading at that shell while MHDuet's stayed frozen.
        - chi * gt[ui, uj] * D(evo_lapse, li) * D(phiR, lj)
        - 2 * evo_lapse * Zvec[ui] * D(phiR, li)
    )

    fun_bs_rhs.add_eqn(
        piI_rhs,
        evo_shift[ui] * Dupwind(piI, li)
        + evo_lapse * (trK + 2 * Theta) * piI
        + evo_lapse * dVdX * phiI
        - evo_lapse * chi * gt[ui, uj] * (
            D(phiI, li, lj) - Gammat[uk, li, lj] * D(phiI, lk)
        )
        + Rational(1, 2) * evo_lapse * gt[ui, uj] * D(chi, li) * D(phiI, lj)
        - chi * gt[ui, uj] * D(evo_lapse, li) * D(phiI, lj)
        - 2 * evo_lapse * Zvec[ui] * D(phiI, li)
    )

    fun_bs_rhs.split_loop()

    # Kreiss-Oliger dissipation on this star's four evolved fields (matches
    # mhduetBS-2.0, which applies dissipation to its scalar fields too,
    # even where the shipped .input files zero out those factors).
    phiR_rhs_diss = boson_star.overwrite(phiR_rhs)
    fun_bs_diss.add_eqn(
        phiR_rhs_diss,
        phiR_rhs + dissipation_epsilon * (
            + div_diss(phiR, l0) + div_diss(phiR, l1) + div_diss(phiR, l2)
        )
    )

    phiI_rhs_diss = boson_star.overwrite(phiI_rhs)
    fun_bs_diss.add_eqn(
        phiI_rhs_diss,
        phiI_rhs + dissipation_epsilon * (
            + div_diss(phiI, l0) + div_diss(phiI, l1) + div_diss(phiI, l2)
        )
    )

    fun_bs_diss.split_loop()

    piR_rhs_diss = boson_star.overwrite(piR_rhs)
    fun_bs_diss.add_eqn(
        piR_rhs_diss,
        piR_rhs + dissipation_epsilon * (
            + div_diss(piR, l0) + div_diss(piR, l1) + div_diss(piR, l2)
        )
    )

    piI_rhs_diss = boson_star.overwrite(piI_rhs)
    fun_bs_diss.add_eqn(
        piI_rhs_diss,
        piI_rhs + dissipation_epsilon * (
            + div_diss(piI, l0) + div_diss(piI, l1) + div_diss(piI, l2)
        )
    )

    # MHDuet Sommerfeld on each evolved scalar (problem.input:
    # phi*_falloff=2, pi*_falloff=3, *_asymptotic=0). Applied after KO
    # dissipation, same schedule as Z4c_upwind.py's metric NewRadX.
    nrx_batches.extend([
        NewRadXBoundaryBatch(
            phiR, sympify(0), sympify(1), radpower_phi, rhs_group,
            schedule_after=["boson_star_apply_dissipation"],
            cond="apply_NewRadX",
            name=f"bs_apply_NewRadX_{name_phiR}",
        ),
        NewRadXBoundaryBatch(
            phiI, sympify(0), sympify(1), radpower_phi, rhs_group,
            schedule_after=["boson_star_apply_dissipation"],
            cond="apply_NewRadX",
            name=f"bs_apply_NewRadX_{name_phiI}",
        ),
        NewRadXBoundaryBatch(
            piR, sympify(0), sympify(1), radpower_pi, rhs_group,
            schedule_after=["boson_star_apply_dissipation"],
            cond="apply_NewRadX",
            name=f"bs_apply_NewRadX_{name_piR}",
        ),
        NewRadXBoundaryBatch(
            piI, sympify(0), sympify(1), radpower_pi, rhs_group,
            schedule_after=["boson_star_apply_dissipation"],
            cond="apply_NewRadX",
            name=f"bs_apply_NewRadX_{name_piI}",
        ),
    ])

    return phiR, phiI, piR, piI, dtphiR, dtphiI, Vpot


star1 = add_star("1", "phiR", "phiI", "piR", "piI", initial_amplitude, initial_width)
stars = [star1]

if nstars == 2:
    # Separate star 1's dissipation eqns from star 2's (add_star() itself
    # must not end on a trailing split_loop() -- there would be no more
    # add_eqn calls after it in fun_bs_diss for the nstars=1 case, and an
    # empty trailing loop segment is a generator error).
    fun_bs_diss.split_loop()
    # mhduetBS-2.0's own field names for its second, independent complex
    # scalar: phe(R/I) for the field, pe(R/I) for its reduction variable.
    star2 = add_star("2", "pheR", "pheI", "peR", "peI", initial_amplitude2, initial_width2)
    stars.append(star2)

###
# Stress-energy tensor, built as a full 4D contraction per star (matching
# mhduetBS-2.0's Tsfu_*/Tsfe_* construction), then linearly superposed and
# handed off through TmunuBaseX -- merged into the SAME scheduled function
# as the scalar RHS above (split_loop() between groups) rather than a
# separate kernel launch, since mhduetBS-2.0 computes both in one fat
# ParallelFor (see module docstring, "Kernel-launch count").
#
#   g^tt = -1/alpha^2
#   g^ti = beta^i/alpha^2
#   g^ij = chi gt^{ij} - beta^i beta^j/alpha^2
#
#   g_tt = -alpha^2 + (gt_ij/chi) beta^i beta^j
#   g_ti = (gt_ij/chi) beta^j
#   g_ij = gt_ij/chi
#
#   d_t phi -- dtphiR/dtphiI, the non-state aux vars computed in add_star()
#   above (not phiR_rhs/phiI_rhs themselves -- see the comment there).
###
g4uu_tt = -1 / evo_lapse**2
g4uu_ti = evo_shift[ui] / evo_lapse**2
g4uu_ij = chi * gt[ui, uj] - evo_shift[ui] * evo_shift[uj] / evo_lapse**2

g4ll_tt = -evo_lapse**2 + (gt[li, lj] / chi) * evo_shift[ui] * evo_shift[uj]
g4ll_ti = (gt[li, lj] / chi) * evo_shift[uj]
g4ll_ij = gt[li, lj] / chi

dphi_sq_terms = []
for label, (phiR, phiI, piR, piI, dtphiR, dtphiI, Vpot) in zip(("1", "2"), stars):
    dphi_sq = boson_star.decl(f"dphi_sq_{label}", [])
    fun_bs_rhs.add_eqn(
        dphi_sq,
        g4uu_tt * (dtphiR * dtphiR + dtphiI * dtphiI)
        + 2 * g4uu_ti * (dtphiR * D(phiR, li) + dtphiI * D(phiI, li))
        + g4uu_ij * (D(phiR, li) * D(phiR, lj) + D(phiI, li) * D(phiI, lj))
    )
    dphi_sq_terms.append((phiR, phiI, dtphiR, dtphiI, dphi_sq, Vpot))

fun_bs_rhs.split_loop()

eTtt_expr = sympify(0)
eTti_expr = sympify(0)
eTij_expr = sympify(0)
for phiR, phiI, dtphiR, dtphiI, dphi_sq, Vpot in dphi_sq_terms:
    eTtt_expr = eTtt_expr + (
        2 * (dtphiR * dtphiR + dtphiI * dtphiI) - g4ll_tt * (dphi_sq + Vpot)
    )
    eTti_expr = eTti_expr + (
        2 * (dtphiR * D(phiR, li) + dtphiI * D(phiI, li)) - g4ll_ti * (dphi_sq + Vpot)
    )
    eTij_expr = eTij_expr + (
        2 * (D(phiR, li) * D(phiR, lj) + D(phiI, li) * D(phiI, lj)) - g4ll_ij * (dphi_sq + Vpot)
    )

fun_bs_rhs.add_eqn(eTtt, eTtt_expr)
fun_bs_rhs.add_eqn(eTti[li], eTti_expr)

fun_bs_rhs.split_loop()

fun_bs_rhs.add_eqn(eTij[li, lj], eTij_expr)

###
# Second Tmunu fill, scheduled into TmunuBaseX's OWN TmunuBaseX_AddToTmunu
# group (which TmunuBaseX/schedule.ccl itself invokes both AT initial
# AFTER ADMBaseX_SetADMVars and IN ODESolvers_PostStep, inside
# TmunuBaseX_SetTmunuVars, AFTER TmunuBaseX_ZeroTmunu).
#
# Without this, eTtt/eTti/eTij are written ONLY by boson_star_rhs above,
# which is scheduled IN ODESolvers_RHS BEFORE z4c_rhs -- correct and
# necessary for z4c_rhs's own matter-sourced RHS terms (recomputed fresh
# every RK/Euler substage), but that means eTtt/eTti/eTij are nonzero only
# for the transient duration of an RHS evaluation. TmunuBaseX_SetTmunuVars
# re-zeroes them (TmunuBaseX_ZeroTmunu) immediately after every step and at
# `initial`, and nothing refills them outside the RHS window -- so ANY
# Analysis-time consumer (z4c_constraints' HamCons/MomCons/ZtCons in
# particular, since z4c_constraints is scheduled IN
# CottonmouthZ4c_AnalysisGroup, which runs after PostStep) always sees a
# vacuum (zeroed) stress-energy, regardless of --metric-thorn. This turns
# every constraint diagnostic into a silent vacuum measurement.
#
# This block duplicates the SAME eTtt/eTti/eTij formula (same g4uu/g4ll,
# same per-star Vpot substitution rule) into its own function scheduled
# into TmunuBaseX_AddToTmunu, so it runs at `initial` and after every step
# -- exactly where constraint diagnostics (and anything else reading
# TmunuBaseX at Analysis/PostStep time) expect matter contributions to
# live. dtphi{R,I} are recomputed fresh here (distinct symbols, not the
# RHS function's own dtphiR/dtphiI aux vars) from the CURRENT phiR/phiI/
# piR/piI state, since those are the only inputs still valid outside the
# RHS window. This does NOT replace the RHS-embedded computation above --
# multi-stage integrators (RK4 etc.) still need it evaluated fresh at every
# substage, using the substage's intermediate state, not just the state at
# the start/end of a full step.
###
tmunu_fill_group = ScheduleBlock(
    group_or_function=GroupOrFunction.Group,
    name=Identifier("CottonmouthBosonStar_TmunuFillGroup"),
    at_or_in=AtOrIn.In,
    schedule_bin=Identifier("TmunuBaseX_AddToTmunu"),
    description=String(
        "Boson star Tmunu fill for TmunuBaseX_AddToTmunu (initial + "
        "post-step), so Analysis-time constraint diagnostics see the real "
        "matter source instead of TmunuBaseX_ZeroTmunu's vacuum default"
    ),
)

fun_bs_tmunu_fill = boson_star.create_function(
    "boson_star_tmunu_fill",
    tmunu_fill_group
)

dphi_sq_terms_fill = []
for label, (phiR, phiI, piR, piI, dtphiR, dtphiI, Vpot) in zip(("1", "2"), stars):
    dtphiR_f = boson_star.decl(f"dtphi{label}R_tmunu", [])
    dtphiI_f = boson_star.decl(f"dtphi{label}I_tmunu", [])
    fun_bs_tmunu_fill.add_eqn(dtphiR_f, evo_shift[ui] * Dupwind(phiR, li) - evo_lapse * piR)
    fun_bs_tmunu_fill.add_eqn(dtphiI_f, evo_shift[ui] * Dupwind(phiI, li) - evo_lapse * piI)

    dphi_sq_f = boson_star.decl(f"dphi_sq_{label}_tmunu", [])
    fun_bs_tmunu_fill.add_eqn(
        dphi_sq_f,
        g4uu_tt * (dtphiR_f * dtphiR_f + dtphiI_f * dtphiI_f)
        + 2 * g4uu_ti * (dtphiR_f * D(phiR, li) + dtphiI_f * D(phiI, li))
        + g4uu_ij * (D(phiR, li) * D(phiR, lj) + D(phiI, li) * D(phiI, lj))
    )
    dphi_sq_terms_fill.append((phiR, phiI, dtphiR_f, dtphiI_f, dphi_sq_f, Vpot))

fun_bs_tmunu_fill.split_loop()

eTtt_fill_expr = sympify(0)
eTti_fill_expr = sympify(0)
eTij_fill_expr = sympify(0)
for phiR, phiI, dtphiR_f, dtphiI_f, dphi_sq_f, Vpot in dphi_sq_terms_fill:
    eTtt_fill_expr = eTtt_fill_expr + (
        2 * (dtphiR_f * dtphiR_f + dtphiI_f * dtphiI_f) - g4ll_tt * (dphi_sq_f + Vpot)
    )
    eTti_fill_expr = eTti_fill_expr + (
        2 * (dtphiR_f * D(phiR, li) + dtphiI_f * D(phiI, li)) - g4ll_ti * (dphi_sq_f + Vpot)
    )
    eTij_fill_expr = eTij_fill_expr + (
        2 * (D(phiR, li) * D(phiR, lj) + D(phiI, li) * D(phiI, lj)) - g4ll_ij * (dphi_sq_f + Vpot)
    )

fun_bs_tmunu_fill.add_eqn(eTtt, eTtt_fill_expr)
fun_bs_tmunu_fill.add_eqn(eTti[li], eTti_fill_expr)

fun_bs_tmunu_fill.split_loop()

fun_bs_tmunu_fill.add_eqn(eTij[li, lj], eTij_fill_expr)

###
# Sync the scalar-field state after each RK substep. Since this thorn
# schedules its RHS "before" the metric thorn's z4c_rhs (in the SAME
# ODESolvers_RHS bin), it benefits from the same pre-RHS state sync
# ODESolvers already provides for the whole registered state vector, so
# no separate explicit sync is required here for the scalar fields' own
# derivatives -- HandsOff sync mode below has ODESolvers manage this.
###

###
# Bake the cake
###
boson_star.bake(
    do_cse=False,
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

wizard = CppCarpetXWizard(
    boson_star,
    CppCarpetXGenerator(
        boson_star,
        sync_mode=SyncMode.HandsOff,
        interior_sync_schedule_target=rhs_group,
        extra_schedule_blocks=[
            initial_group,
            rhs_group,
            tmunu_fill_group,
        ],
        new_rad_x_boundary_fns=nrx_batches,
    ),
    license_header=license_header,
    license_file=license_file
)
wizard.generate_thorn()
install_bs_profile_table(
    Path(wizard.base_dir) / "src",
    ["*_boson_star_initial.cpp"],
)

# References
# [1] mhduetBS-2.0/src/AdvanceLevel.cpp and
#     mhduetBS-2.0/src/documentation/PDEModel-*.xml (Simflowny-generated
#     CCZ4 + complex-Klein-Gordon boson-star model), as analyzed against
#     this repository's own copy under simflowny-aarch64-cuda/.
# [2] Cottonmouth/Z4c.py, Z4c_upwind.py -- structural template for the
#     ThornDef/decl/create_function/bake/CppCarpetXWizard pipeline and the
#     from_thorn="TmunuBaseX" matter-coupling convention.
# [3] fairness.md (repo root) -- gap analysis this revision (--nstars,
#     merged Tmunu kernel, metric-coupling warning) addresses.
