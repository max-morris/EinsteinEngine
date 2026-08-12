#  Copyright (C) 2024-2026 Lucas Timotheo Sanches, Steven R. Brandt, Max Morris, and other Einstein Engine contributors.
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
from typing import Any

from sympy import Rational, Idx

from EinsteinEngine import *

################################
# BEGIN Generate Options
###

###
# Finite difference stencils
###

parser = argparse.ArgumentParser(prog='Cottonmouth BSSNOK', description='A code generator for the BSSNOK equations')
parser.add_argument('--vacuum', action='store_true', default=False, help='Whether to generate matter terms.')
parser.add_argument('--fd-order', type=int, default=4, help='Order of the finite difference equations to use.')
pres=parser.parse_args(sys.argv[1:])

stencil_order = pres.fd_order
use_matter_terms = 0 if pres.vacuum else 1

suffix = f"{stencil_order}{'v' if pres.vacuum else 'm'}"

###
# END Generate Options
################################

###
# Thorn definitions
###
cottonmouth_bssnok = VanillaF90Module("cottonmouth_bssnok_f90", derivative_stencil_width=stencil_order + 1)

####
####

###
# Extra math functions
###
def_max = cottonmouth_bssnok.decl_fun("max", args=2, is_stencil=False)

###
# Thorn parameters
###
eta_B = cottonmouth_bssnok.add_param(
    "eta_b",
    float
)

###
# ADMBaseX vars.
###
g = cottonmouth_bssnok.decl(
    "g",
    [la, lb],
    symmetries=[(la, lb)]
)

###
# TmunuBaseX vars.
###
eTtt = cottonmouth_bssnok.decl("eTtt", [])

eTti = cottonmouth_bssnok.decl("eTt", [la])

eTij = cottonmouth_bssnok.decl(
    "eT",
    [la, lb],
    symmetries=[(la, lb)]
)

###
# Evolved Gauge Vars.
###
evo_lapse_rhs = cottonmouth_bssnok.decl(
    "evo_lapse_rhs",
    []
)

evo_lapse = cottonmouth_bssnok.decl(
    "evo_lapse",
    []
)

evo_shift_rhs = cottonmouth_bssnok.decl(
    "evo_shift_rhs",
    [ua]
)

evo_shift = cottonmouth_bssnok.decl(
    "evo_shift",
    [ua]
)

shift_B_rhs = cottonmouth_bssnok.decl(
    "shift_B_rhs",
    [ua]
)

shift_B = cottonmouth_bssnok.decl(
    "shift_B",
    [ua]
)

###
# Evolved BSSN Vars.
###
# w (conformal factor)
w_rhs = cottonmouth_bssnok.decl("w_rhs", [])
w = cottonmouth_bssnok.decl("w", [])

# \tilde{\gamma_{a b}}
gt_rhs = cottonmouth_bssnok.decl(
    "gt_rhs",
    [la, lb],
    symmetries=[(la, lb)]
)

gt = cottonmouth_bssnok.decl(
    "gt",
    [la, lb],
    symmetries=[(la, lb)]
)

# \tilde{A}_{a b}
At_rhs = cottonmouth_bssnok.decl(
    "At_rhs",
    [la, lb],
    symmetries=[(la, lb)]
)

At = cottonmouth_bssnok.decl(
    "At",
    [la, lb],
    symmetries=[(la, lb)]
)

# K (trace of Extrinsic Curvature)
trK_rhs = cottonmouth_bssnok.decl("trK_rhs", [])
trK = cottonmouth_bssnok.decl("trK", [])

# \tilde{\Gamma}^a
ConfConnect_rhs = cottonmouth_bssnok.decl(
    "ConfConnect_rhs",
    [ua]
)

ConfConnect = cottonmouth_bssnok.decl(
    "ConfConnect",
    [ua]
)

###
# Ricci tensor.
###

# \tilde{R}_{a b}
Rt = cottonmouth_bssnok.decl("Rt", [la, lb], symmetries=[(la, lb)])
Rt_tmp = cottonmouth_bssnok.decl("Rt_tmp", [la, lb], symmetries=[(la, lb)])

# \tilde{R}^{\phi}_{a b}
RPhi = cottonmouth_bssnok.decl("RPhi", [la, lb], symmetries=[(la, lb)])

# R_{a b} = \tilde{R}_{a b} + R^\phi_{a b}
R = cottonmouth_bssnok.decl(
    "R",
    [la, lb],
    symmetries=[(la, lb)]
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
ConfConnect_rhs_tmp2 = cottonmouth_bssnok.decl("ConfConnect_rhs_tmp2", [ua])

# \Delta^i = \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk}
# When \tilde{\Gamma}^{i} appears without derivatives, we replace it by
# \Delta^i = \tilde{\gamma}^{jk} \tilde{\Gamma}^{i}_{jk}
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
# Physical metric and its inverse
g_mat = cottonmouth_bssnok.get_matrix(g[la, lb])
g_imat = inv(g_mat)
detg = det(g_mat)
cottonmouth_bssnok.add_substitution_rule(g[ua, ub], g_imat)

# Conformal metric and its inverse
gt_mat = cottonmouth_bssnok.get_matrix(gt[la, lb])
detgt = det(gt_mat)

# Use the fact that det(gt) = 1 to simplify the inverse expression
# Note that det(gt) = 1 is an *enforced* constraint
gt_imat = inv(gt_mat) * detgt
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
    gt[ub, uc] * Gammat[ua, lb, lc]
)

# Phi derivatives w.r.t the conformal metric
cottonmouth_bssnok.add_substitution_rule(
    cdphi[la],
    -Rational(1, 2) * (1 / w) * D(w, la)
)

# Matter term definitions
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

def add_ricci(fun: DslFunctionFrontend[Any], la: Idx, lb: Idx) -> None:
    fun.add_eqn(
        Rt_tmp[la, lb],
        - Rational(1, 2) * gt[uc, ud] * D(gt[la, lb], lc, ld)
        + Rational(1, 2) * gt[lc, la] * D(ConfConnect[uc], lb)
        + Rational(1, 2) * gt[lc, lb] * D(ConfConnect[uc], la)
    )

    fun.split_loop()

    fun.add_eqn(
        Rt[la, lb],
        Rt_tmp[la, lb]
        + Rational(1, 2) * Delta[uc] * Gammat[la, lb, lc]
        + Rational(1, 2) * Delta[uc] * Gammat[lb, la, lc]
        + Gammat[uc, la, ld] * Gammat[lb, lc, ud]
        + Gammat[uc, lb, ld] * Gammat[la, lc, ud]
        + Gammat[uc, la, ld] * Gammat[lc, lb, ud]
    )


    fun.add_eqn(
        cdphi2[la, lb],
        -Rational(1, 2) * (1 / w) * (
            D(w, la, lb)
            - Gammat[uc, la, lb] * D(w, lc)
        )
        + Rational(1, 2) * (1 / (w**2)) * D(w, la) * D(w, lb)
    )

    fun.add_eqn(
        RPhi[la, lb],
        - 2 * cdphi2[lb, la]
        - 2 * gt[la, lb] * gt[uc, ud] * cdphi2[lc, ld]
        + 4 * cdphi[la] * cdphi[lb]
        - 4 * gt[la, lb] * gt[uc, ud] * cdphi[lc] * cdphi[ld]
    )

    fun.add_eqn(
        R[la, lb],
        Rt[la, lb] + RPhi[la, lb]
    )

###
# BSSN Evolution equations
# Following [1,2], we will replace \tilde{\Gamma}^i with
# \Delta^i \equiv \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk}
# whenever \tilde{\Gamma}^i are needed without derivatives.
###
fun_bssn_rhs = cottonmouth_bssnok.create_function(
    "rhs",
    intent_override=IntentOverride.WriteInterior
)


add_ricci(fun_bssn_rhs, la, lb)
fun_bssn_rhs.split_loop()

fun_bssn_rhs.add_eqn(
    gt_rhs[la, lb],
    - 2 * evo_lapse * At[la, lb]
    + gt[la, lc] * D(evo_shift[uc], lb)
    + gt[lb, lc] * D(evo_shift[uc], la)
    - Rational(2, 3) * gt[la, lb] * D(evo_shift[uc], lc)
    # TODO: Advection: + Upwind[beta[uc], gt[la,lb], lc]
    + evo_shift[uc] * D(gt[la, lb], lc)
)


# Hyperbolic Gamma Driver shift
fun_bssn_rhs.add_eqn(
    evo_shift_rhs[ua],
    Rational(3, 4) * evo_lapse * shift_B[ua]
    # TODO: Advection
    + evo_shift[ub] * D(evo_shift[ua], lb)
)

# 1 + log lapse.
fun_bssn_rhs.add_eqn(
    evo_lapse_rhs,
    - 2 * evo_lapse * trK
    # TODO: Advection: Upwind[beta[ua], alpha, la]
    + evo_shift[ua] * D(evo_lapse, la)
)

fun_bssn_rhs.add_eqn(
    w_rhs,
    Rational(1, 3) * w * (
        evo_lapse * trK
        - D(evo_shift[ua], la)
    )
    # TODO: Advection: + Upwind[beta[ua], phi, la]
    + evo_shift[ua] * D(w, la)
)

fun_bssn_rhs.soft_split()

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
    + use_matter_terms * (-8) * pi * evo_lapse * (
        w**2 * eTij[la, lb] - Rational(1, 3) * gt[la, lb] * trS
    )
    # TODO: Advection: + Upwind[beta[uc], At[la,lb], lc]
    + evo_shift[uc] * D(At[la, lb], lc)
)

fun_bssn_rhs.split_loop()

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
    + use_matter_terms * 4 * pi * evo_lapse * (rho + trS)
    # TODO: Advection: + Upwind[beta[ua], trK, la]
    + evo_shift[ua] * D(trK, la)
)

fun_bssn_rhs.add_eqn(
    ConfConnect_rhs_tmp[ua], ConfConnect_rhs_tmp2[ua]
    - 2 * At[ua, ub] * D(evo_lapse, lb)
    + 2 * evo_lapse * (
        + Gammat[ua, lb, lc] * At[ub, uc]
        - Rational(2, 3) * gt[ua, ub] * D(trK, lb)
        + 6 * At[ua, ub] * cdphi[lb]
    )
)

fun_bssn_rhs.soft_split()

fun_bssn_rhs.add_eqn(
    shift_B_rhs[ua],
    ConfConnect_rhs_tmp[ua]
    # TODO: Advection
    - evo_shift[ub] * D(ConfConnect[ua], lb)
    - eta_B * shift_B[ua]
    # TODO: Advection
    + evo_shift[ub] * D(shift_B[ua], lb)
)

fun_bssn_rhs.add_eqn(
    ConfConnect_rhs_tmp2[ua], 0
    + gt[ub, uc] * D(evo_shift[ua], lb, lc)
    + Rational(1, 3) * gt[ua, ub] * D(evo_shift[uc], lb, lc)
    - Delta[ub] * D(evo_shift[ua], lb)
    + Rational(2, 3) * Delta[ua] * D(evo_shift[ub], lb)
    # Matter
    + use_matter_terms * (-16) * pi * evo_lapse * gt[ua, ub] * S[lb]
    # TODO: Advection: + Upwind[beta[ub], Xt[ua], lb]
    + evo_shift[ub] * D(ConfConnect[ua], lb)
)


fun_bssn_rhs.add_eqn(ConfConnect_rhs[ua], ConfConnect_rhs_tmp[ua])

###
# Bake the cake
###
cottonmouth_bssnok.bake(
    do_cse=True,
    temporary_promotion_strategy=promote_none(),
    do_madd=False,
    do_recycle_temporaries=False,
    cse_optimization_level=CseOptimizationLevel.Fast,
    soft_split_retainment_strategy=retain_rank(50),
    functions={
        "rhs": {
            "soft_split_retainment_strategy": retain_rank(100)
        }
    },
    ordering_fn=functools.partial(
        prioritize_rare_symbols, consider_frequency=True, complexity_factor=0.0
    )
)

recipe_dir = Path(__file__).resolve().parent

with (recipe_dir / 'cottonmouth_agpl3.txt').open('r') as fd:
    license_file = fd.read()

with (recipe_dir / 'cottonmouth_agpl3_header.txt').open('r') as fd:
    license_header = fd.read()

VanillaF90Wizard(
    cottonmouth_bssnok,
    license_header=license_header,
    license_file=license_file
).generate_module()

# References
# [1] https://docs.einsteintoolkit.org/et-docs/images/0/05/PeterDiener15-MacLachlan.pdf
# [2] https://bitbucket.org/einsteintoolkit/mclachlan/src/46157bbd3a716dc36c31fde08b1eaea6cabb1ca4/m/McLachlan_BSSN.m
# [3] https://github.com/nrpy/nrpy/blob/main/nrpy/equations/general_relativity/nrpylatex/test_parse_BSSN.py
# [4] https://arxiv.org/abs/gr-qc/9810065
# [5] https://arxiv.org/pdf/0910.3803
# [6] https://arxiv.org/abs/gr-qc/0605030.
# [7] https://arxiv.org/abs/1212.2901
