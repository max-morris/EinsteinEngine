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

"""ADM initial data for a static solitonic boson star.

The 3-metric is conformally flat, g_ij = psi(r)^4 delta_ij, with K_ij = 0,
beta^i = 0, and lapse alpha(r). psi and alpha are interpolated from the
tabulated equilibrium profile in bs_profile_table.hxx (the same 1D
solution used by CottonmouthBS*). Time derivatives of the ADM variables
are identically zero.

Scalar-field initial data lives in the matter thorn (boson_star.py),
which reads the same table.
"""

if __name__ == "__main__":
    from pathlib import Path

    from EinsteinEngine import *
    from sympy import Integer

    from boson_star_profile import install_bs_profile_table

    boson_star_id = ThornDef("Cottonmouth", "CottonmouthBosonStarID")

    bs_psi = boson_star_id.decl_fun("bs_profile_psi", args=1, is_stencil=False)
    bs_alpha = boson_star_id.decl_fun("bs_profile_alpha", args=1, is_stencil=False)

    g = boson_star_id.decl(
        "g", [li, lj], symmetries=[(li, lj)], from_thorn="ADMBaseX"
    )
    k = boson_star_id.decl(
        "k", [li, lj], symmetries=[(li, lj)], from_thorn="ADMBaseX"
    )
    alp = boson_star_id.decl("alp", [], from_thorn="ADMBaseX")
    beta = boson_star_id.decl("beta", [ua], from_thorn="ADMBaseX")
    dtalp = boson_star_id.decl("dtalp", [], from_thorn="ADMBaseX")
    dtbeta = boson_star_id.decl("dtbeta", [ua], from_thorn="ADMBaseX")
    dtk = boson_star_id.decl(
        "dtk", [la, lb], symmetries=[(la, lb)], from_thorn="ADMBaseX"
    )
    dt2alp = boson_star_id.decl("dt2alp", [], from_thorn="ADMBaseX")
    dt2beta = boson_star_id.decl("dt2beta", [ua], from_thorn="ADMBaseX")

    adm_id_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("CottonmouthBosonStarID_Group"),
        at_or_in=AtOrIn.In,
        schedule_bin=Identifier("ADMBaseX_InitialData"),
        description=String("Initialize ADM variables with boson-star profile data"),
    )

    x, y, z = boson_star_id.mk_coords()
    r = sqrt(x**2 + y**2 + z**2)
    psi4 = bs_psi(r) ** 4
    zero = Integer(0)

    hij = mk_matrix([
        [psi4, zero, zero],
        [zero, psi4, zero],
        [zero, zero, psi4],
    ])
    Kij = mk_matrix([
        [zero, zero, zero],
        [zero, zero, zero],
        [zero, zero, zero],
    ])
    shift = [zero, zero, zero]
    dt_Kij = Kij
    dt_shift = shift
    dt2_shift = shift

    fun_fill_id = boson_star_id.create_function(
        "cottonmouth_boson_star_fill_id",
        adm_id_group,
    )
    fun_fill_id.add_eqn(g[la, lb], hij)
    fun_fill_id.add_eqn(k[la, lb], Kij)
    fun_fill_id.add_eqn(alp, bs_alpha(r))
    fun_fill_id.add_eqn(beta[ua], shift)
    fun_fill_id.add_eqn(dtalp, zero)
    fun_fill_id.add_eqn(dtbeta[ua], dt_shift)
    fun_fill_id.add_eqn(dtk[la, lb], dt_Kij)
    fun_fill_id.add_eqn(dt2alp, zero)
    fun_fill_id.add_eqn(dt2beta[ua], dt2_shift)

    boson_star_id.bake()

    recipe_dir = Path(__file__).resolve().parent
    with (recipe_dir / "cottonmouth_agpl3.txt").open("r") as fd:
        license_file = fd.read()
    with (recipe_dir / "cottonmouth_agpl3_header.txt").open("r") as fd:
        license_header = fd.read()

    wizard = CppCarpetXWizard(
        boson_star_id,
        CppCarpetXGenerator(
            boson_star_id,
            sync_mode=SyncMode.EmulatePresync,
            extra_schedule_blocks=[adm_id_group],
        ),
        license_header=license_header,
        license_file=license_file,
    )
    wizard.generate_thorn()
    install_bs_profile_table(
        Path(wizard.base_dir) / "src",
        ["*_cottonmouth_boson_star_fill_id.cpp"],
    )
