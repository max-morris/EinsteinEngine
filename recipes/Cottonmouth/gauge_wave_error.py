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

"""
Correctness-test diagnostic thorn: at every analysis call, recomputes the
exact gauge-wave solution H(x,t) (same closed-form as gauge_wave.py's
CottonmouthGaugeWaveID, Appendix A.3 of https://arxiv.org/abs/0709.3559)
at the CURRENT simulation time, and writes

    err_gxx = ADMBaseX::g_xx - (1 - H(x,t))
    err_alp = ADMBaseX::alp  - sqrt(1 - H(x,t))

as ordinary (non-evolved) grid functions. Since ADMBaseX::g/alp are the
formulation-agnostic public state any Cottonmouth metric thorn (Z4c,
Z4c-upwind, BSSNOK, ...) writes back into every step (see e.g. Z4c.py's
fun_z4c_to_adm), this diagnostic works unmodified against any of them --
point this at CottonmouthZ4cUpwind4v (or 4m) to check whether the upwinded
advection derivatives still solve the gauge-wave test correctly.

Feed err_gxx/err_alp into CarpetX::out_norm_vars (see
apples_with_apples/gauge_wave_z4c.par for the pattern -- it already dumps
CottonmouthZ4c4m::HamCons the same way) to get an automatic
<out_dir>/norms/cottonmouthgaugewaveerror-err_*.tsv time series of the L2/
Linf error norm. Run the SAME parfile at two or three resolutions ($rho in
gauge_wave_z4c.par) and check the norm drops by ~2^4=16x per doubling for
this scheme's 4th-order finite differencing -- that convergence rate is
itself the correctness signal: a coding bug (wrong sign, missing term,
wrong prefactor) generically breaks or degrades the convergence *order*
even when the solution isn't obviously wrong at a single resolution/instant.

amplitude/wavelength MUST be set to the SAME values as
CottonmouthGaugeWaveID::amplitude/wavelength in the parfile -- they are
independent parameters (this thorn does not read the ID thorn's params) so
that this diagnostic can, in principle, also be pointed at a different
gauge-wave setup without recompiling.
"""

if __name__ == "__main__":
    from pathlib import Path

    from EinsteinEngine import *

    ###
    # Thorn definition
    ###
    gauge_wave_error = ThornDef(
        "Cottonmouth",
        "CottonmouthGaugeWaveError"
    )

    ###
    # Thorn parameters (must match CottonmouthGaugeWaveID's in the parfile)
    ###
    amplitude = gauge_wave_error.add_param(
        "amplitude",
        default=0.01,
        desc="Gauge wave amplitude -- must match CottonmouthGaugeWaveID::amplitude."
    )

    wavelength = gauge_wave_error.add_param(
        "wavelength",
        default=1.0,
        desc="Gauge wave wavelength -- must match CottonmouthGaugeWaveID::wavelength."
    )

    ###
    # ADMBaseX vars (read-only: the formulation-agnostic public metric state
    # every Cottonmouth metric thorn writes back into after each step).
    ###
    g = gauge_wave_error.decl(
        "g",
        [li, lj],
        symmetries=[(li, lj)],
        from_thorn="ADMBaseX"
    )

    alp = gauge_wave_error.decl(
        "alp",
        [],
        from_thorn="ADMBaseX"
    )

    ###
    # Diagnostic outputs
    ###
    err_gxx = gauge_wave_error.decl("err_gxx", [])
    err_alp = gauge_wave_error.decl("err_alp", [])

    ###
    # Group: analysis, same pattern as Z4c.py's own constraint diagnostics.
    ###
    analysis_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("CottonmouthGaugeWaveError_AnalysisGroup"),
        at_or_in=AtOrIn.At,
        schedule_bin=Identifier("analysis"),
        description=String("Gauge-wave exact-solution error diagnostic"),
    )

    ###
    # Exact solution at the CURRENT time (with_time=True binds `t` to
    # cctk_time, not just the t=0 initial-data value gauge_wave.py uses).
    ###
    t, x, y, z = gauge_wave_error.mk_coords(with_time=True)

    pi = sympify(3.141592653589793)
    H = amplitude * sin((2 * pi * (x - t)) / wavelength)

    hxx_exact = 1 - H
    lapse_exact = sqrt(1 - H)

    fun_error = gauge_wave_error.create_function(
        "gauge_wave_error",
        analysis_group
    )

    fun_error.add_eqn(err_gxx, g[l0, l0] - hxx_exact)
    fun_error.add_eqn(err_alp, alp - lapse_exact)

    gauge_wave_error.bake()

    ###
    # Thorn creation
    ###
    recipe_dir = Path(__file__).resolve().parent

    with (recipe_dir / 'cottonmouth_agpl3.txt').open('r') as fd:
        license_file = fd.read()

    with (recipe_dir / 'cottonmouth_agpl3_header.txt').open('r') as fd:
        license_header = fd.read()

    CppCarpetXWizard(
        gauge_wave_error,
        CppCarpetXGenerator(
            gauge_wave_error,
            sync_mode=SyncMode.EmulatePresync,
            extra_schedule_blocks=[analysis_group]
        ),
        license_header=license_header,
        license_file=license_file
    ).generate_thorn()
