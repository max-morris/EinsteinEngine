if __name__ == "__main__":
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import do_sympify, mkMatrix
    from EmitCactus.emit.ccl.schedule.schedule_tree import AtOrIn, GroupOrFunction, ScheduleBlock
    from EmitCactus.emit.tree import Identifier, String
    from EmitCactus.generators.wizards import CppCarpetXWizard
    from EmitCactus.generators.cpp_carpetx_generator import CppCarpetXGenerator
    from EmitCactus.generators.cactus_generator import InteriorSyncMode
    from sympy import Rational, sin, sqrt

    ###
    # Thorn definition
    ###
    pybssn_gauge_wave_id = ThornDef("PyBSSN", "GaugeWaveID")

    ###
    # Thorn parameters
    ###
    amplitude = pybssn_gauge_wave_id.add_param(
        "amplitude",
        default=1.0,
        desc="Gauge wave amplitude"
    )

    wavelength = pybssn_gauge_wave_id.add_param(
        "wavelength",
        default=1.0,
        desc="Gauge wave wavelength"
    )

    ###
    # ADMBaseX vars.
    ###
    # Variables
    g = pybssn_gauge_wave_id.decl("g", [li, lj], from_thorn="ADMBaseX")
    pybssn_gauge_wave_id.add_sym(g[li, lj], li, lj)
    pybssn_gauge_wave_id.mk_subst(g[li, lj], mksymbol_for_tensor_xyz)

    k = pybssn_gauge_wave_id.decl("k", [li, lj], from_thorn="ADMBaseX")
    pybssn_gauge_wave_id.add_sym(k[li, lj], li, lj)
    pybssn_gauge_wave_id.mk_subst(k[li, lj], mksymbol_for_tensor_xyz)

    alp = pybssn_gauge_wave_id.decl("alp", [], from_thorn="ADMBaseX")

    beta = pybssn_gauge_wave_id.decl("beta", [ua], from_thorn="ADMBaseX")
    pybssn_gauge_wave_id.mk_subst(beta[ua], mksymbol_for_tensor_xyz)

    # First derivatives
    dtalp = pybssn_gauge_wave_id.decl("dtalp", [], from_thorn="ADMBaseX")

    dtbeta = pybssn_gauge_wave_id.decl("dtbeta", [ua], from_thorn="ADMBaseX")
    pybssn_gauge_wave_id.mk_subst(dtbeta[ua], mksymbol_for_tensor_xyz)

    dtk = pybssn_gauge_wave_id.decl("dtk", [la, lb], from_thorn="ADMBaseX")
    pybssn_gauge_wave_id.add_sym(dtk[la, lb], la, lb)
    pybssn_gauge_wave_id.mk_subst(dtk[la, lb], mksymbol_for_tensor_xyz)

    # Second derivatives
    dt2alp = pybssn_gauge_wave_id.decl("dt2alp", [], from_thorn="ADMBaseX")

    dt2beta = pybssn_gauge_wave_id.decl(
        "dt2beta",
        [ua],
        from_thorn="ADMBaseX"
    )
    pybssn_gauge_wave_id.mk_subst(dt2beta[ua], mksymbol_for_tensor_xyz)

    ###
    # Groups
    ###
    adm_id_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("LinearWaveID"),
        at_or_in=AtOrIn.In,
        schedule_bin=Identifier("ADMBaseX_InitialData"),
        description=String("Initialize ADM variables with Gauge Wave data"),
    )

    ###
    # Base quantities
    # See:
    #   https://arxiv.org/abs/gr-qc/0305023
    #   https://arxiv.org/abs/0709.3559
    ###
    t, x, y, z = pybssn_gauge_wave_id.mk_coords(with_time=True)

    pi = do_sympify(3.141592653589793)
    H = amplitude * sin((2 * pi * (x - t)) / wavelength)

    # \alpha
    lapse = sqrt(1 - H)

    # \beta^{i}
    shift_x = do_sympify(0)
    shift_y = do_sympify(0)
    shift_z = do_sympify(0)

    # h_{ij}
    hxx = 1 - H
    hxy = do_sympify(0)
    hxz = do_sympify(0)
    hyy = do_sympify(1)
    hyz = do_sympify(0)
    hzz = do_sympify(1)

    # K_{ij}
    Kxx = Rational(1, 2) * H.diff(t) / (sqrt(1 - H))
    Kxy = do_sympify(0)
    Kxz = do_sympify(0)
    Kyy = do_sympify(0)
    Kyz = do_sympify(0)
    Kzz = do_sympify(0)

    # Matrices
    hij = mkMatrix([
        [hxx, hxy, hxz],
        [hxy, hyy, hyz],
        [hxz, hyz, hzz],
    ])

    shift = [
        shift_x,
        shift_y,
        shift_z,
    ]

    Kij = mkMatrix([
        [Kxx, Kxy, Kxz],
        [Kxy, Kyy, Kyz],
        [Kxz, Kyz, Kzz],
    ])

    # Time derivatives
    dt_lapse = lapse.diff(t)

    dt_shift = [
        shift_x.diff(t),
        shift_y.diff(t),
        shift_z.diff(t),
    ]

    dt_Kij = mkMatrix([
        [Kxx.diff(t), Kxy.diff(t), Kxz.diff(t)],
        [Kxy.diff(t), Kyy.diff(t), Kyz.diff(t)],
        [Kxz.diff(t), Kyz.diff(t), Kzz.diff(t)],
    ])

    dt2_lapse = dt_lapse.diff(t)

    dt2_shift = [
        dt_shift[0].diff(t),
        dt_shift[1].diff(t),
        dt_shift[2].diff(t),
    ]

    ###
    # Write initial data
    ###
    fun_fill_id = pybssn_gauge_wave_id.create_function(
        "pybssn_gauge_wave_id_fill_id",
        adm_id_group
    )

    fun_fill_id.add_eqn(g[la, lb], hij)
    fun_fill_id.add_eqn(k[la, lb], Kij)
    fun_fill_id.add_eqn(alp, lapse)
    fun_fill_id.add_eqn(beta[ua], shift)

    fun_fill_id.add_eqn(dtalp, dt_lapse)
    fun_fill_id.add_eqn(dtbeta[ua], dt_shift)
    fun_fill_id.add_eqn(dtk[la, lb], dt_Kij)

    fun_fill_id.add_eqn(dt2alp, dt2_lapse)
    fun_fill_id.add_eqn(dt2beta[ua], dt2_shift)

    fun_fill_id.bake(do_recycle_temporaries=False)

    ###
    # Thorn creation
    ###
    CppCarpetXWizard(
        pybssn_gauge_wave_id,
        CppCarpetXGenerator(
            pybssn_gauge_wave_id,
            interior_sync_mode=InteriorSyncMode.Never,
            extra_schedule_blocks=[adm_id_group]
        )
    ).generate_thorn()
