if __name__ == "__main__":
    from EmitCactus import *
    from sympy import Rational

    ###
    # Thorn definition
    ###
    cottonmouth_diagonal_linear_wave_id = ThornDef(
        "Cottonmouth",
        "CottonmouthDiagLinearWaveID"
    )

    ###
    # Thorn parameters
    ###
    amplitude = cottonmouth_diagonal_linear_wave_id.add_param(
        "amplitude",
        default=1.0,
        desc="Linear wave amplitude"
    )

    wavelength = cottonmouth_diagonal_linear_wave_id.add_param(
        "wavelength",
        default=1.0,
        desc="Linear wave wavelength"
    )

    ###
    # ADMBaseX vars.
    ###
    # Variables
    g = cottonmouth_diagonal_linear_wave_id.decl(
        "g",
        [li, lj],
        symmetries=[(li, lj)],
        from_thorn="ADMBaseX"
    )

    k = cottonmouth_diagonal_linear_wave_id.decl(
        "k",
        [li, lj],
        symmetries=[(li, lj)],
        from_thorn="ADMBaseX"
    )

    alp = cottonmouth_diagonal_linear_wave_id.decl(
        "alp",
        [],
        from_thorn="ADMBaseX"
    )

    beta = cottonmouth_diagonal_linear_wave_id.decl(
        "beta",
        [ua],
        from_thorn="ADMBaseX"
    )

    # First derivatives
    dtalp = cottonmouth_diagonal_linear_wave_id.decl(
        "dtalp",
        [],
        from_thorn="ADMBaseX"
    )

    dtbeta = cottonmouth_diagonal_linear_wave_id.decl(
        "dtbeta",
        [ua],
        from_thorn="ADMBaseX"
    )

    dtk = cottonmouth_diagonal_linear_wave_id.decl(
        "dtk",
        [la, lb],
        symmetries=[(la, lb)],
        from_thorn="ADMBaseX"
    )

    # Second derivatives
    dt2alp = cottonmouth_diagonal_linear_wave_id.decl(
        "dt2alp",
        [],
        from_thorn="ADMBaseX"
    )

    dt2beta = cottonmouth_diagonal_linear_wave_id.decl(
        "dt2beta",
        [ua],
        from_thorn="ADMBaseX"
    )

    ###
    # Groups
    ###
    adm_id_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("Cottonmouth_DiagonalLinearWaveID"),
        at_or_in=AtOrIn.In,
        schedule_bin=Identifier("ADMBaseX_InitialData"),
        description=String(
            "Initialize ADM variables with Diagonal Linear Wave data"
        ),
    )

    ###
    # Base quantities
    # See:
    #   https://arxiv.org/abs/gr-qc/0305023
    #   https://arxiv.org/abs/0709.3559
    ###
    t, x, y, z = cottonmouth_diagonal_linear_wave_id.mk_coords(with_time=True)

    pi = sympify(3.141592653589793)
    two = sympify(2)
    H = amplitude * sin((2 * pi * (x - y - t * sqrt(two))) /
                        (wavelength * sqrt(two)))

    # \alpha
    lapse = sympify(1)

    # \beta^{i}
    shift_x = sympify(0)
    shift_y = sympify(0)
    shift_z = sympify(0)

    # h_{ij}
    hxx = sympify(1)
    hxy = sympify(0)
    hxz = sympify(0)
    hyy = 1 + H
    hyz = sympify(0)
    hzz = 1 - H

    # K_{ij}
    Kxx = sympify(0)
    Kxy = sympify(0)
    Kxz = sympify(0)
    Kyy = -Rational(1, 2) * diff(H, t)
    Kyz = sympify(0)
    Kzz = Rational(1, 2) * diff(H, t)

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
    dt_lapse = diff(lapse, t)

    dt_shift = [
        diff(shift_x, t),
        diff(shift_y, t),
        diff(shift_z, t),
    ]

    dt_Kij = mkMatrix([
        [diff(Kxx, t), diff(Kxy, t), diff(Kxz, t)],
        [diff(Kxy, t), diff(Kyy, t), diff(Kyz, t)],
        [diff(Kxz, t), diff(Kyz, t), diff(Kzz, t)],
    ])

    dt2_lapse = diff(dt_lapse, t)

    dt2_shift = [
        diff(dt_shift[0], t),
        diff(dt_shift[1], t),
        diff(dt_shift[2], t),
    ]

    ###
    # Write initial data
    ###
    fun_fill_id = cottonmouth_diagonal_linear_wave_id.create_function(
        "cottonmouth_diagonal_linear_wave_fill_id",
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
        cottonmouth_diagonal_linear_wave_id,
        CppCarpetXGenerator(
            cottonmouth_diagonal_linear_wave_id,
            interior_sync_mode=InteriorSyncMode.HandsOff,
            extra_schedule_blocks=[adm_id_group]
        )
    ).generate_thorn()
