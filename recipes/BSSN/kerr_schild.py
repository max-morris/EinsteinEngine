if __name__ == "__main__":
    from EmitCactus import *
    from sympy import Rational

    ###
    # Thorn definition
    ###
    pybssn_kerr_schilld_id = ThornDef("PyBSSN", "KerrSchildID")

    ###
    # Thorn parameters
    ###
    mass = pybssn_kerr_schilld_id.add_param(
        "mass",
        default=1.0,
        desc="Black hole mass"
    )

    spin = pybssn_kerr_schilld_id.add_param(
        "spin",
        default=1.0,
        desc="Black hole angular momentum"
    )

    ###
    # ADMBaseX vars.
    ###
    # Variables
    g = pybssn_kerr_schilld_id.decl("g", [li, lj], symmetries=[(li, lj)], from_thorn="ADMBaseX")
    pybssn_kerr_schilld_id.add_substitution_rule(g[li, lj], subst_tensor_xyz)

    k = pybssn_kerr_schilld_id.decl("k", [li, lj], symmetries=[(li, lj)], from_thorn="ADMBaseX")
    pybssn_kerr_schilld_id.add_substitution_rule(k[li, lj], subst_tensor_xyz)

    alp = pybssn_kerr_schilld_id.decl("alp", [], from_thorn="ADMBaseX")

    beta = pybssn_kerr_schilld_id.decl("beta", [ua], from_thorn="ADMBaseX")
    pybssn_kerr_schilld_id.add_substitution_rule(beta[ua], subst_tensor_xyz)

    # First derivatives
    dtalp = pybssn_kerr_schilld_id.decl("dtalp", [], from_thorn="ADMBaseX")

    dtbeta = pybssn_kerr_schilld_id.decl("dtbeta", [ua], from_thorn="ADMBaseX")
    pybssn_kerr_schilld_id.add_substitution_rule(dtbeta[ua], subst_tensor_xyz)

    dtk = pybssn_kerr_schilld_id.decl("dtk", [la, lb], symmetries=[(la, lb)], from_thorn="ADMBaseX")
    pybssn_kerr_schilld_id.add_substitution_rule(dtk[la, lb], subst_tensor_xyz)

    # Second derivatives
    dt2alp = pybssn_kerr_schilld_id.decl("dt2alp", [], from_thorn="ADMBaseX")

    dt2beta = pybssn_kerr_schilld_id.decl(
        "dt2beta",
        [ua],
        from_thorn="ADMBaseX"
    )
    pybssn_kerr_schilld_id.add_substitution_rule(dt2beta[ua], subst_tensor_xyz)

    ###
    # Groups
    ###
    adm_id_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("LinearWaveID"),
        at_or_in=AtOrIn.In,
        schedule_bin=Identifier("ADMBaseX_InitialData"),
        description=String("Initialize ADM variables with Linear Wave data"),
    )

    ###
    # Base quantities
    # See https://arxiv.org/pdf/gr-qc/0002076 eqs (10)-(14)
    ###
    t, x, y, z = pybssn_kerr_schilld_id.mk_coords(with_time=True)

    # Radius
    r2 = Rational(1, 2) * (x**2 + y**2 + z**2 - spin**2) + sqrt(
        Rational(1, 4) * (x**2 + y**2 + z**2 - spin**2)**2 + spin**2 * z**2)
    r = sqrt(r2)
    r4 = r2**2
    r3 = r2 * r

    # H
    H = (mass * r3) / (r4 + spin**2 * z**2)

    # l_{i}
    lx = (r * x + spin * y) / (r2 + spin**2)
    ly = (r * y - spin * x) / (r2 + spin**2)
    lz = z / r

    # \alpha
    lapse = 1 / sqrt(1 + 2 * H)

    # \beta^{i}
    shift_x = 2 * H * lx / (1 + 2 * H)
    shift_y = 2 * H * ly / (1 + 2 * H)
    shift_z = 2 * H * lz / (1 + 2 * H)

    # h_{ij}
    hxx = 1 + 2 * H * lx * lx
    hxy = 2 * H * lx * ly
    hxz = 2 * H * lx * lz
    hyy = 1 + 2 * H * ly * ly
    hyz = 2 * H * ly * lz
    hzz = 1 + 2 * H * lz * lz

    # K_{ij}
    drdx = diff(r, x)
    drdy = diff(r, y)
    drdz = diff(r, z)

    dHdx = diff(H, x)
    dHdy = diff(H, y)
    dHdz = diff(H, z)

    dlxdx = diff(lx, x)
    dlxdy = diff(lx, y)
    dlxdz = diff(lx, z)

    dlydx = diff(ly, x)
    dlydy = diff(ly, y)
    dlydz = diff(ly, z)

    dlzdx = diff(lz, x)
    dlzdy = diff(lz, y)
    dlzdz = diff(lz, z)

    Kxx = (4*H*lx*(lx*(dHdx*ly + dHdy*lz) + 2*H*(dlxdx*ly + dlxdy*lz))) / \
        (1 + 2*H*(lx**2 + ly**2 + lz**2))

    Kxy = (2*(-2*dlxdx*H**2*lx**2 + lx*(2*dHdx*H*ly**2 + 2*dHdy*H*ly*lz + 2*H*lz*(dlydy*H - dlzdx*H + dHdx*lz) + dHdx *
           (1 - 2*H*lz**2)) + H*(2*dlxdy*H*ly*lz + dlxdx*(1 - 2*H*lz**2 + 2*H*(ly**2 + lz**2)))))/(1 + 2*H*(lx**2 + ly**2 + lz**2))

    Kxz = (2*(-2*dlxdy*H**2*lx**2 + lx*(2*H*ly*((-dlydy + dlzdx)*H + dHdx*lz) + dHdy*(1 + 2*H*lz**2)
                                        ) + H*(2*dlxdx*H*ly*lz + dlxdy*(1 + 2*H*lz**2))))/(1 + 2*H*(lx**2 + ly**2 + lz**2))

    Kyy = (4*(dlydx*H + dHdx*H*ly**3 + dHdy*H*ly**2*lz + ly*(2*H*(lx*(-(dlxdx*H) + dHdx*lx) + (dlydy -
           dlzdx)*H*lz + dHdx*lz**2) + dHdx*(1 - 2*H*(lx**2 + lz**2)))))/(1 + 2*H*(lx**2 + ly**2 + lz**2))

    Kyz = (2*(2*H*ly**2*(-(dlydy*H) + dlzdx*H + dHdx*lz) + 2*(dlydy + dlzdx)*H**2*(lx**2 + lz**2) + dHdx*lz*(1 - 2*H*(lx**2 + lz**2)) + ly*(-2*dlxdy*H**2*lx + dHdy *
           (1 + 2*H*lz**2)) + H*(dlydy*(1 - 2*H*lx**2) + 2*lz*(-(dlxdx*H*lx) + dHdx*(lx**2 + lz**2)) + dlzdx*(1 - 2*H*(lx**2 + 2*lz**2)))))/(1 + 2*H*(lx**2 + ly**2 + lz**2))

    Kzz = (4*(dlzdy*H + (dHdy - 2*H**2*(dlxdy*lx + (dlydy - dlzdx)*ly)) *
           lz + dHdx*H*ly*lz**2 + dHdy*H*lz**3))/(1 + 2*H*(lx**2 + ly**2 + lz**2))

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
    fun_fill_id = pybssn_kerr_schilld_id.create_function(
        "pybssn_kerr_schild_id_fill_id",
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
        pybssn_kerr_schilld_id,
        CppCarpetXGenerator(
            pybssn_kerr_schilld_id,
            interior_sync_mode=InteriorSyncMode.HandsOff,
            extra_schedule_blocks=[adm_id_group]
        )
    ).generate_thorn()
