if __name__ == "__main__":
    from typing import List
    from sympy import Expr, cbrt
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import do_sympify, mkMatrix
    from EmitCactus.emit.ccl.schedule.schedule_tree import AtOrIn, GroupOrFunction, ScheduleBlock
    from EmitCactus.emit.tree import Identifier, String
    from EmitCactus.generators.wizards import CppCarpetXWizard
    from EmitCactus.generators.cpp_carpetx_generator import CppCarpetXGenerator
    from EmitCactus.generators.cactus_generator import InteriorSyncMode
    from sympy import Rational, sqrt

    ###
    # Thorn definition
    ###
    pybssn_kerr_schild_id = ThornDef("PyBSSN", "KerrSchildID")

    ###
    # Thorn parameters
    ###
    mass = pybssn_kerr_schild_id.add_param(
        "mass",
        default=1.0,
        desc="Black hole mass"
    )

    spin = pybssn_kerr_schild_id.add_param(
        "spin",
        default=1.0,
        desc="Black hole spin"
    )

    ###
    # ADMBaseX vars.
    ###
    # Variables
    g = pybssn_kerr_schild_id.decl("g", [li, lj], from_thorn="ADMBaseX")
    pybssn_kerr_schild_id.add_sym(g[li, lj], li, lj)
    pybssn_kerr_schild_id.mk_subst(g[li, lj], mksymbol_for_tensor_xyz)

    k = pybssn_kerr_schild_id.decl("k", [li, lj], from_thorn="ADMBaseX")
    pybssn_kerr_schild_id.add_sym(k[li, lj], li, lj)
    pybssn_kerr_schild_id.mk_subst(k[li, lj], mksymbol_for_tensor_xyz)

    alp = pybssn_kerr_schild_id.decl("alp", [], from_thorn="ADMBaseX")

    beta = pybssn_kerr_schild_id.decl("beta", [ua], from_thorn="ADMBaseX")
    pybssn_kerr_schild_id.mk_subst(beta[ua], mksymbol_for_tensor_xyz)

    # First derivatives
    dtalp = pybssn_kerr_schild_id.decl("dtalp", [], from_thorn="ADMBaseX")

    dtbeta = pybssn_kerr_schild_id.decl("dtbeta", [ua], from_thorn="ADMBaseX")
    pybssn_kerr_schild_id.mk_subst(dtbeta[ua], mksymbol_for_tensor_xyz)

    dtk = pybssn_kerr_schild_id.decl("dtk", [la, lb], from_thorn="ADMBaseX")
    pybssn_kerr_schild_id.add_sym(dtk[la, lb], la, lb)
    pybssn_kerr_schild_id.mk_subst(dtk[la, lb], mksymbol_for_tensor_xyz)

    # Second derivatives
    dt2alp = pybssn_kerr_schild_id.decl("dt2alp", [], from_thorn="ADMBaseX")

    dt2beta = pybssn_kerr_schild_id.decl(
        "dt2beta",
        [ua],
        from_thorn="ADMBaseX"
    )
    pybssn_kerr_schild_id.mk_subst(dt2beta[ua], mksymbol_for_tensor_xyz)

    ###
    # Groups
    ###
    adm_id_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("KerrSchildID"),
        at_or_in=AtOrIn.In,
        schedule_bin=Identifier("ADMBaseX_InitialData"),
        description=String("Initialize ADM variables with Kerr-Schild data"),
    )

    ###
    # Base quantities
    # See https://arxiv.org/pdf/gr-qc/0002076 eqs (10)-(14)
    ###
    x, y, z = pybssn_kerr_schild_id.mk_coords(with_time=False)

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

    # h_{ij}
    # fmt: off
    hij = mkMatrix([
        [1 + 2 * H * lx * lx,     2 * H * lx * ly,     2 * H * lx * lz],
        [    2 * H * ly * lx, 1 + 2 * H * ly * ly,     2 * H * ly * lz],
        [    2 * H * lz * lx,     2 * H * lz * ly, 1 + 2 * H * lz * lz],
    ])
    # fmt: on

    # \alpha
    lapse = 1 / sqrt(1 + 2 * H)

    # \beta^{i}
    shift = [
        2 * H * lx / (1 + 2 * H),
        2 * H * ly / (1 + 2 * H),
        2 * H * lz / (1 + 2 * H)
    ]

    # K_{ij}
    drdx = r.diff(x)
    drdy = r.diff(y)
    drdz = r.diff(z)

    dHdx = H.diff(x)
    dHdy = H.diff(y)
    dHdz = H.diff(z)

    dlxdx = lx.diff(x)
    dlxdy = lx.diff(y)
    dlxdz = lx.diff(z)

    dlydx = ly.diff(x)
    dlydy = ly.diff(y)
    dlydz = ly.diff(z)

    dlzdx = lz.diff(x)
    dlzdy = lz.diff(y)
    dlzdz = lz.diff(z)

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

    Kij = mkMatrix([
        [Kxx, Kxy, Kxz],
        [Kxy, Kyy, Kyz],
        [Kxz, Kyz, Kzz],
    ])

    # Time derivatives
    dt_shift = [do_sympify(0), do_sympify(0), do_sympify(0)]

    dt_Kij = mkMatrix([
        [do_sympify(0), do_sympify(0), do_sympify(0)],
        [do_sympify(0), do_sympify(0), do_sympify(0)],
        [do_sympify(0), do_sympify(0), do_sympify(0)]
    ])

    ###
    # Write initial data
    ###
    fun_fill_id = pybssn_kerr_schild_id.create_function(
        "fill_id",
        adm_id_group
    )

    fun_fill_id.add_eqn(g[la, lb], hij)
    fun_fill_id.add_eqn(k[la, lb], Kij)
    fun_fill_id.add_eqn(alp, lapse)
    fun_fill_id.add_eqn(beta[ua], shift)

    fun_fill_id.add_eqn(dtalp, do_sympify(0))
    fun_fill_id.add_eqn(dtbeta[ua], dt_shift)
    fun_fill_id.add_eqn(dtk[la, lb], dt_Kij)

    fun_fill_id.add_eqn(dt2alp, do_sympify(0))
    fun_fill_id.add_eqn(dt2beta[ua], dt_shift)

    fun_fill_id.bake(do_recycle_temporaries=False)

    ###
    # Thorn creation
    ###
    CppCarpetXWizard(
        pybssn_kerr_schild_id,
        CppCarpetXGenerator(
            pybssn_kerr_schild_id,
            interior_sync_mode=InteriorSyncMode.Never,
            extra_schedule_blocks=[adm_id_group]
        )
    ).generate_thorn()
