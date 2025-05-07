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

    ###
    # Thorn definitions
    ###
    pybssn_id_test = ThornDef("PyBSSN", "BSSN_ID_Test")

    ###
    # ADMBaseX vars.
    ###
    g = pybssn_id_test.decl("g", [li, lj], from_thorn="ADMBaseX")
    pybssn_id_test.add_sym(g[li, lj], li, lj)
    pybssn_id_test.mk_subst(g[li, lj], mksymbol_for_tensor_xyz)

    k = pybssn_id_test.decl("k", [li, lj], from_thorn="ADMBaseX")
    pybssn_id_test.add_sym(k[li, lj], li, lj)
    pybssn_id_test.mk_subst(k[li, lj], mksymbol_for_tensor_xyz)

    alp = pybssn_id_test.decl("alp", [], from_thorn="ADMBaseX")

    beta = pybssn_id_test.decl("beta", [ua], from_thorn="ADMBaseX")
    pybssn_id_test.mk_subst(beta[ua], mksymbol_for_tensor_xyz)

    ###
    # Groups
    ###
    test_id_group = ScheduleBlock(
        group_or_function=GroupOrFunction.Group,
        name=Identifier("BSSN_TestID"),
        at_or_in=AtOrIn.In,
        schedule_bin=Identifier("ADMBaseX_InitialData"),
        description=String("Initialize ADM variables with test data"),
    )

    ###
    # Test tensors
    ###
    diagonal_1 = mkMatrix([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ])

    diagonal_2 = mkMatrix([
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 4]
    ])

    vector:List[Expr] = [do_sympify(x) for x in [6, 6, 6]]
    scalar = 8

    ###
    # Initialize
    ###
    fun_fill_id = pybssn_id_test.create_function(
        "fill_id",
        test_id_group
    )

    fun_fill_id.add_eqn(g[li, lj], diagonal_1)
    fun_fill_id.add_eqn(k[li, lj], diagonal_2)
    fun_fill_id.add_eqn(alp, do_sympify(scalar))
    fun_fill_id.add_eqn(beta[ua], vector)

    fun_fill_id.bake()

    ###
    # Thorn creation
    ###
    CppCarpetXWizard(
        pybssn_id_test,
        CppCarpetXGenerator(
            pybssn_id_test,
            interior_sync_mode=InteriorSyncMode.Never,
            extra_schedule_blocks=[test_id_group]
        )
    ).generate_thorn()
