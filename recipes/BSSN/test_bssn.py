if __name__ == "__main__":
    from EmitCactus.dsl.use_indices import *
    from EmitCactus.dsl.sympywrap import mkMatrix
    from EmitCactus.emit.ccl.schedule.schedule_tree import AtOrIn, GroupOrFunction, ScheduleBlock
    from EmitCactus.emit.tree import Identifier, String
    from EmitCactus.generators.wizards import CppCarpetXWizard
    from EmitCactus.generators.cpp_carpetx_generator import CppCarpetXGenerator
    from EmitCactus.generators.cactus_generator import InteriorSyncMode
    from sympy import Rational, sqrt

    ###
    # Thorn definition
    ###
    pybssn_test_bssn = ThornDef("PyBSSN", "TestBSSN")

    ###
    # Imported BSSN Vars.
    ###
    HamCons = pybssn_test_bssn.decl("HamCons", [], from_thorn="BSSN")

    ###
    # Thorn vars.
    ###
    HamConsTest = pybssn_test_bssn.decl("HamConsTest", [])

    ###
    # Thorn functions
    ###
    tanh = pybssn_test_bssn.declfun("tanh", args=1, is_stencil=False)

    ###
    # Write initial data
    ###
    fun_test = pybssn_test_bssn.create_function(
        "pybssn_test_bssn",
        ScheduleBin.Analysis,
        schedule_after=["BSSN_AnalysisGroup"]
    )

    fun_test.add_eqn(HamConsTest, tanh(HamCons))

    fun_test.bake()

    ###
    # Thorn creation
    ###
    CppCarpetXWizard(
        pybssn_test_bssn,
        CppCarpetXGenerator(
            pybssn_test_bssn,
            interior_sync_mode=InteriorSyncMode.HandsOff
        )
    ).generate_thorn()
