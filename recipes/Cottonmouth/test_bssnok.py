if __name__ == "__main__":
    from EmitCactus import *

    ###
    # Thorn definition
    ###
    cottonmouth_test_bssnok = ThornDef("Cottonmouth", "CottonmouthTestBSSNOK")

    ###
    # Imported BSSN Vars.
    ###
    HamCons = cottonmouth_test_bssnok.decl(
        "HamCons",
        [],
        from_thorn="CottonmouthBSSNOK"
    )

    ###
    # Thorn vars.
    ###
    HamConsTest = cottonmouth_test_bssnok.decl("HamConsTest", [])

    ###
    # Thorn functions
    ###
    tanh = cottonmouth_test_bssnok.declfun("tanh", args=1, is_stencil=False)

    ###
    # Transform hamiltonian constraint
    ###
    fun_test = cottonmouth_test_bssnok.create_function(
        "cottonmouth_test_bssnok",
        ScheduleBin.Analysis,
        schedule_after=["CottonmouthBSSNOK_AnalysisGroup"]
    )

    fun_test.add_eqn(HamConsTest, tanh(HamCons))

    fun_test.bake()

    ###
    # Thorn creation
    ###
    CppCarpetXWizard(
        cottonmouth_test_bssnok,
        CppCarpetXGenerator(
            cottonmouth_test_bssnok,
            interior_sync_mode=InteriorSyncMode.HandsOff
        )
    ).generate_thorn()
