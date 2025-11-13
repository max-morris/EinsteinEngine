if __name__ == "__main__":
    """
    This recipe tests EmitCactus's step functions by creating boxcar and
    checking if the output is as expected.
    """

    from EmitCactus import *
    import nrpy.helpers.conditional_file_updater as cfu

    # Create a set of grid functions
    gf = ThornDef("TestEmitCactus", "TestStep")

    # Declare gfs
    boxcar = gf.decl("boxcar", [], centering=Centering.VVC)

    # Get coords
    x, y, z = gf.mk_coords()
    r = sqrt(x**2 + y**2 + z**2)

    # Add the equations we want to evolve.
    fun = gf.create_function(
        "init_boxcar",
        ScheduleBin.Init
    )
    fun.add_eqn(boxcar, h_step(r + 0.25) - h_step(r - 0.25))
    fun.bake()

    CppCarpetXWizard(gf).generate_thorn()
