#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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

if __name__ == "__main__":
    """
    This recipe tests EinsteinEngine's step functions by creating boxcar and
    checking if the output is as expected.
    """

    from EinsteinEngine import *
    import nrpy.helpers.conditional_file_updater as cfu

    # Create a set of grid functions
    gf = ThornDef("TestEinsteinEngine", "TestStep")

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
