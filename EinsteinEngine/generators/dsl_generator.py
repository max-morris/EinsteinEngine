#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
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

from abc import ABC
from typing import Any

from EinsteinEngine.common.util import wprint
from EinsteinEngine.intermediate.loop_region import NON_GRID_VARIABLE_NAMES, infer_loop_region

from EinsteinEngine.generators.generator_exception import GeneratorException

from EinsteinEngine.emit.ccl.schedule.schedule_tree import IntentRegion
from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend
from EinsteinEngine.generators.generator import Generator
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend


class DslGenerator[F: DslFrontend[Any, Any, Any]](Generator[F], ABC):
    vars_to_ignore: set[str] = set(NON_GRID_VARIABLE_NAMES)

    def __init__(self, frontend: F):
        super().__init__(frontend)

    def _get_output_region_for_loop(self,
                                    frontend: DslFunctionFrontend[F],
                                    var_names: set[str],
                                    loop_idx: int) -> IntentRegion:

        """
        Figure out what kind of loop we need (all, int, bnd) based on the write region of the loop's outputs, or, failing that, the inputs.
        All of this loop's outputs need to have the same write region. See `infer_loop_region`.
        """

        eqn_list = frontend.eqn_complex.eqn_lists[loop_idx]
        inferred = infer_loop_region(eqn_list, lambda sym: str(sym).replace("'", "") in var_names)

        if inferred.region is None:
            if inferred.writes_grid_vars:
                raise GeneratorException(
                    f"In {frontend.name}@{loop_idx}: Output vars have mixed write regions: {list(eqn_list.write_decls.items())}\n\n"
                    f"Hint: You can normalize the write regions to Interior by supplying intent_override=IntentOverride.WriteInterior to create_function()."
                )
            raise GeneratorException(
                f"In {frontend.name}@{loop_idx}: Input vars have mixed read regions: {list(eqn_list.read_decls.items())}\nSince there are no output vars, the loop region cannot be inferred."
            )

        if inferred.from_mixed_reads:
            wprint(f"In {frontend.name}@{loop_idx}:"
                   f" While trying to infer the loop region, we found that there were no output vars,"
                   f" and we found the input vars to have a mix of Interior and Everywhere read regions."
                   f" It looks like you are trying to write to a tile temp based on a stencil function, e.g.,"
                   f" finite difference, so we will infer Interior as the loop region.")

        return inferred.region
