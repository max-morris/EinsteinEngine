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
from EinsteinEngine.intermediate.eqnlist import stencil

from EinsteinEngine.generators.generator_exception import GeneratorException

from EinsteinEngine.emit.ccl.schedule.schedule_tree import IntentRegion
from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend
from EinsteinEngine.generators.generator import Generator
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend


class DslGenerator[F: DslFrontend[Any, Any, Any]](Generator[F], ABC):
    def __init__(self, frontend: F):
        super().__init__(frontend)

    def _get_output_region_for_loop(self,
                                    frontend: DslFunctionFrontend[F],
                                    var_names: set[str],
                                    loop_idx: int) -> IntentRegion:

        """
        Figure out what kind of loop we need (all, int, bnd) based on the write region of the loop's outputs, or, failing that, the inputs.
        All of this loop's outputs need to have the same write region.
        """

        eqn_list = frontend.eqn_complex.eqn_lists[loop_idx]
        write_decls = eqn_list.write_decls
        read_decls = eqn_list.read_decls

        writes = {
            var: spec
            for var, spec in ((str(var).replace("'", ""), spec) for var, spec in write_decls.items())
            if var in var_names
        }

        reads = {
            var: spec
            for var, spec in ((str(var).replace("'", ""), spec) for var, spec in read_decls.items())
            if var in var_names
        }

        if len(writes) == 0 and len(reads) == 0:
            return IntentRegion.Everywhere  # No inputs and outputs; assume analytical

        if len(writes) == 0:
            input_regions = set(reads.values())

            if None in input_regions or len(input_regions) == 0:
                raise GeneratorException(f"In {frontend.name}@{loop_idx}: All input vars must have a read region. There are no output vars.")

            for rhs in eqn_list.eqns.values():
                for sten in rhs.find(stencil):  # type: ignore[no-untyped-call]
                    if sten.args[1] != 0 or sten.args[2] != 0 or sten.args[3] != 0:
                        return IntentRegion.Interior

            if len(input_regions) > 1:
                if len(input_regions) == 2 and IntentRegion.Everywhere in input_regions and IntentRegion.Interior in input_regions:
                    wprint(f"In {frontend.name}@{loop_idx}:"
                           f" While trying to infer the loop region, we found that there were no output vars,"
                           f" and we found the input vars to have a mix of Interior and Everywhere read regions."
                           f" It looks like you are trying to write to a tile temp based on a stencil function, e.g.,"
                           f" finite difference, so we will infer Interior as the loop region.")
                    return IntentRegion.Interior

                raise GeneratorException(
                    f"In {frontend.name}@{loop_idx}: Input vars have mixed read regions: {list(write_decls.items())}\nSince there are no output vars, the loop region cannot be inferred."
                )

            [input_region] = input_regions
            return input_region
        else:
            output_regions = set(writes.values())

            if None in output_regions or len(output_regions) == 0:
                raise GeneratorException(f"In {frontend.name}@{loop_idx}: All output vars for must have a write region.")

            if len(output_regions) > 1:
                raise GeneratorException(
                    f"In {frontend.name}@{loop_idx}: Output vars have mixed write regions: {list(write_decls.items())}\n\n"
                    f"Hint: You can normalize the write regions to Interior by supplying intent_override=IntentOverride.WriteInterior to create_function()."
                )

            [output_region] = output_regions
            return output_region
