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

"""The generator of EE's shared AMReX backend: shared analysis (analysis.py), then assembly by mode (modes.py)."""

import sympy as sy

from EinsteinEngine.emit.code.common.code_tree import CodeNode
from EinsteinEngine.frontend.dsl.amrex.amrex_frontend import AMReXModule
from EinsteinEngine.frontend.dsl.amrex.schemes import KernelLaunchMode
from EinsteinEngine.generators.amrex.analysis import LoopAnalysis, analyze_function
from EinsteinEngine.generators.amrex.modes import kernel_class, point_header
from EinsteinEngine.generators.dsl_generator import DslGenerator
from EinsteinEngine.generators.generator_exception import GeneratorException


class AMReXGenerator(DslGenerator[AMReXModule]):
    """Generates the code tree of an AMReX module whose functions are baked."""

    def generate(self, banner: str = '') -> CodeNode:
        module = self.frontend
        groups: list[tuple[str, list[str], list[LoopAnalysis]]] = list()
        prior: set[str] = set()
        guards: dict[str, sy.Expr] = {}
        used_params: set[str] = set()
        for name in module.function_order:
            fn = module.functions[name]
            if not fn.been_baked:
                raise GeneratorException(f"{name} is not baked.")
            # Point-mode functions are independent (one never reads another's outputs): nothing carries across.
            loops, tiles, writes = analyze_function(module, fn, prior,
                                                    guards if isinstance(module.mode, KernelLaunchMode) else {})
            groups.append((name, tiles, loops))
            prior |= writes
            used_params |= {p for loop in loops for p in loop.params}
        declared = module.output_bases_components()
        if unwritten := sorted(declared - prior):
            raise GeneratorException(f"{module.name}: the outputs {unwritten} are declared but no function writes them "
                                     f"(a caller would read them uninitialized).")
        if isinstance(module.mode, KernelLaunchMode):
            return kernel_class(module, groups, used_params, banner)
        order = [c for base in module.output_bases for (b, _), c in module.component_of.items() if b == base]
        return point_header(module, groups, order, banner)
