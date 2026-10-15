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

import sympy as sy

from EinsteinEngine.emit.code.common.code_tree import Stmt
from EinsteinEngine.emit.code.grchombo.grchombo_sympy_visitor import GRChomboSympyVisitor
from EinsteinEngine.emit.code.grchombo.grchombo_tree import PointerDecl, TempDecl, StoreVar, ComputeClassParam, \
    ComputeClassDecl
from EinsteinEngine.frontend.dsl.grchombo.grchombo_frontend import GRChomboComputeClass
from EinsteinEngine.generators.dsl_generator import DslGenerator
from EinsteinEngine.generators.generator_exception import GeneratorException
from EinsteinEngine.generators.substitute_recycled_temporaries import substitute_recycled_temporaries


def _name(s: sy.Basic) -> str:
    return str(s).replace("'", "")


class GRChomboGenerator(DslGenerator[GRChomboComputeClass]):
    """
    Generates a GRChombo compute class from a GRChomboComputeClass frontend with one baked function. The function's
    equations become the body of compute(Cell<data_t>): one pointer per state variable read, the scalar temporaries in
    the baked order, and a store_vars per written rhs component.
    """

    def generate_compute_class(self, banner: str = '') -> ComputeClassDecl:
        frontend = self.frontend
        if len(frontend.functions) != 1:
            raise GeneratorException("A GRChombo compute class needs exactly one function.")
        fn = next(iter(frontend.functions.values()))
        assert fn.been_baked

        complex_ = fn.eqn_complex
        complex_._calc_tile_temps()
        complex_._calc_vars()

        if len(complex_.eqn_lists) != 1 or len(complex_.tile_temporaries) > 0:
            raise GeneratorException(
                f"{fn.name}: GRChombo calls compute() once per cell, so the function cannot be hard split "
                f"(got {len(complex_.eqn_lists)} loops)."
            )
        if len(frontend.global_temporaries) > 0:
            raise GeneratorException(f"{fn.name}: global temporaries are not supported by the GRChombo backend.")

        reads = sorted({_name(s) for s in complex_.read_decls} - self.vars_to_ignore)
        writes = sorted({_name(s) for s in complex_.write_decls} - self.vars_to_ignore)

        stored: dict[str, str] = dict()
        for w in writes:
            state = frontend.state_of_rhs_component(w)
            if state is None:
                raise GeneratorException(f"{fn.name}: {w} is written but is not the rhs of a state variable.")
            stored[w] = state
        if read_written := sorted(set(reads) & set(writes)):
            raise GeneratorException(
                f"{fn.name}: {read_written} are both written and read; GRChombo's compute() can only read "
                f"the input FArrayBox."
            )

        eqn_list = complex_.eqn_lists[0]
        subst = substitute_recycled_temporaries(eqn_list)
        reassigned = {s.eqn_idx for s in subst.substitutions}
        recycled_targets = {_name(s.new) for s in subst.substitutions}
        annotations = {str(lhs): ann for lhs, ann in fn.source_annotations.eqns[0].items()}
        read_set = set(reads)
        visitor = GRChomboSympyVisitor(
            stencil_fns={'stencil'},
            is_grid_var=lambda name: name in read_set
        )

        body: list[Stmt] = [PointerDecl(var, frontend.enum_prefix + var) for var in reads]
        for eqn_idx, (lhs, rhs) in enumerate(subst.eqns):
            lhs_name = _name(lhs)
            expr = visitor.visit(rhs)
            comment = annotations.get(lhs_name)
            if lhs_name in stored:
                body.append(StoreVar(frontend.enum_prefix + stored[lhs_name], expr, comment))
            else:
                body.append(TempDecl(lhs_name, expr, reassign=eqn_idx in reassigned,
                                     mutable=lhs_name in recycled_targets, comment=comment))

        params = [ComputeClassParam(p.name, p.cpp_type, p.init)
                  for p in sorted(frontend.params.values(), key=lambda p: p.name)
                  if sy.Symbol(p.name) in complex_.params]  # type: ignore[no-untyped-call]

        return ComputeClassDecl(
            name=frontend.name,
            namespace=frontend.namespace,
            includes=frontend.includes,
            ctor_args=[(a.cpp_type, a.name, a.default) for a in frontend.ctor_args],
            dx=frontend.dx,
            ctor_checks=[(c.failure_condition, c.message) for c in frontend.ctor_checks],
            params=params,
            body=body,
            banner=banner
        )
