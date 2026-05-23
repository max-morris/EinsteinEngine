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

from collections import OrderedDict

from EinsteinEngine.emit.code.common.code_tree import BinOp, IdExpr
from EinsteinEngine.generators.substitute_recycled_temporaries import substitute_recycled_temporaries

from EinsteinEngine.emit.tree import Identifier, LineComment
from EinsteinEngine.emit.ccl.schedule.schedule_tree import IntentRegion
from EinsteinEngine.frontend.dsl.f90.vanilla_f90_frontend import VanillaF90Frontend
from EinsteinEngine.common.util import OrderedSet
from EinsteinEngine.emit.code.f90.f90_tree import F90CodeRoot, VarDecl, PrimitiveType, IntentIn, F90TopLevelNode
from EinsteinEngine.emit.code.common.code_tree import BinOpExpr, IntLiteralExpr
from EinsteinEngine.emit.code.f90.f90_sympy_visitor import F90SympyVisitor
from EinsteinEngine.emit.code.f90.f90_tree import F90CodeElem, SubroutineDecl, TypeSpecifier, Dimension, DoLoop, Assignment, F90ExprNode
from EinsteinEngine.generators.dsl_generator import DslGenerator

import sympy as sy


class VanillaF90Generator(DslGenerator[VanillaF90Frontend]):
    grid_names: OrderedSet[str]
    read_decls: OrderedDict[str, IntentRegion]
    write_decls: OrderedDict[str, IntentRegion]
    local_temp_names: OrderedSet[str]
    tile_temp_names: OrderedSet[str]

    _sympy_visitor: F90SympyVisitor

    vars_to_ignore: set[str] = {'t', 'x', 'y', 'z', 'DXI', 'DYI', 'DZI'}

    def __init__(self, frontend: VanillaF90Frontend):
        super().__init__(frontend)
        self.grid_names = OrderedSet()
        self.read_decls = OrderedDict()
        self.write_decls = OrderedDict()
        self.local_temp_names = OrderedSet()
        self.tile_temp_names = OrderedSet()

        for tf in self.frontend.functions.values():
            tf.eqn_complex._calc_tile_temps()
            tf.eqn_complex._calc_vars()

            for symbol, region in tf.eqn_complex.read_decls.items():
                var_name = str(symbol).replace("'", "")
                if var_name not in self.vars_to_ignore:
                    self.grid_names.add(var_name)
                    self.read_decls[var_name] = region

            for symbol, region in tf.eqn_complex.write_decls.items():
                var_name = str(symbol).replace("'", "")
                if var_name not in self.vars_to_ignore:
                    self.grid_names.add(var_name)
                    self.write_decls[var_name] = region

            for tt in tf.eqn_complex.tile_temporaries:
                self.tile_temp_names.add(str(tt).replace("'", ""))

            local_temps = tf.eqn_complex.temporaries - tf.eqn_complex.tile_temporaries

            for lt in local_temps:
                self.local_temp_names.add(lt_name := str(lt).replace("'", ""))
                assert lt_name not in self.tile_temp_names

            def should_inject_array_access(name: str, in_stencil_args: bool) -> bool:
                return not in_stencil_args and (name in self.grid_names or name in self.tile_temp_names)

            self._sympy_visitor = F90SympyVisitor(
                stencil_fns={'stencil'},
                should_inject_array_access=should_inject_array_access
            )

    def generate_function_code(self, which_fn: str) -> F90CodeRoot:
        nodes: list[F90CodeElem] = list()
        fn = self.frontend.functions[which_fn]
        fn_name: str = fn.name

        assert fn.been_baked

        boilerplate_decls = [
            VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Integer,
                    attributes=[IntentIn()]
                ),
                names=list(
                    Identifier(s) for s in (
                        'nx',
                        'ny',
                        'nz',
                        'ngx',
                        'ngy',
                        'ngz',
                        'i',  # loop indices
                        'j',  # "
                        'k'   # "
                    )
                )
            ),
            VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Double,
                    attributes=[IntentIn()]
                ),
                names=list(
                    Identifier(s) for s in (
                        'dx',
                        'dy',
                        'dz',
                        'dt',
                    )
                )
            )
        ]

        #base_to_grid_vars: OrderedDict[str, set[str]] = OrderedDict()
        #for name in self.grid_names:
        #    base_to_grid_vars.setdefault(self.frontend.var2base.get(name, name), set()).add(name)

        grid_decls = [
            *(VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Double,
                    attributes=[IntentIn(), Dimension((None, None, None))]
                ),
                names=[Identifier(var)],
            ) for var in self.read_decls.keys()),
            *(VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Double,
                    attributes=[IntentIn(), Dimension((None, None, None))]
                ),
                names=[Identifier(var)],
            ) for var in self.write_decls.keys()),
        ]

        param_decls = [
            *(VarDecl(
                type=TypeSpecifier(
                    type=param.get_type(),
                    attributes=[IntentIn()]
                ),
                names=[Identifier(var)],
            ) for var, param in self.frontend.params.items())
        ]

        temp_decls: list[VarDecl] = list()

        loop_to_output_region = [
            self._get_output_region_for_loop(fn, self.grid_names, loop_idx)
            for loop_idx, _ in enumerate(fn.eqn_complex.eqn_lists)
        ]

        loops: list[F90TopLevelNode] = list()
        for loop_idx, eqn_list in enumerate(fn.eqn_complex.eqn_lists):
            output_region = loop_to_output_region[loop_idx]
            subst_result = substitute_recycled_temporaries(eqn_list)
            reassigned_lhses = {subst.eqn_idx: subst for subst in subst_result.substitutions}

            def _resolve_overwrite(s: sy.Symbol) -> sy.Symbol:
                return s if "'" not in str(s) else sy.Symbol(str(s).replace("'", ""))  # type: ignore[no-untyped-call]

            eqns: list[tuple[sy.Symbol, F90ExprNode]] = [(_resolve_overwrite(lhs), self._sympy_visitor.visit(rhs)) for lhs, rhs in subst_result.eqns]
            annotations: dict[str, str] = {str(lhs): ann for lhs, ann in fn.source_annotations.eqns[loop_idx].items()}
            temporaries = [
                str(lhs) for lhs in OrderedSet(eqn_list.eqns.keys())
                if lhs in (eqn_list.temporaries - self.frontend.global_temporaries - eqn_list.tile_temporaries) and str(lhs) not in self.grid_names
            ]

            for temp_name in temporaries:
                temp_decls.append(VarDecl(
                    type=TypeSpecifier(
                        type=PrimitiveType.Double,
                        attributes=[]
                    ),
                    names=[Identifier(temp_name)],
                ))

            if (loop_annotation := fn.source_annotations.loops[loop_idx]) != '':
                loops.append(LineComment(loop_annotation))

            loop_body: list[F90TopLevelNode] = list()
            for i, (lhs, rhs) in enumerate(eqns):
                if (lhs_name := str(lhs)) in annotations:
                    loop_body.append(LineComment(annotations[lhs_name]))

                loop_body.append(
                    Assignment(
                        Identifier(str(lhs)),
                        tuple(IdExpr(Identifier(x)) for x in ('i', 'j', 'k')),
                        rhs
                    )
                )

            # todo: this assumes output_region of Interior
            loops.append(
                DoLoop(
                    Identifier('j'),
                    BinOpExpr(IdExpr(Identifier('ngy')), BinOp.Add, IntLiteralExpr(1)),
                    BinOpExpr(IdExpr(Identifier('ny')), BinOp.Sub, IdExpr(Identifier('ngy'))),
                    step=None,
                    body=[
                        DoLoop(
                            Identifier('i'),
                            BinOpExpr(IdExpr(Identifier('ngx')), BinOp.Add, IntLiteralExpr(1)),
                            BinOpExpr(IdExpr(Identifier('nx')), BinOp.Sub, IdExpr(Identifier('ngx'))),
                            step=None,
                            body=loop_body
                        )
                    ]
                )
            )

        nodes.append(
            SubroutineDecl(
                name=Identifier(fn_name),
                args=[
                    Identifier(s) for s in (
                        'nx',   # dimensions of the 3d arrays
                        'ny',   # "
                        'nz',   # "
                        'ngx',  # number of ghost zones on either side in each direction
                        'ngy',  # "
                        'ngz',  # "
                        'dx',   # grid spacing in each direction
                        'dy',   # "
                        'dz',   # "
                        'dt',    # time step size
                        *self.grid_names,
                        *self.frontend.params.keys()
                    )
                ],
                body=[
                    *boilerplate_decls,
                    *grid_decls,
                    *param_decls,
                    *temp_decls,
                    *loops
                ]
            )
        )

        return F90CodeRoot(nodes)

