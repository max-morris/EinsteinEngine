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

from collections import OrderedDict, defaultdict
from typing import Literal, TypeAlias

from EinsteinEngine.generators.generator_exception import GeneratorException

from EinsteinEngine.emit.code.common.code_tree import BinOp, IdExpr
from EinsteinEngine.generators.substitute_recycled_temporaries import substitute_recycled_temporaries

from EinsteinEngine.emit.tree import Identifier, LineComment
from EinsteinEngine.emit.ccl.schedule.schedule_tree import IntentRegion
from EinsteinEngine.frontend.dsl.f90.vanilla_f90_frontend import VanillaF90Frontend
from EinsteinEngine.common.util import OrderedSet
from EinsteinEngine.emit.code.f90.f90_tree import F90CodeRoot, VarDecl, PrimitiveType, IntentIn, F90TopLevelNode
from EinsteinEngine.emit.code.common.code_tree import BinOpExpr, IntLiteralExpr, FloatLiteralExpr, FunctionCall, ExprStmt
from EinsteinEngine.emit.code.f90.f90_sympy_visitor import F90SympyVisitor
from EinsteinEngine.emit.code.f90.f90_tree import F90CodeElem, SubroutineDecl, TypeSpecifier, Dimension, DoLoop, Assignment, F90ExprNode
from EinsteinEngine.generators.dsl_generator import DslGenerator

import sympy as sy

from EinsteinEngine.emit.code.f90.f90_tree import IntentOut
from EinsteinEngine.emit.code.f90.f90_tree import Allocatable, ArrayAccess

ThornFnName: TypeAlias = str
SymbolName: TypeAlias = str

class VanillaF90Generator(DslGenerator[VanillaF90Frontend]):
    grid_names: OrderedSet[str]
    read_decls: dict[ThornFnName, OrderedDict[SymbolName, IntentRegion]]
    write_decls: dict[ThornFnName, OrderedDict[SymbolName, IntentRegion]]
    local_temp_names: dict[ThornFnName, OrderedSet[SymbolName]]
    tile_temp_names: dict[ThornFnName, OrderedSet[SymbolName]]

    vars_to_ignore: set[str] = {'t', 'x', 'y', 'z', 'DXI', 'DYI', 'DZI'}

    def __init__(self, frontend: VanillaF90Frontend):
        super().__init__(frontend)
        self.grid_names = OrderedSet()
        self.read_decls = defaultdict(OrderedDict)
        self.write_decls = defaultdict(OrderedDict)
        self.local_temp_names = defaultdict(OrderedSet)
        self.tile_temp_names = defaultdict(OrderedSet)

        for tf in self.frontend.functions.values():
            tf.eqn_complex._calc_tile_temps()
            tf.eqn_complex._calc_vars()

            for symbol, region in tf.eqn_complex.read_decls.items():
                var_name = str(symbol).replace("'", "")
                if var_name not in self.vars_to_ignore:
                    self.grid_names.add(var_name)
                    self.read_decls[tf.name][var_name] = region

            for symbol, region in tf.eqn_complex.write_decls.items():
                var_name = str(symbol).replace("'", "")
                if var_name not in self.vars_to_ignore:
                    self.grid_names.add(var_name)
                    self.write_decls[tf.name][var_name] = region

            for tt in tf.eqn_complex.tile_temporaries:
                self.tile_temp_names[tf.name].add(str(tt).replace("'", ""))

            local_temps = tf.eqn_complex.temporaries - tf.eqn_complex.tile_temporaries

            for lt in local_temps:
                self.local_temp_names[tf.name].add(lt_name := str(lt).replace("'", ""))
                assert lt_name not in self.tile_temp_names[tf.name]

    def _mk_sympy_visitor(self, fn_name: ThornFnName) -> F90SympyVisitor:
        def should_inject_array_access(name: str, in_stencil_args: bool) -> bool:
            return not in_stencil_args and (
                    name in self.grid_names
                    or name in self.tile_temp_names[fn_name]
                    or name in self.local_temp_names[fn_name]
            )

        return F90SympyVisitor(
            stencil_fns={'stencil'},
            should_inject_array_access=should_inject_array_access
        )

    def generate_function_code(self, which_fn: str) -> F90CodeRoot:
        nodes: list[F90CodeElem] = list()
        fn = self.frontend.functions[which_fn]
        fn_name: str = fn.name

        assert fn.been_baked

        if len(self.frontend.global_temporaries) > 0:
            raise GeneratorException("Global temporaries are not supported in vanilla f90")

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
                    )
                )
            ),
            VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Integer,
                    attributes=[]
                ),
                names=list(
                    Identifier(s) for s in (
                        'i',
                        'j',
                        'k',
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
            ),
            VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Double,
                    attributes=[]
                ),
                names=list(
                    Identifier(s) for s in (
                        'DXI',
                        'DYI',
                        'DZI'
                    )
                )
            )
        ]

        boilerplate_inits = [
            Assignment(
                Identifier(f'D{d.upper()}I'),
                None,
                BinOpExpr(
                    FloatLiteralExpr(1.0),
                    BinOp.Div,
                    IdExpr(Identifier(f'd{d}'))
                )
            ) for d in ('x', 'y', 'z')
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
            ) for var in self.read_decls[fn_name].keys()),
            *(VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Double,
                    attributes=[IntentOut(), Dimension((None, None, None))]
                ),
                names=[Identifier(var)],
            ) for var in self.write_decls[fn_name].keys()),
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
        temp_allocs: list[F90TopLevelNode] = list()

        loop_to_output_region = [
            self._get_output_region_for_loop(fn, self.grid_names, loop_idx)
            for loop_idx, _ in enumerate(fn.eqn_complex.eqn_lists)
        ]

        loops: list[F90TopLevelNode] = list()
        sympy_visitor = self._mk_sympy_visitor(fn.name)

        def _add_temp(temp_name: str) -> None:
            temp_decls.append(VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Double,
                    attributes=[Allocatable(), Dimension((None, None, None))]
                ),
                names=[Identifier(temp_name)],
            ))

            temp_allocs.append(
                ExprStmt(
                    FunctionCall(
                        Identifier('ALLOCATE'),
                        [
                            ArrayAccess(
                                Identifier(temp_name),
                                [IdExpr(Identifier(s)) for s in ('nx', 'ny', 'nz')],
                            )
                        ],
                        []
                    )
                )
            )

        for tile_temp in fn.eqn_complex.tile_temporaries:
            assert str(tile_temp) not in self.grid_names and str(tile_temp) not in self.local_temp_names[fn_name]
            _add_temp(str(tile_temp))

        for loop_idx, eqn_list in enumerate(fn.eqn_complex.eqn_lists):
            output_region = loop_to_output_region[loop_idx]
            subst_result = substitute_recycled_temporaries(eqn_list)
            reassigned_lhses = {subst.eqn_idx: subst for subst in subst_result.substitutions}

            def _resolve_overwrite(s: sy.Symbol) -> sy.Symbol:
                return s if "'" not in str(s) else sy.Symbol(str(s).replace("'", ""))  # type: ignore[no-untyped-call]

            eqns: list[tuple[sy.Symbol, F90ExprNode]] = [(_resolve_overwrite(lhs), sympy_visitor.visit(rhs)) for lhs, rhs in subst_result.eqns]
            annotations: dict[str, str] = {str(lhs): ann for lhs, ann in fn.source_annotations.eqns[loop_idx].items()}
            temporaries = [
                str(lhs) for lhs in OrderedSet(eqn_list.eqns.keys())
                if lhs in (eqn_list.temporaries - self.frontend.global_temporaries - fn.eqn_complex.tile_temporaries) and str(lhs) not in self.grid_names
            ]

            for temp_name in temporaries:
                _add_temp(temp_name)

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

            def mk_loop_bounds(direction: Literal['x', 'y', 'z']) -> tuple[F90ExprNode, F90ExprNode]:
                if output_region == IntentRegion.Interior:
                    return (
                        BinOpExpr(IdExpr(Identifier(f'ng{direction}')), BinOp.Add, IntLiteralExpr(1)),
                        BinOpExpr(IdExpr(Identifier(f'n{direction}')), BinOp.Sub, IdExpr(Identifier(f'ng{direction}')))
                    )
                elif output_region == IntentRegion.Everywhere:
                    return (
                        IntLiteralExpr(1),
                        IdExpr(Identifier(f'n{direction}'))
                    )
                else:
                    raise GeneratorException(f"Unsupported output region: {output_region}")

            k_bounds, j_bounds, i_bounds = mk_loop_bounds('z'), mk_loop_bounds('y'), mk_loop_bounds('x')

            loops.append(
                DoLoop(
                    Identifier('k'),
                    *k_bounds,
                    step=None,
                    body=[
                        DoLoop(
                            Identifier('j'),
                            *j_bounds,
                            step=None,
                            body=[
                                DoLoop(
                                    Identifier('i'),
                                    *i_bounds,
                                    step=None,
                                    body=loop_body
                                )
                            ]
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
                        'dt',   # time step size
                        *self.grid_names,
                        *self.frontend.params.keys()
                    )
                ],
                body=[
                    *boilerplate_decls,
                    *grid_decls,
                    *param_decls,
                    *temp_decls,
                    *boilerplate_inits,
                    *temp_allocs,
                    *loops
                ]
            )
        )

        return F90CodeRoot(nodes)

