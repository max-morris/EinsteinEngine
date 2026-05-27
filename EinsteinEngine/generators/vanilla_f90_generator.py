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
from dataclasses import dataclass
from typing import Literal, TypeAlias, Sequence, Optional

from EinsteinEngine.generators.generator_exception import GeneratorException

from EinsteinEngine.emit.code.common.code_tree import BinOp, IdExpr
from EinsteinEngine.generators.substitute_recycled_temporaries import substitute_recycled_temporaries

from EinsteinEngine.emit.tree import Identifier, LineComment
from EinsteinEngine.emit.ccl.schedule.schedule_tree import IntentRegion
from EinsteinEngine.frontend.dsl.f90.vanilla_f90_frontend import VanillaF90Module, VanillaF90Param
from EinsteinEngine.common.util import OrderedSet
from EinsteinEngine.emit.code.f90.f90_tree import *
from EinsteinEngine.emit.code.common.code_tree import BinOpExpr, IntLiteralExpr, FloatLiteralExpr, FunctionCall, ExprStmt
from EinsteinEngine.emit.code.f90.f90_sympy_visitor import F90SympyVisitor
from EinsteinEngine.emit.code.f90.f90_tree import F90CodeElem, SubroutineDecl, TypeSpecifier, Dimension, DoLoop, Assignment, F90ExprNode
from EinsteinEngine.generators.dsl_generator import DslGenerator

import sympy as sy

from EinsteinEngine.emit.code.f90.f90_tree import IntentOut
from EinsteinEngine.emit.code.f90.f90_tree import Allocatable, ArrayAccess
from EinsteinEngine.common.util import flatten
from EinsteinEngine.emit.code.f90.f90_tree import F90Decl

ThornFnName: TypeAlias = str
SymbolName: TypeAlias = str

class VanillaF90Generator(DslGenerator[VanillaF90Module]):
    grid_names: OrderedSet[str]
    read_decls: dict[ThornFnName, OrderedDict[SymbolName, IntentRegion]]
    write_decls: dict[ThornFnName, OrderedDict[SymbolName, IntentRegion]]
    params: dict[ThornFnName, OrderedDict[SymbolName, VanillaF90Param]]
    local_temp_names: dict[ThornFnName, OrderedSet[SymbolName]]
    tile_temp_names: dict[ThornFnName, OrderedSet[SymbolName]]

    function_pieces: OrderedDict[str, 'VanillaF90Generator.FunctionPieces']

    vars_to_ignore: set[str] = {'t', 'x', 'y', 'z', 'DXI', 'DYI', 'DZI'}

    def __init__(self, frontend: VanillaF90Module):
        super().__init__(frontend)
        self.grid_names = OrderedSet()
        self.read_decls = defaultdict(OrderedDict)
        self.write_decls = defaultdict(OrderedDict)
        self.params = defaultdict(OrderedDict)
        self.local_temp_names = defaultdict(OrderedSet)
        self.tile_temp_names = defaultdict(OrderedSet)

        for tf in self.frontend.functions.values():
            tf.eqn_complex._calc_tile_temps()
            tf.eqn_complex._calc_vars()

            for symbol, region in sorted(tf.eqn_complex.read_decls.items(), key=lambda kv: str(kv[0])):
                var_name = str(symbol).replace("'", "")
                if var_name not in self.vars_to_ignore:
                    self.grid_names.add(var_name)
                    self.read_decls[tf.name][var_name] = region

            for symbol, region in sorted(tf.eqn_complex.write_decls.items(), key=lambda kv: str(kv[0])):
                var_name = str(symbol).replace("'", "")
                if var_name not in self.vars_to_ignore:
                    self.grid_names.add(var_name)
                    self.write_decls[tf.name][var_name] = region

            for symbol in sorted(tf.eqn_complex.params, key=str):
                var_name = (sym_name := str(symbol)).replace("'", "")
                if var_name not in self.vars_to_ignore:
                    self.params[tf.name][var_name] = self.frontend.params[sym_name]

            for tt in tf.eqn_complex.tile_temporaries:
                self.tile_temp_names[tf.name].add(str(tt).replace("'", ""))

            local_temps = tf.eqn_complex.temporaries - tf.eqn_complex.tile_temporaries

            for lt in local_temps:
                self.local_temp_names[tf.name].add(lt_name := str(lt).replace("'", ""))
                assert lt_name not in self.tile_temp_names[tf.name]

        self.function_pieces = self._generate_all_function_pieces()

    def _mk_sympy_visitor(self, fn_name: ThornFnName) -> F90SympyVisitor:
        def should_inject_array_access(name: str, in_stencil_args: bool) -> bool:
            return not in_stencil_args and (
                    name in self.grid_names
                    or name in self.tile_temp_names[fn_name]
            )

        return F90SympyVisitor(
            stencil_fns={'stencil'},
            should_inject_array_access=should_inject_array_access
        )

    @dataclass
    class FunctionPieces:
        name: Identifier
        args: Sequence[Identifier]
        param_args: Sequence[Identifier]
        decls: Sequence[F90Decl]
        param_decls: Sequence[F90Decl]
        inits: Sequence[F90TopLevelNode]
        loops: Sequence[F90TopLevelNode]
        destructs: Sequence[F90TopLevelNode]

    def _generate_function_pieces(self, which_fn: str) -> FunctionPieces:
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

        grid_dimensions = Dimension(tuple(IdExpr(Identifier(s)) for s in ('nx', 'ny', 'nz')))

        grid_decls = [
            *(VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Double,
                    attributes=[IntentIn(), grid_dimensions]
                ),
                names=[Identifier(var)],
            ) for var in self.read_decls[fn_name].keys()),
            *(VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Double,
                    attributes=[IntentOut(), grid_dimensions]
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
            ) for var, param in self.params[fn_name].items())
        ]

        grid_temp_decls: list[VarDecl] = list()
        grid_temp_allocs: list[F90TopLevelNode] = list()
        grid_temp_deallocs: list[IdExpr] = list()

        loop_to_output_region = [
            self._get_output_region_for_loop(fn, self.grid_names, loop_idx)
            for loop_idx, _ in enumerate(fn.eqn_complex.eqn_lists)
        ]

        loops: list[F90TopLevelNode] = list()
        sympy_visitor = self._mk_sympy_visitor(fn.name)

        def _add_grid_temp(temp_name: str) -> None:
            grid_temp_decls.append(VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Double,
                    attributes=[Allocatable(), grid_dimensions]
                ),
                names=[Identifier(temp_name)],
            ))

            grid_temp_allocs.append(
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

            grid_temp_deallocs.append(IdExpr(Identifier(temp_name)))

        for tile_temp in fn.eqn_complex.tile_temporaries:
            assert str(tile_temp) not in self.grid_names and str(tile_temp) not in self.local_temp_names[fn_name]
            _add_grid_temp(str(tile_temp))

        for loop_idx, eqn_list in enumerate(fn.eqn_complex.eqn_lists):
            output_region = loop_to_output_region[loop_idx]
            subst_result = substitute_recycled_temporaries(eqn_list)
            reassigned_lhses = {subst.eqn_idx: subst for subst in subst_result.substitutions}

            def _resolve_overwrite(s: sy.Symbol) -> sy.Symbol:
                return s if "'" not in str(s) else sy.Symbol(str(s).replace("'", ""))  # type: ignore[no-untyped-call]

            eqns: list[tuple[sy.Symbol, F90ExprNode]] = [(_resolve_overwrite(lhs), sympy_visitor.visit(rhs)) for lhs, rhs in subst_result.eqns]
            annotations: dict[str, str] = {str(lhs): ann for lhs, ann in fn.source_annotations.eqns[loop_idx].items()}
            local_temporaries = [
                str(lhs) for lhs in OrderedSet(eqn_list.eqns.keys())
                if lhs in (eqn_list.temporaries - self.frontend.global_temporaries - fn.eqn_complex.tile_temporaries) and str(lhs) not in self.grid_names
            ]

            loop_body: list[F90TopLevelNode] = list()

            loop_body.append(VarDecl(
                type=TypeSpecifier(
                    type=PrimitiveType.Double,
                    attributes=[]
                ),
                names=[Identifier(temp_name) for temp_name in local_temporaries]
            ))

            if (loop_annotation := fn.source_annotations.loops[loop_idx]) != '':
                loops.append(LineComment(loop_annotation))

            for i, (lhs, rhs) in enumerate(eqns):
                if (lhs_name := str(lhs)) in annotations:
                    loop_body.append(LineComment(annotations[lhs_name]))

                if lhs_name in local_temporaries:
                    dims = None
                else:
                    dims = tuple(IdExpr(Identifier(x)) for x in ('i', 'j', 'k'))

                loop_body.append(
                    Assignment(
                        Identifier(str(lhs)),
                        dims,
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
                LineComment(
                    '$OMP PARALLEL DO DEFAULT(SHARED) COLLAPSE(2)'
                )
            )
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
                                    body=[Block(loop_body)]
                                )
                            ]
                        )
                    ]
                )
            )
            loops.append(
                LineComment(
                    '$OMP END PARALLEL DO'
                )
            )

        param_names = sorted(map(str, self.params[fn_name].keys()))
        deallocate_stmt = ExprStmt(FunctionCall(Identifier('DEALLOCATE'), list(grid_temp_deallocs), []))

        return self.FunctionPieces(
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
                    *self.read_decls[fn_name].keys(),
                    *self.write_decls[fn_name].keys(),
                )
            ],
            param_args=[
                Identifier(s) for s in param_names
            ],
            decls=[
                *boilerplate_decls,
                *grid_decls,
                *grid_temp_decls
            ],
            param_decls=[
                *param_decls,
            ],
            inits=[
                *boilerplate_inits,
                *grid_temp_allocs
            ],
            loops=loops,
            destructs=[
                deallocate_stmt
            ]
        )

    def _generate_all_function_pieces(self) -> OrderedDict[str, FunctionPieces]:
        od = OrderedDict()

        for fn_name in sorted(self.frontend.functions.keys()):
            od[fn_name] = self._generate_function_pieces(fn_name)

        return od

    def generate_standalone_subroutine_code(self, which_fn: str) -> F90CodeRoot:
        pieces = self.function_pieces[which_fn]

        return F90CodeRoot(
            [
                SubroutineDecl(
                    name=pieces.name,
                    args=[*pieces.args, *pieces.param_args],
                    body=[
                        *pieces.decls,
                        *pieces.param_decls,
                        *pieces.inits,
                        *pieces.loops,
                        *pieces.destructs
                    ]
                )
            ]
        )

    def generate_module_code(self) -> F90CodeRoot:
        fn_names = list(self.function_pieces.keys())
        assert sorted(fn_names) == fn_names

        param_decls: list[VarDecl] = list(
            VarDecl(
                type=TypeSpecifier(
                    type=param.get_type(),
                    attributes=[Public()]
                ),
                names=[Identifier(var)],
            ) for var, param in flatten(self.params[fn_name].items() for fn_name in fn_names)
        )

        interfaces = [
            ModuleInterface(
                name=None,
                decls=[
                    ModuleSubroutineDecl(
                        name=fn.name,
                        args=fn.args,
                        body=fn.decls
                    )
                    for fn in self.function_pieces.values()
                ]
            )
        ]

        return F90CodeRoot(
            [
                Module(
                    name=Identifier(self.frontend.name),
                    decls=[
                        ImplicitNone(),
                        *param_decls
                    ],
                    interfaces=interfaces,
                )
            ]
        )

    def generate_submodule_code(self, which_fn: str) -> F90CodeRoot:
        pieces = self.function_pieces[which_fn]
        mod_name = self.frontend.name

        return F90CodeRoot(
            [
                Submodule(
                    name=Identifier(f'{mod_name}_{pieces.name.identifier}'),
                    parent=Identifier(mod_name),
                    decls=[
                        ImplicitNone()
                    ],
                    procedures=[
                        ModuleProcedureDecl(
                            name=pieces.name,
                            body=[
                                *pieces.inits,
                                *pieces.loops,
                                *pieces.destructs
                            ]
                        )
                    ]
                )
            ]
        )

