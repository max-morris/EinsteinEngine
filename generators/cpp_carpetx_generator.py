import typing

from dsl.use_indices import ThornDef, ThornFunction, ScheduleBin
from emit.ccl.interface.interface_tree import *
from emit.ccl.param.param_tree import *
from emit.ccl.schedule.schedule_tree import *
from emit.code.code_tree import *
from emit.tree import String, Identifier, Bool, Integer, Float, Language, Verbatim
from generators.cactus_generator import CactusGenerator
from generators.generator_exception import GeneratorException
from typing import Optional, List, Set


class CppCarpetXGenerator(CactusGenerator):
    boilerplate_includes: List[Identifier] = [Identifier(s) for s in
                                              ["cctk.h", "cctk_Arguments.h", "cctk_Parameters.h",
                                               "loop_device.hxx", "simd.hxx", "cmath", "tuple"]]

    boilerplate_namespace_usings: List[Identifier] = [Identifier(s) for s in ["Arith", "Loop"]]

    boilerplate_usings: List[Identifier] = [Identifier(s) for s in ["std::cbrt", "std::fmax", "std::fmin", "std::sqrt"]]

    boilerplate_div_macros: str = """
        #define CARPETX_GF3D5
        #define divx(GF) (GF(GF ## _layout, p.I + p.DI[0]) - GF(GF ## _layout, p.I - p.DI[0]))/(2*CCTK_DELTA_SPACE(0))
        #define divy(GF) (GF(GF ## _layout, p.I + p.DI[1]) - GF(GF ## _layout, p.I - p.DI[1]))/(2*CCTK_DELTA_SPACE(1))
        #define divz(GF) (GF(GF ## _layout, p.I + p.DI[2]) - GF(GF ## _layout, p.I - p.DI[2]))/(2*CCTK_DELTA_SPACE(2))
    """.strip().replace('    ', '')

    def __init__(self, thorn_def: ThornDef) -> None:
        super().__init__(thorn_def)

    def get_src_file_name(self, which_fn: str) -> str:
        assert which_fn in self.thorn_def.thorn_functions

        return f'{self.thorn_def.name}_{which_fn}.cpp'

    def generate_makefile(self) -> str:
        srcs = [self.get_src_file_name(fn_name) for fn_name in self.thorn_def.thorn_functions.keys()]

        return f'SRCS = {" ".join(srcs)}\n\nSUBDIRS = '

    def generate_schedule_ccl(self) -> ScheduleRoot:
        storage_lines: list[StorageLine] = list()
        schedule_blocks: list[ScheduleBlock] = list()

        for group in self.variable_groups.keys():
            storage_lines.append(StorageLine([
                StorageDecl(
                    Identifier(group),
                    Integer(1)
                )
            ]))

        for fn_name, fn in self.thorn_def.thorn_functions.items():
            schedule_bin: Identifier
            if fn.schedule_bin is ScheduleBin.INIT:
                schedule_bin = Identifier('initial')
            elif fn.schedule_bin is ScheduleBin.ANALYSIS:
                schedule_bin = Identifier('analysis')
            else:
                assert fn.schedule_bin is ScheduleBin.EVOL
                schedule_bin = Identifier('ODESolvers_RHS')

            reads: list[Intent] = list()
            writes: list[Intent] = list()

            for var, spec in fn.eqnlist.read_decls.items():
                if var in fn.eqnlist.inputs and (var_name := str(var)) not in self.vars_to_ignore:
                    reads.append(Intent(
                        name=Identifier(var_name),
                        region=spec
                    ))

            for var, spec in fn.eqnlist.write_decls.items():
                if var in fn.eqnlist.outputs and (var_name := str(var)) not in self.vars_to_ignore:
                    writes.append(Intent(
                        name=Identifier(var_name),
                        region=spec
                    ))

            schedule_blocks.append(ScheduleBlock(
                group_or_function=GroupOrFunction.Function,
                name=Identifier(fn_name),
                at_or_in=AtOrIn.At if fn.schedule_bin.is_builtin else AtOrIn.In,
                schedule_bin=schedule_bin,
                description=String(fn_name),
                lang=Language.C,
                reads=reads,
                writes=writes
            ))

        return ScheduleRoot(
            storage_section=StorageSection(storage_lines),
            schedule_section=ScheduleSection(schedule_blocks)
        )

    def generate_interface_ccl(self) -> InterfaceRoot:
        return InterfaceRoot(
            HeaderSection(
                implements=Identifier(self.thorn_def.name),
                inherits=[],
                friends=[]
            ),
            IncludeSection([]),
            FunctionSection([]),
            VariableSection(list(self.variable_groups.values()))
        )

    def generate_param_ccl(self) -> ParamRoot:
        params: list[Param] = list()

        for param_name, param_def in self.thorn_def.params.items():
            py_param_type: type = param_def.get_type()
            py_param_range = param_def.values
            py_param_default = param_def.default

            param_type: ParamType
            param_range: Optional[ParamRange]
            param_default: String | Integer | Float | Bool

            if py_param_type is set:
                param_type = ParamType.Keyword

                assert type(py_param_range) is set[str]
                param_range = KeywordParamRange([String(s) for s in py_param_range], String(''))

                assert type(py_param_default) is str
                param_default = String(py_param_default)
            elif py_param_type is str:
                param_type = ParamType.String

                if py_param_range is None:
                    param_range = StringParamRange([String('')], String(''))
                else:
                    assert type(py_param_range) is str
                    param_range = StringParamRange([String(py_param_range)], String(''))

                assert type(py_param_default) is str
                param_default = String(py_param_default)
            elif py_param_type is int:
                param_type = ParamType.Int

                if py_param_range is None:
                    param_range = IntParamRange(IntParamDescWildcard(), String(''))
                else:
                    assert type(py_param_range) is tuple[int, int]
                    lo_i, hi_i = typing.cast(tuple[int, int], py_param_range)
                    param_range = IntParamRange(IntParamDescRange(
                        IntParamOpenLowerBound(Integer(lo_i)),
                        IntParamOpenUpperBound(Integer(hi_i))
                    ), String(''))

                assert type(py_param_default) is int
                param_default = Integer(py_param_default)
            elif py_param_type is float:
                param_type = ParamType.Real

                if py_param_range is None:
                    param_range = RealParamRange(RealParamDescWildcard(), String(''))
                else:
                    assert type(py_param_range) is tuple[float, float]
                    lo_f, hi_f = typing.cast(tuple[float, float], py_param_range)
                    param_range = RealParamRange(RealParamDescRange(
                        RealParamOpenLowerBound(Float(lo_f)),
                        RealParamOpenUpperBound(Float(hi_f))
                    ), String(''))

                assert type(py_param_default) is float
                param_default = Float(py_param_default)
            elif py_param_type is bool:
                param_type = ParamType.Bool
                param_range = None

                assert type(py_param_default) is bool
                param_default = Bool(py_param_default)
            else:
                raise GeneratorException(f"Didn't expect parameter type {py_param_type}")

            assert param_type is not None
            assert param_range is not None or param_type is ParamType.Bool

            params.append(Param(
                param_access=ParamAccess.Restricted,
                param_type=param_type,
                param_name=Identifier(param_name),
                param_desc=String(param_def.desc),
                range_descriptions=[param_range] if param_range is not None else [],
                default_value=param_default
            ))

        return ParamRoot(params)

    def generate_function_code(self, which_fn: str) -> CodeRoot:
        nodes: list[CodeElem] = list()
        thorn_fn: ThornFunction = self.thorn_def.thorn_functions[which_fn]
        fn_name: str = thorn_fn.name

        # div{x,y,z} macros
        nodes.append(Verbatim(self.boilerplate_div_macros))

        # Includes, usings...
        for include in self.boilerplate_includes:
            nodes.append(IncludeDirective(include))

        for ns in self.boilerplate_namespace_usings:
            nodes.append(UsingNamespace(ns))

        nodes.append(Using(self.boilerplate_usings))

        # Each variable needs to have a corresponding decl of the form
        # `const GF3D5layout ${VAR_NAME}_layout(${LAYOUT_NAME}_layout);`
        # for the div macros to work; layout here really means centering.

        decls: list[CodeElem] = list()
        declared_layouts: set[Centering] = set()

        for var_name in self.var_names:
            var_centering: Optional[Centering]

            # Try looking up the var's centering directly...
            if (var_centering := self.thorn_def.centering.get(var_name, None)) is not None:
                pass
            # Otherwise, try looking it up by the var's base...
            elif (var_base := self.thorn_def.var2base.get(var_name, None)) is not None:
                var_centering = self.thorn_def.centering.get(var_base, None)

            # If this var doesn't have a defined centering, skip it.
            if var_centering is None:
                continue

            assert var_centering is not None

            # Make sure the referenced layout has a preceding, corresponding decl of the form
            # `const GF3D5layout ${LAYOUT_NAME}_layout(cctkGH, {$I, $J, $K});`
            if var_centering not in declared_layouts:
                declared_layouts.add(var_centering)

                i, j, k = var_centering.int_repr
                centering_init_list = f'{{{i}, {j}, {k}}}'

                decls.append(ConstConstructDecl(
                    Identifier('GF3D5layout'),
                    Identifier(f'{var_centering.string_repr}_layout'),
                    [IdExpr(Identifier('cctkGH')), VerbatimExpr(Verbatim(centering_init_list))]
                ))

            # Now build the var's centering decl.
            decls.append(ConstConstructDecl(
                Identifier('GF3D5layout'),
                Identifier(f'{var_name}_layout'),
                [IdExpr(Identifier(f'{var_centering.string_repr}_layout'))]
            ))

        # Figure out which centering to pass to grid.loop_int_device<...>
        # All of this function's outputs need to have the same centering. If they do, use that centering.
        bases_of_outputs = {self.thorn_def.var2base[output_var] for output_var in
                            {str(output) for output in thorn_fn.eqnlist.outputs} if
                            output_var in self.thorn_def.var2base}
        output_centerings = {self.thorn_def.centering[base] for base in bases_of_outputs}

        if None in output_centerings or len(output_centerings) == 0:
            raise GeneratorException("All output vars must have a centering.")

        if len(output_centerings) > 1:
            raise GeneratorException(f"Output vars have mixed centerings: {output_centerings}")

        # Got the centering.
        output_centering: Centering
        [output_centering] = typing.cast(Set[Centering], output_centerings)

        # x, y, and z are special
        xyz_decls = [
            ConstAssignDecl(Identifier('auto&'), Identifier(s), IdExpr(Identifier(f'p.{s}'))) for s in ['x', 'y', 'z']
        ]

        # Build the function decl and its body.
        nodes.append(
            ThornFunctionDecl(
                Identifier(fn_name),
                [DeclareCarpetXArgs(Identifier(fn_name)),
                 DeclareCarpetParams(),
                 *decls,
                 CarpetXGridLoopCall(
                     output_centering,
                     CarpetXGridLoopLambda(
                         xyz_decls,
                         {str(lhs): SympyExpr(rhs) for lhs, rhs in thorn_fn.eqnlist.eqns.items()},
                         []),
                 )]
            )
        )

        return CodeRoot(nodes)
