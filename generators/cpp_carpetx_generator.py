import typing

from dsl.use_indices import ThornDef, ThornFunction, ScheduleBin
from emit.ccl.interface.interface_tree import *
from emit.ccl.param.param_tree import *
from emit.ccl.schedule.schedule_tree import *
from emit.code.code_tree import *
from emit.tree import String, Identifier, Bool, Integer, Float, Language
from generators.cactus_generator import CactusGenerator
from generators.generator_exception import GeneratorException
from typing import Optional


class CppCarpetXGenerator(CactusGenerator):
    boilerplate_includes: list[Identifier] = [Identifier(s) for s in
                                              ["fixmath.hxx", "cctk.h", "cctk_Arguments.h", "cctk_Parameters.h",
                                               "loop_device.hxx", "simd.hxx", "cmath", "tuple"]]

    boilerplate_namespace_usings: list[Identifier] = [Identifier(s) for s in ["Arith", "Loop"]]

    boilerplate_usings: list[Identifier] = [Identifier(s) for s in ["std::cbrt", "std::fmax", "std::fmin", "std::sqrt"]]

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
            else:
                assert fn.schedule_bin is ScheduleBin.EVOL
                schedule_bin = Identifier('ODESolvers_RHS')

            reads: list[Intent] = list()
            writes: list[Intent] = list()

            for var, spec in fn.eqnlist.read_decls.items():
                if var in fn.eqnlist.inputs:
                    reads.append(Intent(
                        name=Identifier(str(var)),
                        region=spec
                    ))

            for var, spec in fn.eqnlist.write_decls.items():
                if var in fn.eqnlist.outputs:
                    writes.append(Intent(
                        name=Identifier(str(var)),
                        region=spec
                    ))

            schedule_blocks.append(ScheduleBlock(
                group_or_function=GroupOrFunction.Function,
                name=Identifier(fn_name),
                at_or_in=AtOrIn.At,
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
                param_access=ParamAccess.Global,
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

        for include in self.boilerplate_includes:
            nodes.append(IncludeDirective(include))

        for ns in self.boilerplate_namespace_usings:
            nodes.append(UsingNamespace(ns))

        nodes.append(Using(self.boilerplate_usings))

        bases_of_outputs = {self.thorn_def.base_of[base] for base in
                            {str(output) for output in thorn_fn.eqnlist.outputs} if base in self.thorn_def.base_of}
        output_centerings = {self.thorn_def.centering[base] for base in bases_of_outputs}

        if None in output_centerings or len(output_centerings) == 0:
            raise GeneratorException("All output vars must have a centering.")

        if len(output_centerings) > 1:
            raise GeneratorException(f"Output vars have mixed centerings: {output_centerings}")

        output_centering: Centering
        [output_centering] = typing.cast(set[Centering], output_centerings)

        nodes.append(
            ThornFunctionDecl(
                Identifier(fn_name),
                [DeclareCarpetXArgs(Identifier(fn_name)),
                 DeclareCarpetParams(),
                 CarpetXGridLoopCall(
                     output_centering,
                     CarpetXGridLoopLambda(
                         {str(lhs): SympyExpr(rhs) for lhs, rhs in thorn_fn.eqnlist.eqns.items()})
                 )]
            )
        )

        return CodeRoot(nodes)
