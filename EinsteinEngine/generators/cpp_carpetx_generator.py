import functools
from dataclasses import dataclass
from typing import Optional, List, Collection, Protocol, NamedTuple

import sympy as sy
from sympy import IndexedBase, Indexed
from typing_extensions import Unpack, OrderedDict

from EinsteinEngine.dsl.carpetx import ExplicitSyncBatch, NewRadXBoundaryBatch
from EinsteinEngine.dsl.eqnlist import stencil
from EinsteinEngine.dsl.stencil_idx import StencilIdxWithCentering
from EinsteinEngine.dsl.use_indices import ThornDef, ThornFunction, ScheduleBin, ScheduleTarget
from EinsteinEngine.dsl.util import require
from EinsteinEngine.emit.ccl.interface.interface_tree import InterfaceRoot, HeaderSection, IncludeSection, FunctionSection, \
    VariableSection, UsesInclude
from EinsteinEngine.emit.ccl.param.param_tree import ParamRoot, Param, ParamAccess, ParamType, ParamRange, \
    KeywordParamRange, StringParamRange, IntParamRange, IntParamDescWildcard, IntParamDescRange, IntParamOpenLowerBound, \
    IntParamOpenUpperBound, RealParamRange, RealParamDescWildcard, RealParamDescRange, RealParamOpenLowerBound, \
    RealParamOpenUpperBound
from EinsteinEngine.emit.ccl.schedule.schedule_tree import ScheduleRoot, StorageLine, ScheduleBlock, StorageDecl, Intent, \
    GroupOrFunction, AtOrIn, StorageSection, ScheduleSection, IntentRegion
from EinsteinEngine.emit.code.code_tree import CodeRoot, CodeElem, IncludeDirective, UsingNamespace, Using, \
    ConstConstructDecl, IdExpr, VerbatimExpr, ConstAssignDecl, BinOpExpr, BinOp, FloatLiteralExpr, ThornFunctionDecl, \
    DeclareCarpetXArgs, DeclareCarpetParams, UsingAlias, ConstExprAssignDecl, CarpetXGridLoopCall, \
    CarpetXGridLoopLambda, ExprStmt, FunctionCall, IntLiteralExpr, MutableAssignDecl, Expr, IfElseStmt, Stmt
from EinsteinEngine.emit.code.sympy_visitor import SympyExprVisitor
from EinsteinEngine.emit.tree import String, Identifier, Bool, Integer, Float, Language, Verbatim, Centering
from EinsteinEngine.emit.util import encode_stencil_idx
from EinsteinEngine.generators.cactus_generator import CactusGenerator, CactusGeneratorOptions, SyncMode
from EinsteinEngine.generators.generator_exception import GeneratorException
from EinsteinEngine.generators.substitute_recycled_temporaries import substitute_recycled_temporaries
from EinsteinEngine.generators.util import VarCenteringFn
from EinsteinEngine.util import OrderedSet, wprint


@dataclass(frozen=True)
class _TileTempCenteringData:
    temps_by_centering: dict[Centering, set[sy.Symbol]]
    temps_to_centering: dict[sy.Symbol, Centering]
    final_loop_centerings: list[Centering]

class CppCarpetXGeneratorOptions(CactusGeneratorOptions, total=False):
    explicit_syncs: Collection[ExplicitSyncBatch]
    new_rad_x_boundary_fns: Collection[NewRadXBoundaryBatch]

class _HasName(Protocol):
    name: str


class CppCarpetXGenerator(CactusGenerator):
    _boilerplate_includes: List[Identifier] = [
        Identifier(s) for s in [
            "cctk.h", "cctk_Arguments.h", "cctk_Parameters.h",
            "loop_device.hxx", "simd.hxx", "defs.hxx", "vect.hxx",
            "cmath", "tuple"
        ]
    ]

    _boilerplate_quoted_includes: List[Identifier] = [
        Identifier(s) for s in [
            "../../../CarpetX/CarpetX/src/timer.hxx"
        ]
    ]

    _boilerplate_nv_tools_include: str = """
        #ifdef __CUDACC__
        #include <nvtx3/nvToolsExt.h>
        #endif
    """.strip().replace('    ', '')

    @staticmethod
    def _boilerplate_nv_tools_init(fn_name: str) -> str:
        return f"""
            #ifdef __CUDACC__
            const nvtxRangeId_t range = nvtxRangeStartA("{fn_name}");
            #endif
        """.strip().replace('    ', '')

    _boilerplate_nv_tools_destructor: str = """
        #ifdef __CUDACC__
        nvtxRangeEnd(range);
        #endif
    """.strip().replace('    ', '')

    _boilerplate_namespace_usings: List[Identifier] = [Identifier(s) for s in ["Arith", "Loop"]]
    _boilerplate_usings: List[Identifier] = [Identifier(s) for s in ["std::cbrt", "std::fmax", "std::fmin", "std::sqrt"]]

    # TODO: We want to be able to
    #  specify a header file with these
    #  or alternate defs.
    _boilerplate_setup: str = "#define CARPETX_GF3D5"
    _boilerplate_div_macros: str = """
        #define access(GF, IDX) (GF(p.mask, IDX))
        #define store(GF, IDX, VAL) (GF.store(p.mask, IDX, VAL))
        #define stencil(GF, IDX) (GF(p.mask, IDX))
        #define CCTK_ASSERT(X) if(!(X)) { CCTK_Error(__LINE__, __FILE__, CCTK_THORNSTRING, "Assertion Failure: " #X); }
    """.strip().replace('    ', '')

    options: CppCarpetXGeneratorOptions

    def __init__(self, thorn_def: ThornDef, **options: Unpack[CppCarpetXGeneratorOptions]) -> None:
        super().__init__(thorn_def, options)

        unbaked_fns = {name for name, fn in thorn_def.thorn_functions.items() if not fn.been_baked}
        if len(unbaked_fns) > 0:
            raise GeneratorException(f"One or more functions have not been baked. Namely: {unbaked_fns}")

    def get_fn_src_file_name(self, which_fn: str) -> str:
        assert which_fn in self.thorn_def.thorn_functions

        return f'{self.thorn_def.name}_{which_fn}.cpp'

    def get_explicit_src_file_name(self, which: _HasName | str) -> str:
        return f'{self.thorn_def.name}_{which if isinstance(which, str) else which.name}.cpp'

    def generate_makefile(self) -> str:
        srcs = [self.get_fn_src_file_name(fn_name) for fn_name in OrderedSet(self.thorn_def.thorn_functions.keys())]

        for sync_batch in self.options.get('explicit_syncs', list()):
            srcs.append(self.get_explicit_src_file_name(sync_batch))

        for rad_x_batch in self.options.get('new_rad_x_boundary_fns', list()):
            srcs.append(self.get_explicit_src_file_name(rad_x_batch))

        srcs.append(self.get_explicit_src_file_name(f'StateSync_{self.thorn_def.name}'))

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

        for fn_name, fn in sorted(self.thorn_def.thorn_functions.items()):
            schedule_bin, at_or_in = self._resolve_schedule_target(fn.schedule_target)

            reads: OrderedSet[Intent] = OrderedSet()
            writes: OrderedSet[Intent] = OrderedSet()
            sync_var_groups: OrderedSet[str] = OrderedSet()
            fn_group_name = f'{fn_name}_group'

            for var, spec in fn.eqn_complex.read_decls.items():
                if var in fn.eqn_complex.inputs and (var_name := str(var).replace("'", "")) not in self.vars_to_ignore:
                    qualified_var_name = self._get_qualified_var_name(var_name)

                    reads.add(Intent(
                        name=Identifier(qualified_var_name),
                        region=spec
                    ))

                    if spec is IntentRegion.Everywhere:
                        sync_var_groups.add(self._get_qualified_group_name_from_var_name(var_name))

            for var, spec in fn.eqn_complex.write_decls.items():
                if var in fn.eqn_complex.outputs and (var_name := str(var).replace("'", "")) not in self.vars_to_ignore:
                    qualified_var_name = self._get_qualified_var_name(var_name)
                    qualified_var_id = Identifier(qualified_var_name)

                    writes.add(Intent(
                        name=qualified_var_id,
                        region=spec
                    ))

            if fn.schedule_target is ScheduleBin.SpecialEvolve:
                reads.add(Intent(Identifier('ODESolvers::substep_counter'), None))

            schedule_blocks.append(ScheduleBlock(
                group_or_function=GroupOrFunction.Group,
                name=Identifier(fn_group_name),
                at_or_in=at_or_in,
                schedule_bin=schedule_bin,
                description=String(f'Group for function `{fn_name}`. Generated by EinsteinEngine.'),
                lang=Language.C,
                before=[Identifier(f'{s}_group') for s in fn.schedule_before],
                after=[Identifier(f'{s}_group') for s in fn.schedule_after]
            ))

            schedule_blocks.append(ScheduleBlock(
                group_or_function=GroupOrFunction.Function,
                name=Identifier(fn_name),
                at_or_in=AtOrIn.In,
                schedule_bin=Identifier(fn_group_name),
                description=String(f'Function `{fn_name}` generated by EinsteinEngine.'),
                lang=Language.C,
                reads=list(reads),
                writes=list(writes)
            ))

            if len(sync_var_groups) > 0 and self.options['sync_mode'] is SyncMode.EmulatePresync:
                syncs = [Identifier(g) for g in sync_var_groups]
                syncs_desc = ", ".join(s.identifier for s in syncs)
                schedule_blocks.append(ScheduleBlock(
                    group_or_function=GroupOrFunction.Function,
                    name=Identifier(f'StateSync_{self.thorn_def.name}'),
                    at_or_in=AtOrIn.In,
                    schedule_bin=Identifier(fn_group_name),
                    description=String(f'Empty function for explicitly SYNCing variables read Everywhere in function `{fn_name}`. Syncs: {syncs_desc}. Generated by EinsteinEngine.'),
                    lang=Language.C,
                    sync=syncs,
                    before=[Identifier(fn_name)]
                ))

            # Rancid hack: In CarpetX, Evolve DOES NOT run on step 0, while Analysis DOES. This breaks global temps
            #  if they happen to be initialized in Evolve then read in Analysis. Originally, we worked around this by
            #  identifying which synthetics are read in Analysis and precomputing them in PostInit. This didn't do the
            #  trick due to dependency issues with symbols defined in the recipe. Instead, we will just duplicate
            #  all Evolve and Analysis thorn functions into PostPostInit to ensure everything is initialized.
            if fn.schedule_target in [ScheduleBin.Evolve, ScheduleBin.SpecialEvolve]:
                post_init_bin, post_init_at_in = self._resolve_schedule_target(ScheduleBin.InitEvolve)
                fn_group_name = f'{fn_name}_init_evolve_group'

                schedule_blocks.append(ScheduleBlock(
                    group_or_function=GroupOrFunction.Group,
                    name=Identifier(fn_group_name),
                    at_or_in=post_init_at_in,
                    schedule_bin=post_init_bin,
                    description=String(f'Group for function `{fn_name}` in InitEvolve. Generated by EinsteinEngine.'),
                    lang=Language.C,
                    before=[Identifier(f'{s}_init_evolve_group') for s in fn.schedule_before],
                    after=[Identifier(f'{s}_init_evolve_group') for s in fn.schedule_after]
                ))

                schedule_blocks.append(ScheduleBlock(
                    group_or_function=GroupOrFunction.Function,
                    name=Identifier(fn_name),
                    at_or_in=AtOrIn.In,
                    schedule_bin=Identifier(fn_group_name),
                    description=String(f'Function `{fn_name}` in InitEvolve generated by EinsteinEngine.'),
                    lang=Language.C,
                    reads=list(reads),
                    writes=list(writes)
                ))

                if len(sync_var_groups) > 0 and self.options['sync_mode'] is SyncMode.EmulatePresync:
                    syncs = [Identifier(g) for g in sync_var_groups]
                    syncs_desc = ", ".join(s.identifier for s in syncs)
                    schedule_blocks.append(ScheduleBlock(
                        group_or_function=GroupOrFunction.Function,
                        name=Identifier(f'StateSync_{self.thorn_def.name}'),
                        at_or_in=AtOrIn.In,
                        schedule_bin=Identifier(fn_group_name),
                        description=String(f'Empty function for explicitly SYNCing variables read Everywhere in function `{fn_name}` in InitEvolve. Syncs: {syncs_desc}. Generated by EinsteinEngine.'),
                        lang=Language.C,
                        sync=syncs,
                        before=[Identifier(fn_name)]
                    ))

            if fn.schedule_target is ScheduleBin.Analysis:
                post_init_bin, post_init_at_in = self._resolve_schedule_target(ScheduleBin.InitAnalysis)
                fn_group_name = f'{fn_name}_init_analysis_group'

                schedule_blocks.append(ScheduleBlock(
                    group_or_function=GroupOrFunction.Group,
                    name=Identifier(fn_group_name),
                    at_or_in=post_init_at_in,
                    schedule_bin=post_init_bin,
                    description=String(f'Group for function `{fn_name}` in InitAnalysis. Generated by EinsteinEngine.'),
                    lang=Language.C,
                    before=[Identifier(f'{s}_init_analysis_group') for s in fn.schedule_before],
                    after=[Identifier(f'{s}_init_analysis_group') for s in fn.schedule_after]
                ))

                schedule_blocks.append(ScheduleBlock(
                    group_or_function=GroupOrFunction.Function,
                    name=Identifier(fn_name),
                    at_or_in=AtOrIn.In,
                    schedule_bin=Identifier(fn_group_name),
                    description=String(f'Function `{fn_name}` in InitAnalysis generated by EinsteinEngine.'),
                    lang=Language.C,
                    reads=list(reads),
                    writes=list(writes)
                ))

                if len(sync_var_groups) > 0 and self.options['sync_mode'] is SyncMode.EmulatePresync:
                    syncs = [Identifier(g) for g in sync_var_groups]
                    syncs_desc = ", ".join(s.identifier for s in syncs)
                    schedule_blocks.append(ScheduleBlock(
                        group_or_function=GroupOrFunction.Function,
                        name=Identifier(f'StateSync_{self.thorn_def.name}'),
                        at_or_in=AtOrIn.In,
                        schedule_bin=Identifier(fn_group_name),
                        description=String(f'Empty function for explicitly SYNCing variables read Everywhere in function `{fn_name}` in InitAnalysis. Syncs: {syncs_desc}. Generated by EinsteinEngine.'),
                        lang=Language.C,
                        sync=syncs,
                        before=[Identifier(fn_name)]
                    ))

        if 'extra_schedule_blocks' in self.options:
            for block in self.options['extra_schedule_blocks']:
                schedule_blocks.append(block)

        if (sync_batch_items := self.options.get('explicit_syncs', None)) is not None:
            for sync_batch in sync_batch_items:
                schedule_bin, at_or_in = self._resolve_schedule_target(sync_batch.schedule_target)
                var_names = [str(v) for v in sync_batch.vars]

                syncs = [Identifier(self._get_qualified_group_name_from_var_name(var_name)) for var_name in var_names]
                syncs_desc = ", ".join(s.identifier for s in syncs)
                schedule_blocks.append(ScheduleBlock(
                    group_or_function=GroupOrFunction.Function,
                    name=Identifier(sync_batch.name),
                    at_or_in=at_or_in,
                    schedule_bin=schedule_bin,
                    description=String(f'Empty function for explicitly SYNCing variables as provided in an ExplicitSyncBatch. Syncs: {syncs_desc}. Generated by EinsteinEngine.'),
                    lang=Language.C,
                    sync=syncs,
                    before=[Identifier(s) for s in sync_batch.schedule_before],
                    after=[Identifier(s) for s in sync_batch.schedule_after]
                ))

        if (new_rad_x_items := self.options.get('new_rad_x_boundary_fns', None)) is not None:
            for batch in new_rad_x_items:
                schedule_bin, at_or_in = self._resolve_schedule_target(batch.schedule_target)
                base_name, rhs_name, var_names, rhs_names = self._get_names_from_new_rad_x_batch(batch)

                reads = OrderedSet(
                    Intent(
                        Identifier(self._get_qualified_group_name_from_var_name(var_name)),
                        IntentRegion.Interior
                    )
                    for var_name in var_names
                )

                writes = OrderedSet(
                    Intent(
                        Identifier(self._get_qualified_group_name_from_var_name(rhs_name)),
                        IntentRegion.Interior
                    )
                    for rhs_name in rhs_names
                )

                schedule_blocks.append(ScheduleBlock(
                    group_or_function=GroupOrFunction.Function,
                    name=Identifier(batch.name),
                    at_or_in=at_or_in,
                    schedule_bin=schedule_bin,
                    description=String(f'Function for applying NewRadX boundary conditions: {base_name} -> {rhs_name}. Generated by EinsteinEngine.'),
                    lang=Language.C,
                    reads=list(reads.union(writes)),
                    writes=list(writes),
                    before=[Identifier(f'{s}_group') for s in batch.schedule_before],
                    after=[Identifier(f'{s}_group') for s in batch.schedule_after]
                ))

        post_post_init_bin, post_post_init_at_in = self._resolve_schedule_target(ScheduleBin.PostPostInit)

        schedule_blocks.append(ScheduleBlock(
            group_or_function=GroupOrFunction.Group,
            name=Identifier(f'Init_Evolve_{self.thorn_def.name}'),
            at_or_in=post_post_init_at_in,
            schedule_bin=post_post_init_bin,
            description=String('Group containing all functions from Evolve that should run before step 0. Generated by EinsteinEngine.'),
            lang=Language.C
        ))

        schedule_blocks.append(ScheduleBlock(
            group_or_function=GroupOrFunction.Group,
            name=Identifier(f'Init_Analysis_{self.thorn_def.name}'),
            at_or_in=post_post_init_at_in,
            schedule_bin=post_post_init_bin,
            description=String('Group containing all functions from Analysis that should run before step 0. Generated by EinsteinEngine.'),
            lang=Language.C,
            after=[Identifier(f'Init_Evolve_{self.thorn_def.name}')]
        ))

        state_vec = self.thorn_def.get_state()
        state_vec_strs = {str(v) for v in state_vec}

        for tf in self.thorn_def.thorn_functions.values():
            if len(bad_vars := {self.thorn_def.var2base.get(str_v := str(v), str_v) for v in tf.eqn_complex.tile_temporaries}.intersection(state_vec_strs)) > 0:
                raise GeneratorException(f'Thorn function {tf.name} uses one or more tile temporaries {bad_vars} that are also in the state vector by way of being declared with an `rhs` property. This is not allowed.')

        if self.options['sync_mode'] is SyncMode.EmulatePresync:
            state_vec_sync = [Identifier(self._get_qualified_group_name_from_var_name(str(v))) for v in state_vec]
            state_vec_sync_desc = ", ".join(s.identifier for s in state_vec_sync)

            schedule_blocks.append(ScheduleBlock(
                group_or_function=GroupOrFunction.Function,
                name=Identifier(f'StateSync_{self.thorn_def.name}'),
                at_or_in=post_post_init_at_in,
                schedule_bin=post_post_init_bin,
                description=String(
                    f'Empty function to SYNC the state vector. Syncs: {state_vec_sync_desc}. Generated by EinsteinEngine.'),
                lang=Language.C,
                before=[Identifier(f'Init_Evolve_{self.thorn_def.name}')],
                sync=state_vec_sync
            ))

            post_step_bin, post_step_at_in = self._resolve_schedule_target(ScheduleBin.PostStep)

            schedule_blocks.append(ScheduleBlock(
                group_or_function=GroupOrFunction.Function,
                name=Identifier(f'StateSync_{self.thorn_def.name}'),
                at_or_in=post_step_at_in,
                schedule_bin=post_step_bin,
                description=String(
                    f'Empty function to SYNC the state vector. Syncs: {state_vec_sync_desc}. Generated by EinsteinEngine.'),
                lang=Language.C,
                sync=state_vec_sync
            ))

        return ScheduleRoot(
            storage_section=StorageSection(storage_lines),
            schedule_section=ScheduleSection(schedule_blocks)
        )

    def _resolve_schedule_target(self, schedule_target: ScheduleTarget) -> tuple[Identifier, AtOrIn]:
        schedule_bin: Identifier
        at_or_in: AtOrIn

        if isinstance(schedule_target, ScheduleBlock):
            schedule_bin = schedule_target.name
            at_or_in = AtOrIn.In
        else:
            assert isinstance(schedule_target, ScheduleBin)
            at_or_in = AtOrIn.At if schedule_target.is_builtin else AtOrIn.In

            if schedule_target is ScheduleBin.Init:
                schedule_bin = Identifier('initial')
            elif schedule_target is ScheduleBin.PostInit:
                schedule_bin = Identifier('postinitial')
            elif schedule_target is ScheduleBin.PostPostInit:
                schedule_bin = Identifier('postpostinitial')
            elif schedule_target is ScheduleBin.InitEvolve:
                schedule_bin = Identifier(f'Init_Evolve_{self.thorn_def.name}')
            elif schedule_target is ScheduleBin.InitAnalysis:
                schedule_bin = Identifier(f'Init_Analysis_{self.thorn_def.name}')
            elif schedule_target is ScheduleBin.Analysis:
                schedule_bin = Identifier('analysis')
            elif schedule_target is ScheduleBin.EstimateError:
                schedule_bin = Identifier('ODESolvers_EstimateError')
            elif schedule_target is ScheduleBin.Evolve:
                schedule_bin = Identifier('ODESolvers_RHS')
            elif schedule_target is ScheduleBin.SpecialEvolve:
                schedule_bin = Identifier('ODESolvers_RHS')
            elif schedule_target is ScheduleBin.DriverInit:
                schedule_bin = Identifier('ODESolvers_Initial')
            elif schedule_target is ScheduleBin.PostStep:
                schedule_bin = Identifier('poststep')
            elif schedule_target is ScheduleBin.PostSubStep:
                schedule_bin = Identifier('ODESolvers_PostStep')
            else:
                raise NotImplementedError(f'Bad ScheduleBin enum member {schedule_target}')

        return schedule_bin, at_or_in

    def generate_interface_ccl(self) -> InterfaceRoot:
        inherits_from = {Identifier(inherited_thorn) for inherited_thorn in self.thorn_def.base2thorn.values()}

        # We always want to inherit from CarpetX even if no vars explicitly need it
        inherits_from.update({Identifier('Driver'), Identifier('ODESolvers')})

        return InterfaceRoot(
            HeaderSection(
                implements=Identifier(self.thorn_def.name),
                inherits=[*inherits_from],
                friends=[]
            ),
            IncludeSection([
                UsesInclude(Verbatim('newradx.hxx'))
            ]),
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
                    lo_i, hi_i = py_param_range
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
                    lo_f, hi_f = py_param_range
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

    def generate_new_rad_x_boundary_function_code(self, batch: NewRadXBoundaryBatch) -> CodeRoot:
        nodes: list[CodeElem] = list()

        # Includes, usings...
        for include in self._boilerplate_includes:
            nodes.append(IncludeDirective(include))

        nodes.append(IncludeDirective(Identifier('newradx.hxx')))

        for include in self._boilerplate_quoted_includes:
            nodes.append(IncludeDirective(include, True))

        nodes.append(Verbatim(self._boilerplate_nv_tools_include))

        for ns in self._boilerplate_namespace_usings:
            nodes.append(UsingNamespace(ns))

        nodes.append(UsingNamespace(Identifier('NewRadX')))

        base_name, rhs_name, var_names, rhs_names = self._get_names_from_new_rad_x_batch(batch)

        sympy_visitor = SympyExprVisitor()
        val_at_infinity = sympy_visitor.visit(batch.val_at_infinity)
        propagation_speed = sympy_visitor.visit(batch.propagation_speed)
        radial_falloff_exponent = sympy_visitor.visit(batch.radial_falloff_exponent)

        apply_calls: list[Stmt] = list()
        for (v, r) in zip(var_names, rhs_names):
            apply_calls.append(
                ExprStmt(
                    FunctionCall(
                        Identifier(f'NewRadX_Apply'),
                        [
                            IdExpr(Identifier('cctkGH')),
                            IdExpr(Identifier(v)),
                            IdExpr(Identifier(r)),
                            val_at_infinity,
                            propagation_speed,
                            radial_falloff_exponent
                        ],
                        []
                    )
                )
            )

        nodes.append(
            ThornFunctionDecl(
                Identifier(batch.name),
                [
                    DeclareCarpetXArgs(Identifier(batch.name)),
                    DeclareCarpetParams(),
                    UsingAlias(Identifier('vreal'), VerbatimExpr(Verbatim('Arith::simd<CCTK_REAL>'))),
                    ConstExprAssignDecl(Identifier('std::size_t'), Identifier('vsize'),
                                        VerbatimExpr(Verbatim('std::tuple_size_v<vreal>'))),
                    Verbatim(self._boilerplate_nv_tools_init(batch.name)),
                    *apply_calls,
                    Verbatim(self._boilerplate_nv_tools_destructor)
                ]
            )
        )

        return CodeRoot(nodes)


    class _NewRadXBatchNames(NamedTuple):
        base_var: str
        rhs_var: str
        var_names: list[str]
        rhs_names: list[str]

    @functools.cache
    def _get_names_from_new_rad_x_batch(self, batch: NewRadXBoundaryBatch) -> _NewRadXBatchNames:
        if isinstance(batch.var, IndexedBase):
            base_name = str(batch.var)

            if base_name not in self.thorn_def.rhs:
                raise GeneratorException(
                    f"NewRadXBoundaryBatch {batch.name} uses base variable {base_name} which has no RHS.")
            rhs_sym = self.thorn_def.rhs[base_name]

            group_name = self.thorn_def.base2group.get(base_name, None)

            rhs_name = str(rhs_sym)
            rhs_group_name = self.thorn_def.base2group.get(rhs_name, None)

            var_names = sorted(self.thorn_def.groups.get(group_name, [base_name]))  # type: ignore[arg-type]
            rhs_names = sorted(self.thorn_def.groups.get(rhs_group_name, [rhs_name]))  # type: ignore[arg-type]

            return CppCarpetXGenerator._NewRadXBatchNames(base_name, rhs_name, var_names, rhs_names)
        else:
            assert isinstance(batch.var, Indexed)

            base_name = str(batch.var.base)

            if base_name not in self.thorn_def.rhs:
                raise GeneratorException(
                    f"NewRadXBoundaryBatch {batch.name} uses base variable {base_name} which has no RHS.")
            rhs_sym = self.thorn_def.rhs[base_name]

            rhs_name = str(rhs_sym)

            rhs_indexed = self.thorn_def.gfs[rhs_name][*batch.var.args[1:]]

            var_names = sorted(str(x) for x in self.thorn_def._flatten_indexed(batch.var))
            rhs_names = sorted(str(x) for x in self.thorn_def._flatten_indexed(rhs_indexed))

            return CppCarpetXGenerator._NewRadXBatchNames(str(batch.var), str(rhs_indexed), var_names, rhs_names)


    def generate_function_code(self, which_fn: str) -> CodeRoot:
        nodes: list[CodeElem] = list()
        thorn_fn: ThornFunction = self.thorn_def.thorn_functions[which_fn]
        fn_name: str = thorn_fn.name

        assert thorn_fn.been_baked

        nodes.append(Verbatim(self._boilerplate_setup))

        # Includes, usings...
        for include in self._boilerplate_includes:
            nodes.append(IncludeDirective(include))

        for include in self._boilerplate_quoted_includes:
            nodes.append(IncludeDirective(include, True))

        nodes.append(Verbatim(self._boilerplate_nv_tools_include))

        # div{x,y,z} macros
        nodes.append(Verbatim(self._boilerplate_div_macros))

        for ns in self._boilerplate_namespace_usings:
            nodes.append(UsingNamespace(ns))

        nodes.append(Using(self._boilerplate_usings))

        # Each variable needs to have a corresponding decl of the form
        # `const GF3D5layout ${VAR_NAME}_layout(${LAYOUT_NAME}_layout);`
        # for the div macros to work; layout here really means centering.

        layout_decls: list[CodeElem] = list()
        declared_layouts: set[Centering] = set()
        var_centerings: dict[str, Centering] = dict()

        used_var_names = [str(v) for v in thorn_fn.eqn_complex.variables]

        # This loop builds the centering decls for each used var
        for var_name in self.var_names:

            # If a var is not used in this function, skip it
            if var_name not in used_var_names:
                continue

            var_centering = self.thorn_def.get_centering_from_var_name(var_name)

            # If this var doesn't have a defined centering, skip it.
            if var_centering is None:
                continue

            assert var_centering is not None

            var_centerings[var_name] = var_centering

            # Make sure the referenced layout has a preceding, corresponding decl of the form
            # `const GF3D5layout ${LAYOUT_NAME}_layout(cctkGH, {$I, $J, $K});`
            if var_centering not in declared_layouts and "'" not in var_name:
                declared_layouts.add(var_centering)

                i, j, k = var_centering.int_repr
                centering_init_list = f'{{{i}, {j}, {k}}}'

                layout_decls.append(ConstConstructDecl(
                    Identifier('GF3D5layout'),
                    Identifier(f'{var_centering.string_repr}_layout'),
                    [IdExpr(Identifier('cctkGH')), VerbatimExpr(Verbatim(centering_init_list))]
                ))

            # Now build the var's centering decl.
            # layout_decls.append(ConstConstructDecl(
            #     Identifier('GF3D5layout'),
            #     Identifier(f'{var_name}_layout'),
            #     [IdExpr(Identifier(f'{var_centering.string_repr}_layout'))]
            # ))

            if "'" not in var_name:
                layout_decls.append(Verbatim(f'#define {var_name}_layout {var_centering.string_repr}_layout'))

        input_var_strs = [str(i) for i in thorn_fn.eqn_complex.inputs]

        # x, y, and z are special, but x is extra special
        xyz_decls = [
            ConstAssignDecl(Identifier('auto'), Identifier(s), IdExpr(Identifier(f'p.{s}'))) for s in ['y', 'z']
            if s in input_var_strs
        ]

        if 'x' in input_var_strs:
            xyz_decls.append(
                ConstAssignDecl(
                    Identifier('vreal'),
                    Identifier('x'),
                    BinOpExpr(
                        IdExpr(Identifier('p.x')),
                        BinOp.Add,
                        BinOpExpr(
                            VerbatimExpr(Verbatim('Arith::iota<vreal>()')),
                            BinOp.Mul,
                            IdExpr(Identifier('p.dx'))
                        )
                    )
                )
            )

        if 't' in input_var_strs:
            xyz_decls.append(
                ConstAssignDecl(
                    Identifier('vreal'),
                    Identifier('t'),
                    IdExpr(Identifier('cctk_time'))
                )
            )

        def calc_stencil_idx(stencil_idx: StencilIdxWithCentering) -> list[Expr]:
            result = 'p.I'

            for i in range(3):
                if stencil_idx.indices[i] == 1:
                    result += f' + p.DI[{i}]'
                elif stencil_idx.indices[i] == -1:
                    result += f' - p.DI[{i}]'
                elif stencil_idx.indices[i] < 0:
                    result += f' - {-stencil_idx.indices[i]}*p.DI[{i}]'
                elif stencil_idx.indices[i] > 0:
                    result += f' + {stencil_idx.indices[i]}*p.DI[{i}]'

            return [
                VerbatimExpr(Verbatim(f'{stencil_idx.centering.string_repr}_layout')),
                VerbatimExpr(Verbatim(result))
            ]

        idxes_with_centerings = OrderedSet(
            StencilIdxWithCentering(
                stencil_idx.indices,
                require(
                    self.thorn_def.get_centering_from_var_name(stencil_idx.var_name),
                    lambda: f'Stencil index {stencil_idx} refers to variable {stencil_idx.var_name} with an unknown centering.'
                )
            )
            for stencil_idx in sorted(thorn_fn.eqn_complex.stencil_idxes)
        )

        stencil_idx_decls = [
            ConstConstructDecl(
                Identifier('GF3D5index'),
                Identifier(encode_stencil_idx(stencil_idx)),
                calc_stencil_idx(stencil_idx)
            )
            for stencil_idx in idxes_with_centerings
        ]

        # DXI, DYI, DZI decls
        di_decls = [
            ConstAssignDecl(Identifier('auto'), Identifier(s), BinOpExpr(FloatLiteralExpr(1.0), BinOp.Div, VerbatimExpr(Verbatim(f'CCTK_DELTA_SPACE({n})'))))
            for n, s in enumerate(['DXI', 'DYI', 'DZI'])
        ]

        stencil_limits = thorn_fn.eqn_complex.stencil_limits
        stencil_limit_checks = [
            ExprStmt(
                FunctionCall(
                    Identifier('CCTK_ASSERT'),
                    [
                        BinOpExpr(
                            VerbatimExpr(Verbatim(f'cctk_nghostzones[{i}]')),
                            BinOp.Gte,
                            IntLiteralExpr(stencil_limits[i])
                        )
                    ],
                    []
                )
            )
        for i in range(3) if stencil_limits[i] != 0]

        one_and_zero = [
           ConstAssignDecl(Identifier("vreal"), Identifier("v_one"), IntLiteralExpr(1)),
           ConstAssignDecl(Identifier("vreal"), Identifier("v_zero"), IntLiteralExpr(0))
        ]

        loop_to_output_centering = [
            self._get_output_centering_for_loop(thorn_fn, loop_idx, eqn_list.outputs, var_centerings)
            for loop_idx, eqn_list in enumerate(thorn_fn.eqn_complex.eqn_lists)
        ]

        loop_to_output_region = [
            self._get_output_region_for_loop(thorn_fn, loop_idx)
            for loop_idx, _ in enumerate(thorn_fn.eqn_complex.eqn_lists)
        ]

        tile_temp_centerings = self._get_tile_temp_centerings(loop_to_output_centering, thorn_fn)
        tile_temps_by_centering, tile_temps_to_centering, final_loop_centerings = tile_temp_centerings.temps_by_centering, tile_temp_centerings.temps_to_centering, tile_temp_centerings.final_loop_centerings

        tile_temp_setup = self._generate_tile_temp_setup(tile_temps_by_centering)
        
        sympy_visitor = self._mk_sympy_visitor(
            tile_temps_to_centering.keys(),
            centering_fn=lambda vn: self.thorn_def.get_centering_from_var_name(vn)
        )

        carpetx_loops: list[Stmt] = list()
        for loop_idx, eqn_list in enumerate(thorn_fn.eqn_complex.eqn_lists):
            output_centering = final_loop_centerings[loop_idx]
            output_region = loop_to_output_region[loop_idx]

            subst_result = substitute_recycled_temporaries(eqn_list)

            def _resolve_overwrite(s: sy.Symbol) -> sy.Symbol:
                return s if "'" not in str(s) else sy.Symbol(str(s).replace("'", ""))  # type: ignore[no-untyped-call]

            eqns: list[tuple[sy.Symbol, Expr]] = [(_resolve_overwrite(lhs), sympy_visitor.visit(rhs)) for lhs, rhs in subst_result.eqns]
            temporaries = [
                str(lhs) for lhs in OrderedSet(eqn_list.eqns.keys())
                if lhs in (eqn_list.temporaries - self.thorn_def.global_temporaries - eqn_list.tile_temporaries) and str(lhs) not in self.var_names
            ]

            carpetx_loops.append(
                CarpetXGridLoopCall(
                    output_centering,
                    output_region,
                    CarpetXGridLoopLambda(
                        preceding=xyz_decls+stencil_idx_decls,  # type: ignore[operator]
                        equations=eqns,
                        succeeding=[],
                        temporaries=temporaries,
                        reassigned_lhses=subst_result.substituted_lhs_idxes
                    )
                )
            )
        
        if thorn_fn.schedule_target is ScheduleBin.SpecialEvolve:
            carpetx_loops = [
                IfElseStmt(
                    BinOpExpr(
                        IdExpr(Identifier('*substep_counter')),
                        BinOp.Lte,
                        IntLiteralExpr(1)
                    ),
                    carpetx_loops,
                    []
                )
            ]

        # Build the function decl and its body.
        nodes.append(
            ThornFunctionDecl(
                Identifier(fn_name),
                [
                    DeclareCarpetXArgs(Identifier(fn_name)),
                    DeclareCarpetParams(),
                    UsingAlias(Identifier('vreal'), VerbatimExpr(Verbatim('Arith::simd<CCTK_REAL>'))),
                    ConstExprAssignDecl(Identifier('std::size_t'), Identifier('vsize'), VerbatimExpr(Verbatim('std::tuple_size_v<vreal>'))),
                    Verbatim(self._boilerplate_nv_tools_init(fn_name)),
                    *layout_decls,
                    *di_decls,
                    *stencil_limit_checks,
                    *one_and_zero,
                    *tile_temp_setup,
                    *carpetx_loops,
                    Verbatim(self._boilerplate_nv_tools_destructor)
                ]
            )
        )

        return CodeRoot(nodes)

    @staticmethod
    def _get_tile_temp_centerings(loop_to_output_centering: list[Optional[Centering]], thorn_fn: ThornFunction) -> _TileTempCenteringData:
        tile_temps_by_centering: dict[Centering, set[sy.Symbol]] = OrderedDict()
        tile_temps_to_centering: dict[sy.Symbol, Centering] = OrderedDict()

        final_loop_centerings = list()

        td = thorn_fn.thorn_def

        thorn_fn.eqn_complex._calc_tile_temps()

        for loop_idx, eqn_list in enumerate(thorn_fn.eqn_complex.eqn_lists):
            loop_centering = loop_to_output_centering[loop_idx]

            for tile_temp in eqn_list.uninitialized_tile_temporaries:
                centering = loop_centering or td.centering.get(td.var2base.get(str(tile_temp), str(tile_temp)), None)

                if not centering:
                    raise GeneratorException(
                        f"In {thorn_fn.name}@{loop_idx}: Tile temporary {tile_temp} does not have a declared centering, and no centering is specified for the loop, so its centering cannot be inferred."
                    )

                tile_temps_by_centering.setdefault(centering, set()).add(tile_temp)
                tile_temps_to_centering[tile_temp] = centering

            if loop_centering is None:
                inferred_centerings = set(tile_temps_by_centering.keys())

                if len(inferred_centerings) == 0:
                    raise GeneratorException(
                        f"In {thorn_fn.name}@{loop_idx}: Cannot infer output centering because there are no output vars or tile temps."
                    )
                elif len(inferred_centerings) > 1:
                    raise GeneratorException(
                        f"In {thorn_fn.name}@{loop_idx}: Cannot infer output centering because there are no output vars, and multiple possible centerings were inferred from the tile temps: {inferred_centerings}."
                    )
                else:
                    loop_centering = inferred_centerings.pop()

            final_loop_centerings.append(loop_centering)

            for tile_temp in eqn_list.preinitialized_tile_temporaries:
                assert tile_temp in tile_temps_to_centering
                if tile_temps_to_centering[tile_temp] != loop_centering:
                    raise GeneratorException(
                        f"In {thorn_fn}: All loops accessing tile temporary '{tile_temp}' must have the same centering."
                        f"  Declared with centering: {tile_temps_to_centering[tile_temp]}"
                        f"  Read in loop {loop_idx} with centering: {loop_centering.string_repr}"
                    )

        return _TileTempCenteringData(
            temps_by_centering=tile_temps_by_centering,
            temps_to_centering=tile_temps_to_centering,
            final_loop_centerings=final_loop_centerings
        )

    def _mk_sympy_visitor(self, tile_temps: Collection[sy.Symbol], centering_fn: VarCenteringFn) -> SympyExprVisitor:
        stencil_fn_names = {str(fn) for fn, fn_is_stencil in self.thorn_def.is_stencil.items() if fn_is_stencil}
        tile_temp_names = {str(sym) for sym in tile_temps}

        def should_wrap_with_access_fn(name: str, in_stencil_args: bool) -> bool:
            return not in_stencil_args and (name in self.var_names or name in tile_temp_names)

        sympy_visitor = SympyExprVisitor(
            stencil_fns=stencil_fn_names,
            should_wrap_with_access_fn=should_wrap_with_access_fn,
            centering_fn=centering_fn
        )
        return sympy_visitor

    @staticmethod
    def _generate_tile_temp_setup(tile_temps_by_centering: dict[Centering, set[sy.Symbol]]) -> list[CodeElem]:
        tile_temp_setup: list[CodeElem] = list()
        
        for centering, temps in tile_temps_by_centering.items():
            tile_temp_setup.append(
                ConstAssignDecl(
                    Identifier('int'),
                    Identifier(f'nTileTemps_{centering.string_repr}'),
                    IntLiteralExpr(len(temps))
                )
            )

            tile_temp_setup.append(
                ConstConstructDecl(
                    Identifier('GF3D5vector<CCTK_REAL>'),
                    Identifier(f'tileTemps_{centering.string_repr}'),
                    [
                        IdExpr(Identifier(f'{centering.string_repr}_layout')),
                        IdExpr(Identifier(f'nTileTemps_{centering.string_repr}'))
                    ]
                )
            )

            tile_temp_setup.append(
                MutableAssignDecl(
                    Identifier('int'),
                    Identifier(f'idxTileTemp_{centering.string_repr}'),
                    IntLiteralExpr(0)
                )
            )

            tile_temp_setup.append(
                ConstAssignDecl(
                    Identifier('auto'),
                    Identifier(f'mkTileTemp_{centering.string_repr}'),
                    VerbatimExpr(Verbatim(
                        f'[&]() {{ return GF3D5<CCTK_REAL>(tileTemps_{centering.string_repr}(idxTileTemp_{centering.string_repr}++)); }}'
                    ))
                )
            )

            for temp in sorted(temps, key=lambda sym: str(sym)):
                tile_temp_setup.append(
                    ConstConstructDecl(
                        Identifier('GF3D5<CCTK_REAL>'),
                        Identifier(str(temp)),
                        [
                            FunctionCall(
                                Identifier(f'mkTileTemp_{centering.string_repr}'),
                                [],
                                []
                            )
                        ]
                    )
                )

                tile_temp_setup.append(
                    Verbatim(f'#define {str(temp)}_layout {centering.string_repr}_layout')
                )

        return tile_temp_setup

    def _get_output_region_for_loop(self,
                                    thorn_fn: ThornFunction,
                                    loop_idx: int) -> IntentRegion:

        """
        Figure out what kind of loop we need (all, int, bnd) based on the write region of the loop's outputs, or, failing that, the inputs.
        All of this loop's outputs need to have the same write region.
        """

        eqn_list = thorn_fn.eqn_complex.eqn_lists[loop_idx]
        write_decls = eqn_list.write_decls
        read_decls = eqn_list.read_decls

        writes = {
            var: spec
            for var, spec in ((str(var).replace("'", ""), spec) for var, spec in write_decls.items())
            if var in self.var_names
        }

        reads = {
            var: spec
            for var, spec in ((str(var).replace("'", ""), spec) for var, spec in read_decls.items())
            if var in self.var_names
        }

        if len(writes) == 0 and len(reads) == 0:
            return IntentRegion.Everywhere  # No inputs and outputs; assume analytical

        if len(writes) == 0:
            input_regions = set(reads.values())

            if None in input_regions or len(input_regions) == 0:
                raise GeneratorException(f"In {thorn_fn.name}@{loop_idx}: All input vars must have a read region. There are no output vars.")

            for rhs in eqn_list.eqns.values():
                for sten in rhs.find(stencil):  # type: ignore[no-untyped-call]
                    if sten.args[1] != 0 or sten.args[2] != 0 or sten.args[3] != 0:
                        return IntentRegion.Interior

            if len(input_regions) > 1:
                if len(input_regions) == 2 and IntentRegion.Everywhere in input_regions and IntentRegion.Interior in input_regions:
                    wprint(f"In {thorn_fn.name}@{loop_idx}:"
                           f" While trying to infer the loop region, we found that there were no output vars,"
                           f" and we found the input vars to have a mix of Interior and Everywhere read regions."
                           f" It looks like you are trying to write to a tile temp based on a stencil function, e.g.,"
                           f" finite difference, so we will infer Interior as the loop region.")
                    return IntentRegion.Interior

                raise GeneratorException(
                    f"In {thorn_fn.name}@{loop_idx}: Input vars have mixed read regions: {list(write_decls.items())}\nSince there are no output vars, the loop region cannot be inferred."
                )

            [input_region] = input_regions
            return input_region
        else:
            output_regions = set(writes.values())

            if None in output_regions or len(output_regions) == 0:
                raise GeneratorException(f"In {thorn_fn.name}@{loop_idx}: All output vars for must have a write region.")

            if len(output_regions) > 1:
                raise GeneratorException(
                    f"In {thorn_fn.name}@{loop_idx}: Output vars have mixed write regions: {list(write_decls.items())}"
                )

            [output_region] = output_regions
            return output_region

    def _get_output_centering_for_loop(self,
                                       thorn_fn: ThornFunction,
                                       loop_idx: int,
                                       output_vars: set[sy.Symbol],
                                       var_centerings: dict[str, Centering]) -> Optional[Centering]:
        """
        Figure out which centering to pass to grid.loop_int_device<...>
        All of this loop's outputs need to have the same centering.
        """

        output_centerings = {
            var_centerings[var_name] for var_name in (str(var).replace("'", "") for var in output_vars)
            if var_name in self.var_names
        }

        if None in output_centerings or len(output_centerings) == 0:
            return None  # This may be a case where a loop has no outputs but writes to a tile temp, so just return None for now.

        if len(output_centerings) > 1:
            raise GeneratorException(
                f"Output vars for '{thorn_fn.name}@{loop_idx}' have mixed centerings: {output_centerings}")

        [output_centering] = output_centerings
        return output_centering


    def generate_sync_batch_function_code(self, sync_batch: _HasName | str) -> CodeRoot:
        return CodeRoot([
            Verbatim(self._boilerplate_setup),
            *[IncludeDirective(include) for include in self._boilerplate_includes],
            ThornFunctionDecl(
                Identifier(sync_batch if isinstance(sync_batch, str) else sync_batch.name),
                []
            )
        ])
