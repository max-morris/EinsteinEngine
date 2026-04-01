import os
import sys
from abc import ABC
from typing import TypeVar, Generic, Optional

from EinsteinEngine import ExplicitSyncBatch
from EinsteinEngine.util import OrderedSet
from EinsteinEngine.dsl.use_indices import ThornDef
from EinsteinEngine.emit.ccl.interface.interface_visitor import InterfaceVisitor
from EinsteinEngine.emit.ccl.param.param_visitor import ParamVisitor
from EinsteinEngine.emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from EinsteinEngine.emit.code.code_tree import CodeNode
from EinsteinEngine.emit.code.cpp.cpp_visitor import CppVisitor
from nrpy.helpers.conditional_file_updater import ConditionalFileUpdater

from EinsteinEngine.emit.visitor import Visitor
from EinsteinEngine.generators.cactus_generator import CactusGenerator
from EinsteinEngine.generators.cpp_carpetx_generator import CppCarpetXGenerator

G = TypeVar('G', bound=CactusGenerator)
CV = TypeVar('CV', bound=Visitor[CodeNode])


class ThornWizard(ABC, Generic[G, CV]):
    thorn_def: ThornDef
    generator: G
    code_visitor: CV
    base_dir: str

    def __init__(self, thorn_def: ThornDef, generator: G, code_visitor: CV) -> None:
        self.thorn_def = thorn_def
        self.generator = generator
        self.code_visitor = code_visitor
        self.base_dir = os.path.join(self.thorn_def.arrangement, self.thorn_def.name)

    def generate_thorn(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "src"), exist_ok=True)

        for fn_name in OrderedSet(self.thorn_def.thorn_functions.keys()):
            print('=====================')
            code_tree = self.generator.generate_function_code(fn_name)
            code = self.code_visitor.visit(code_tree)
            code_fname = os.path.join(self.base_dir, "src", self.generator.get_fn_src_file_name(fn_name))
            with ConditionalFileUpdater(code_fname) as fd:
                fd.write(code)

        print('== param.ccl ==')
        param_tree = self.generator.generate_param_ccl()
        param_ccl = ParamVisitor().visit(param_tree)
        if param_ccl == "":
            param_ccl = "# Empty"  # TODO: Hack for bug in ConditionalFileUpdater
        param_ccl_fname = os.path.join(self.base_dir, "param.ccl")
        with ConditionalFileUpdater(param_ccl_fname) as fd:
            fd.write(param_ccl)

        print('== interface.ccl ==')
        interface_tree = self.generator.generate_interface_ccl()
        interface_ccl = InterfaceVisitor().visit(interface_tree)
        interface_ccl_fname = os.path.join(self.base_dir, "interface.ccl")
        with ConditionalFileUpdater(interface_ccl_fname) as fd:
            fd.write(interface_ccl)

        print('== schedule.ccl ==')
        schedule_tree = self.generator.generate_schedule_ccl()
        schedule_ccl = ScheduleVisitor().visit(schedule_tree)
        schedule_ccl_fname = os.path.join(self.base_dir, "schedule.ccl")
        with ConditionalFileUpdater(schedule_ccl_fname) as fd:
            fd.write(schedule_ccl)

        print('== configuration.ccl ==')
        configuration_ccl = f"""
REQUIRES Arith Loop {self.thorn_def.name}_gen AMReX NewRadX

PROVIDES {self.thorn_def.name}_gen
{{
#   SCRIPT bin/generate.py
#   LANG python3
}}
""".strip()
        configuration_ccl_fname = os.path.join(self.base_dir, "configuration.ccl")
        with ConditionalFileUpdater(configuration_ccl_fname) as fd:
            fd.write(configuration_ccl)

        print('== make.code.defn ==')
        makefile = self.generator.generate_makefile()
        makefile_fname = os.path.join(self.base_dir, "src/make.code.defn")
        with ConditionalFileUpdater(makefile_fname) as fd:
            fd.write(makefile)

        gitignore_filename = os.path.join(self.base_dir, ".gitignore")
        if not os.path.exists(gitignore_filename):
            with open(gitignore_filename, "w") as fd:
                fd.write("*")


class CppCarpetXWizard(ThornWizard[CppCarpetXGenerator, CppVisitor]):
    def __init__(self, thorn_def: ThornDef, generator: Optional[CppCarpetXGenerator] = None):
        if generator is None:
            generator = CppCarpetXGenerator(thorn_def)
        super().__init__(thorn_def, generator, CppVisitor(generator))

    def generate_thorn(self) -> None:
        super().generate_thorn()

        sync_batch: ExplicitSyncBatch | str
        for sync_batch in OrderedSet(self.generator.options.get('explicit_syncs', list()) + [f'StateSync_{self.thorn_def.name}']):  # type: ignore[operator]
            code_tree = self.generator.generate_sync_batch_function_code(sync_batch)
            code = self.code_visitor.visit(code_tree)
            code_fname = os.path.join(self.base_dir, "src", self.generator.get_explicit_src_file_name(sync_batch))
            with ConditionalFileUpdater(code_fname) as fd:
                fd.write(code)

        for rad_batch in OrderedSet(self.generator.options.get('new_rad_x_boundary_fns', list())):
            code_tree = self.generator.generate_new_rad_x_boundary_function_code(rad_batch)
            code = self.code_visitor.visit(code_tree)
            code_fname = os.path.join(self.base_dir, "src", self.generator.get_explicit_src_file_name(rad_batch))
            with ConditionalFileUpdater(code_fname) as fd:
                fd.write(code)
