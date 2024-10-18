import os
import sys
from abc import ABC
from typing import TypeVar, Generic, Optional

from EmitCactus.util import OrderedSet
from EmitCactus.dsl.use_indices import ThornDef
from EmitCactus.emit.ccl.interface.interface_visitor import InterfaceVisitor
from EmitCactus.emit.ccl.param.param_visitor import ParamVisitor
from EmitCactus.emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from EmitCactus.emit.code.code_tree import CodeNode
from EmitCactus.emit.code.cpp.cpp_visitor import CppVisitor
from nrpy.helpers.conditional_file_updater import ConditionalFileUpdater

from EmitCactus.emit.visitor import Visitor
from EmitCactus.generators.cactus_generator import CactusGenerator
from EmitCactus.generators.cpp_carpetx_generator import CppCarpetXGenerator

G = TypeVar('G', bound=CactusGenerator)
CV = TypeVar('CV', bound=Visitor[CodeNode])


class ThornWizard(ABC, Generic[G, CV]):
    thorn_def: ThornDef
    generator: G
    code_visitor: CV

    def __init__(self, thorn_def: ThornDef, generator: G, code_visitor: CV) -> None:
        self.thorn_def = thorn_def
        self.generator = generator
        self.code_visitor = code_visitor

    def generate_thorn(self, schedule_txt:str="") -> None:
        base_dir = os.path.join(self.thorn_def.arrangement, self.thorn_def.name)
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "src"), exist_ok=True)

        for fn_name in OrderedSet(self.thorn_def.thorn_functions.keys()):
            print('=====================')
            code_tree = self.generator.generate_function_code(fn_name)
            code = self.code_visitor.visit(code_tree)
            # print(code)
            code_fname = os.path.join(base_dir, "src", self.generator.get_src_file_name(fn_name))
            with ConditionalFileUpdater(code_fname) as fd:
                fd.write(code)

        print('== param.ccl ==')
        param_tree = self.generator.generate_param_ccl()
        param_ccl = ParamVisitor().visit(param_tree)
        if param_ccl == "":
            param_ccl = " " # Hack for bug in ConditionalFileUpdater
        param_ccl_fname = os.path.join(base_dir, "param.ccl")
        with ConditionalFileUpdater(param_ccl_fname) as fd:
            fd.write(param_ccl)

        print('== interface.ccl ==')
        interface_tree = self.generator.generate_interface_ccl()
        interface_ccl = InterfaceVisitor().visit(interface_tree)
        # print(interface_ccl)
        interface_ccl_fname = os.path.join(base_dir, "interface.ccl")
        with ConditionalFileUpdater(interface_ccl_fname) as fd:
            fd.write(interface_ccl)

        print('== schedule.ccl ==')
        schedule_tree = self.generator.generate_schedule_ccl()
        schedule_ccl = ScheduleVisitor().visit(schedule_tree)
        # print(schedule_ccl)
        schedule_ccl_fname = os.path.join(base_dir, "schedule.ccl")
        with ConditionalFileUpdater(schedule_ccl_fname) as fd:
            fd.write(schedule_ccl)
            fd.write(schedule_txt)

        print('== configuration.ccl ==')
        configuration_ccl = f"""
REQUIRES Arith Loop {self.thorn_def.name}_gen AMReX

PROVIDES {self.thorn_def.name}_gen
{{
#   SCRIPT bin/generate.py
#   LANG python3
}}
""".strip()
        # print(configuration_ccl)
        configuration_ccl_fname = os.path.join(base_dir, "configuration.ccl")
        with ConditionalFileUpdater(configuration_ccl_fname) as fd:
            fd.write(configuration_ccl)

        print('== make.code.defn ==')
        makefile = self.generator.generate_makefile()
        # print(makefile)
        makefile_fname = os.path.join(base_dir, "src/make.code.defn")
        with ConditionalFileUpdater(makefile_fname) as fd:
            fd.write(makefile)

        PYTHON = os.path.abspath(sys.executable)

        # We want the currently executing script
        with open(sys.argv[0], "r") as fd:
            generate_py = fd.read()

        generate_py_fname = os.path.join(base_dir, "bin/generate.py")
        os.makedirs(os.path.dirname(generate_py_fname), exist_ok=True)
        if not os.path.exists(generate_py_fname):
            with open(generate_py_fname, "w") as fd:
                fd.write(generate_py)


class CppCarpetXWizard(ThornWizard[CppCarpetXGenerator, CppVisitor]):
    def __init__(self, thorn_def: ThornDef, generator: Optional[CppCarpetXGenerator] = None):
        if generator is None:
            generator = CppCarpetXGenerator(thorn_def)
        super().__init__(thorn_def, generator, CppVisitor(generator))
