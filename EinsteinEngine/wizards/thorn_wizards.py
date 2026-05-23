#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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

import os
from typing import TypeVar, Generic, Optional

from nrpy.helpers.conditional_file_updater import ConditionalFileUpdater

from EinsteinEngine.emit.ccl.interface.interface_visitor import InterfaceVisitor
from EinsteinEngine.emit.ccl.param.param_visitor import ParamVisitor
from EinsteinEngine.emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from EinsteinEngine.emit.code.common.code_tree import CodeNode
from EinsteinEngine.emit.code.cpp_carpetx.cpp_carpetx_visitor import CppVisitor
from EinsteinEngine.emit.visitor import Visitor
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ThornDef
from EinsteinEngine.frontend.dsl.cactus.carpetx import ExplicitSyncBatch
from EinsteinEngine.generators.cactus_generator import CactusGenerator
from EinsteinEngine.generators.cpp_carpetx_generator import CppCarpetXGenerator
from EinsteinEngine.common.util import OrderedSet
from EinsteinEngine.wizards.wizard import Wizard

G = TypeVar('G', bound=CactusGenerator)
CV = TypeVar('CV', bound=Visitor[CodeNode])


class ThornWizard(Generic[G, CV], Wizard[ThornDef, G, CV]):
    base_dir: str
    license_file: Optional[str]

    @property
    def thorn_def(self) -> ThornDef:
        return self.frontend

    @thorn_def.setter
    def thorn_def(self, thorn_def: ThornDef) -> None:
        self.frontend = thorn_def

    def __init__(self, thorn_def: ThornDef, generator: G, code_visitor: CV, *,
                 license_header: Optional[str],
                 license_file: Optional[str]) -> None:
        super().__init__(thorn_def, generator, code_visitor, license_header=license_header)
        self.license_file = license_file
        self.base_dir = os.path.join(self.thorn_def.arrangement, self.thorn_def.name)

    def generate_thorn(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "src"), exist_ok=True)

        for fn_name in OrderedSet(self.thorn_def.functions.keys()):
            print('=====================')
            code_tree = self.generator.generate_function_code(fn_name)
            code = self.code_visitor.visit(code_tree)
            code_fname = os.path.join(self.base_dir, "src", self.generator.get_fn_src_file_name(fn_name))
            with ConditionalFileUpdater(code_fname) as fd:
                self.write_with_header(fd, code, '//')

        print('== param.ccl ==')
        param_tree = self.generator.generate_param_ccl()
        param_ccl = ParamVisitor().visit(param_tree)
        if param_ccl == "":
            param_ccl = "# Empty"  # TODO: Hack for bug in ConditionalFileUpdater
        param_ccl_fname = os.path.join(self.base_dir, "param.ccl")
        with ConditionalFileUpdater(param_ccl_fname) as fd:
            self.write_with_header(fd, param_ccl, '#')

        print('== interface.ccl ==')
        interface_tree = self.generator.generate_interface_ccl()
        interface_ccl = InterfaceVisitor().visit(interface_tree)
        interface_ccl_fname = os.path.join(self.base_dir, "interface.ccl")
        with ConditionalFileUpdater(interface_ccl_fname) as fd:
            self.write_with_header(fd, interface_ccl, '#')

        print('== schedule.ccl ==')
        schedule_tree = self.generator.generate_schedule_ccl()
        schedule_ccl = ScheduleVisitor().visit(schedule_tree)
        schedule_ccl_fname = os.path.join(self.base_dir, "schedule.ccl")
        with ConditionalFileUpdater(schedule_ccl_fname) as fd:
            self.write_with_header(fd, schedule_ccl, '#')

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
            self.write_with_header(fd, configuration_ccl, '#')

        print('== make.code.defn ==')
        makefile = self.generator.generate_makefile()
        makefile_fname = os.path.join(self.base_dir, "src/make.code.defn")
        with ConditionalFileUpdater(makefile_fname) as fd:
            self.write_with_header(fd, makefile, '#')

        gitignore_filename = os.path.join(self.base_dir, ".gitignore")
        if not os.path.exists(gitignore_filename):
            with open(gitignore_filename, "w") as fd:
                fd.write("*")

        if self.license_file is not None:
            license_filename = os.path.join(self.base_dir, "LICENSE")
            with open(license_filename, "w") as fd:
                fd.write(self.license_file)


class CppCarpetXWizard(ThornWizard[CppCarpetXGenerator, CppVisitor]):
    def __init__(self, thorn_def: ThornDef, generator: Optional[CppCarpetXGenerator] = None, *,
                 license_header: Optional[str] = None,
                 license_file: Optional[str] = None):
        if generator is None:
            generator = CppCarpetXGenerator(thorn_def)
        super().__init__(
            thorn_def, generator, CppVisitor(generator), license_header=license_header, license_file=license_file
        )

    def generate_thorn(self) -> None:
        super().generate_thorn()

        sync_batch: ExplicitSyncBatch | str
        for sync_batch in OrderedSet(self.generator.options.get('explicit_syncs', list()) + [f'StateSync_{self.thorn_def.name}']):  # type: ignore[operator]
            code_tree = self.generator.generate_sync_batch_function_code(sync_batch)
            code = self.code_visitor.visit(code_tree)
            code_fname = os.path.join(self.base_dir, "src", self.generator.get_explicit_src_file_name(sync_batch))
            with ConditionalFileUpdater(code_fname) as fd:
                self.write_with_header(fd, code, '//')

        for rad_batch in OrderedSet(self.generator.options.get('new_rad_x_boundary_fns', list())):
            code_tree = self.generator.generate_new_rad_x_boundary_function_code(rad_batch)
            code = self.code_visitor.visit(code_tree)
            code_fname = os.path.join(self.base_dir, "src", self.generator.get_explicit_src_file_name(rad_batch))
            with ConditionalFileUpdater(code_fname) as fd:
                self.write_with_header(fd, code, '//')
