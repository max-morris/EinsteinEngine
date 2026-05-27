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

import os
from typing import Optional

from nrpy.helpers.conditional_file_updater import ConditionalFileUpdater

from EinsteinEngine.emit.code.f90.f90_visitor import F90Visitor
from EinsteinEngine.frontend.dsl.f90.vanilla_f90_frontend import VanillaF90Module
from EinsteinEngine.generators.vanilla_f90_generator import VanillaF90Generator
from EinsteinEngine.wizards.wizard import Wizard


class VanillaF90Wizard(Wizard[VanillaF90Module, VanillaF90Generator, F90Visitor]):
    base_dir: str
    license_file: Optional[str]

    def __init__(self, frontend: VanillaF90Module,
                 *,
                 license_header: Optional[str],
                 license_file: Optional[str]) -> None:
        super().__init__(frontend, VanillaF90Generator(frontend), F90Visitor(), license_header=license_header)
        self.license_file = license_file
        self.base_dir = frontend.name

    def generate_module(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "src"), exist_ok=True)

        if self.license_file:
            with ConditionalFileUpdater(os.path.join(self.base_dir, "LICENSE")) as fd:
                fd.write(self.license_file)

        gitignore_filename = os.path.join(self.base_dir, ".gitignore")
        if not os.path.exists(gitignore_filename):
            with open(gitignore_filename, "w") as fd:
                fd.write("*")

        mod_tree = self.generator.generate_module_code()
        mod = self.code_visitor.visit(mod_tree)
        mod_filename = os.path.join(self.base_dir, 'src', f'{self.frontend.name}.f90')
        with ConditionalFileUpdater(mod_filename) as fd:
            self.write_with_header(fd, mod, '!')

        for fn_name in sorted(self.frontend.functions.keys()):
            code_tree = self.generator.generate_submodule_code(fn_name)
            code = self.code_visitor.visit(code_tree)
            code_filename = os.path.join(self.base_dir, "src", f'{self.frontend.name}_{fn_name}.f90')
            with ConditionalFileUpdater(code_filename) as fd:
                self.write_with_header(fd, code, '!')
