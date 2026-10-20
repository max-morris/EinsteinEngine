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

from EinsteinEngine.emit.code.amrex.amrex_tree import FieldLoad
from EinsteinEngine.emit.code.amrex.amrex_visitor import AMReXVisitor
from EinsteinEngine.frontend.dsl.amrex.amrex_frontend import AMReXModule
from EinsteinEngine.generators.amrex import AMReXGenerator
from EinsteinEngine.generators.generator_exception import GeneratorException
from EinsteinEngine.wizards.wizard import Wizard


class AMReXWizard(Wizard[AMReXModule, AMReXGenerator, AMReXVisitor]):
    """
    Writes an AMReX module to <out_dir>/<name><extension>. The banner is the caller's fixed text (recipe, arguments,
    purpose); it never contains the EE commit or a date, so the header is deterministic (docs, section 5.6).
    """
    out_dir: str
    extension: str

    def __init__(self, frontend: AMReXModule, *, out_dir: str = '.', extension: str = '.hpp',
                 visitor: Optional[AMReXVisitor] = None, license_header: Optional[str] = None) -> None:
        super().__init__(frontend, AMReXGenerator(frontend), visitor or AMReXVisitor(), license_header=license_header)
        self.out_dir = out_dir
        self.extension = extension
        arrays = {a.name: a for a in frontend.arrays}

        def field_load(n: FieldLoad) -> str:
            if (binding := frontend.field_binding(n.var)) is None:
                raise GeneratorException(f'{n.var} is not a field')
            array = arrays[binding.array]
            return array.access.load(array.name, n.var, binding.component, n.offset)

        self.code_visitor.field_load = field_load

    def generate_header(self, banner: str = '') -> str:
        """Writes the header and returns its path."""
        os.makedirs(self.out_dir, exist_ok=True)
        code = self.code_visitor.visit(self.generator.generate(banner))
        path = os.path.join(self.out_dir, f'{self.frontend.name}{self.extension}')
        with ConditionalFileUpdater(path) as fd:
            self.write_with_header(fd, code, '//')
        return path
