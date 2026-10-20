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

from EinsteinEngine.emit.code.amrex.amrex_visitor import AMReXVisitor
from EinsteinEngine.frontend.dsl.amrex.amrex_frontend import AMReXKernelClass
from EinsteinEngine.generators.amrex_generator import AMReXGenerator
from EinsteinEngine.wizards.wizard import Wizard


class AMReXWizard(Wizard[AMReXKernelClass, AMReXGenerator, AMReXVisitor]):
    """Writes an AMReX kernel class to <out_dir>/<class name>.hpp."""
    out_dir: str

    def __init__(self, frontend: AMReXKernelClass, *, out_dir: str = '.',
                 license_header: Optional[str] = None) -> None:
        super().__init__(frontend, AMReXGenerator(frontend), AMReXVisitor(), license_header=license_header)
        self.out_dir = out_dir

    def generate_header(self, banner: str = '') -> str:
        """Writes the header and returns its path."""
        os.makedirs(self.out_dir, exist_ok=True)
        code = self.code_visitor.visit(self.generator.generate_kernel_class(banner))
        path = os.path.join(self.out_dir, f'{self.frontend.name}.hpp')
        with ConditionalFileUpdater(path) as fd:
            self.write_with_header(fd, code, '//')
        return path
