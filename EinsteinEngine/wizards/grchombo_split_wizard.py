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

"""PROBE (branch probe/grchombo-split; never merged): writes a hard-split GRChombo compute class
(generators/grchombo_split_generator.py) to <out_dir>/<class name>.hpp."""
import os
from typing import Any, Optional

from nrpy.helpers.conditional_file_updater import ConditionalFileUpdater

from EinsteinEngine.emit.code.grchombo.grchombo_visitor import GRChomboVisitor
from EinsteinEngine.frontend.dsl.grchombo.grchombo_frontend import GRChomboComputeClass
from EinsteinEngine.generators.grchombo_split_generator import GRChomboSplitGenerator, SplitComputeClassDecl, \
    TileLoad, TileStore, render_split
from EinsteinEngine.wizards.wizard import Wizard


class GRChomboSplitVisitor(GRChomboVisitor):
    """GRChomboVisitor plus the split class and its scratch loads and stores."""

    def visit(self, n: Any) -> str:
        if isinstance(n, SplitComputeClassDecl):
            return render_split(self.visit, n)
        if isinstance(n, TileLoad):
            return f'const data_t {n.name} = data_t(SIMDIFY<data_t>(m_tile[{n.index}])[oidx]);'
        if isinstance(n, TileStore):
            return f'SIMDIFY<data_t>(m_tile[{n.index}])[oidx] = {n.name};'
        return str(super().visit(n))


class GRChomboSplitWizard(Wizard[GRChomboComputeClass, GRChomboSplitGenerator, GRChomboVisitor]):
    out_dir: str

    def __init__(self, frontend: GRChomboComputeClass, *, out_dir: str = '.',
                 license_header: Optional[str] = None) -> None:
        super().__init__(frontend, GRChomboSplitGenerator(frontend), GRChomboSplitVisitor(), license_header=license_header)
        self.out_dir = out_dir

    def generate_header(self, banner: str = '') -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        code = self.code_visitor.visit(self.generator.generate_compute_class(banner))
        path = os.path.join(self.out_dir, f'{self.frontend.name}.hpp')
        with ConditionalFileUpdater(path) as fd:
            self.write_with_header(fd, code, '//')
        return path
