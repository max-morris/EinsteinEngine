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

from enum import Enum, auto


class CseOptimizationLevel(Enum):
    """
    Deprecated: use the cse_isolate_grid_variables bake option. Optimal is cse_isolate_grid_variables=True and Fast is
    cse_isolate_grid_variables=False. The name was misleading: this never chose any optimization, only whether grid
    variables are isolated from the CSE (see cse_isolate).
    """
    Fast = auto()
    Optimal = auto()
