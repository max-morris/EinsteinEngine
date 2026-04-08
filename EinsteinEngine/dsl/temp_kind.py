#  Copyright (C) 2024-2026 Max Morris and other Einstein Engine contributors.
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

from enum import Enum
from functools import total_ordering


@total_ordering
class TempKind(Enum):
    Inline = 0
    Local = 1
    Tile = 2
    Global = 3

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TempKind):
            return NotImplemented

        return self.value == other.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TempKind):
            return NotImplemented

        return self.value < other.value

    def clamp(self, max_kind: 'TempKind', *, min_kind: 'TempKind|None' = None) -> 'TempKind':
        min_kind = min_kind or TempKind.Inline
        return max(min(self, max_kind), min_kind)

    def __repr__(self) -> str:
        return self.name.split('.')[1]
