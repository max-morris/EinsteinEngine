#  Copyright (C) 2025-2026 Max Morris and other Einstein Engine contributors.
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

from typing import Protocol, Optional

from EinsteinEngine import Centering


class SympyNameSubstitutionFn(Protocol):
    def __call__(self, name: str, in_stencil_args: bool) -> str: ...

class ShouldWrapWithAccessFn(Protocol):
    def __call__(self, name: str, in_stencil_args: bool) -> bool: ...

class VarCenteringFn(Protocol):
    def __call__(self, var_name: str) -> Optional[Centering]: ...
