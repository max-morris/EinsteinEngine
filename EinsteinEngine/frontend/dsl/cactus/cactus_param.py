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

import sys
from typing import cast, Optional, Type, Any

CactusParamDefaultType = float | int | str | bool
CactusParamValuesType = Optional[tuple[float, float] | tuple[int, int] | tuple[bool, bool] | str | set[str]]
CactusMinMaxType = tuple[float, float] | tuple[int, int]

class CactusParam:
    def __init__(self, name: str, default: CactusParamDefaultType, desc: str, values: CactusParamValuesType) -> None:
        self.name = name
        self.values = values
        self.desc = desc
        self.default = default

    def get_min_max(self) -> CactusMinMaxType:
        ty = self.get_type()
        if ty == int:
            if self.values is not None:
                return cast(CactusMinMaxType, self.values)
            return -2 ** 31, 2 ** 31 - 1
        elif ty == float:
            if self.values is not None:
                return cast(CactusMinMaxType, self.values)
            return sys.float_info.min, sys.float_info.max
        else:
            assert False

    def get_values(self) -> CactusParamValuesType:
        if self.values is not None:
            return self.values
        ty = self.get_type()
        if ty == bool:
            return False, True
        elif ty == str:
            return ".*"
        else:
            return self.get_min_max()

    def get_type(self) -> Type[Any]:
        if self.values is None:
            return type(self.default)
        elif isinstance(self.values, set):
            assert isinstance(self.default, str)
            return set  # keywords
        elif isinstance(self.values, str):
            # values is a regex
            assert isinstance(self.default, str)
            return str
        elif isinstance(self.values, tuple) and len(self.values) == 2:
            assert type(self.default) in [int, float]
            assert type(self.values[0]) in [int, float]
            assert type(self.values[1]) in [int, float]
            if isinstance(self.default, float) or isinstance(self.values[0], float) or isinstance(self.values[1],
                                                                                                  float):
                return float
            else:
                return int
        else:
            assert False

    def __repr__(self) -> str:
        return f"Param({self.name})"
