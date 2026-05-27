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

from typing import TYPE_CHECKING, TypeAlias

from EinsteinEngine.emit.ccl.schedule.schedule_tree import ScheduleBlock

if TYPE_CHECKING:
    from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin
else:
    # Runtime stub to avoid importing cactus_frontend and creating a circular import.
    class ScheduleBin:
        pass


ScheduleTarget: TypeAlias = ScheduleBin | ScheduleBlock

def safe_name(schedule_target: ScheduleTarget) -> str:
    name = getattr(schedule_target, "name", None)
    if name is not None:
        identifier = getattr(name, "identifier", None)
        if identifier is not None:
            return str(identifier)
    if hasattr(schedule_target, "generic_name"):
        return schedule_target.generic_name
    raise TypeError(f"Unsupported schedule target type: {type(schedule_target)}")
