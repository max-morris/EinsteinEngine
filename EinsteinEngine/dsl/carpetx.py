#  Copyright (C) 2025-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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

from typing import Collection, Optional

import sympy
from sympy import IndexedBase, Indexed

from EinsteinEngine.dsl.use_indices import ScheduleBin
from EinsteinEngine.emit.ccl.schedule.schedule_tree import ScheduleBlock


class ExplicitSyncBatch:
    vars: Collection[IndexedBase]
    schedule_target: ScheduleBin | ScheduleBlock
    schedule_before: Collection[str]
    schedule_after: Collection[str]
    name: str

    _name_counter: int = 0

    def __init__(self,
                 vars: Collection[IndexedBase],
                 schedule_target: ScheduleBin | ScheduleBlock,
                 *,
                 schedule_before: Optional[Collection[str]] = None,
                 schedule_after: Optional[Collection[str]] = None,
                 name: Optional[str] = None):
        self.vars = vars
        self.schedule_target = schedule_target
        self.schedule_before = schedule_before or list()
        self.schedule_after = schedule_after or list()

        if name is None:
            self.name = f'DummySyncFn_{ExplicitSyncBatch._name_counter}'
            ExplicitSyncBatch._name_counter += 1
        else:
            self.name = name


class NewRadXBoundaryBatch:
    var: IndexedBase | Indexed
    val_at_infinity: sympy.Expr
    propagation_speed: sympy.Expr
    radial_falloff_exponent: sympy.Expr
    schedule_target: ScheduleBin | ScheduleBlock
    schedule_before: Collection[str]
    schedule_after: Collection[str]
    name: str
    cond: Optional[str]

    _name_counter: int = 0


    def __init__(self,
                 var: IndexedBase | Indexed,
                 val_at_infinity: sympy.Expr,
                 propagation_speed: sympy.Expr,
                 radial_falloff_exponent: sympy.Expr,
                 schedule_target: ScheduleBin | ScheduleBlock,
                 *,
                 schedule_before: Optional[Collection[str]] = None,
                 schedule_after: Optional[Collection[str]] = None,
                 name: Optional[str] = None,
                 cond: Optional[str] = None):
        self.var = var
        self.val_at_infinity = val_at_infinity
        self.propagation_speed = propagation_speed
        self.radial_falloff_exponent = radial_falloff_exponent
        self.schedule_target = schedule_target
        self.schedule_before = schedule_before or list()
        self.schedule_after = schedule_after or list()
        self.cond = cond

        if name is None:
            self.name = f'NewRadXBoundaryFn_{NewRadXBoundaryBatch._name_counter}'
            NewRadXBoundaryBatch._name_counter += 1
        else:
            self.name = name
