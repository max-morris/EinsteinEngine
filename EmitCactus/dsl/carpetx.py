from typing import Collection, Optional

import sympy
from sympy import IndexedBase, Indexed

from EmitCactus.dsl.use_indices import ScheduleBin
from EmitCactus.emit.ccl.schedule.schedule_tree import ScheduleBlock


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
                 name: Optional[str] = None):
        self.var = var
        self.val_at_infinity = val_at_infinity
        self.propagation_speed = propagation_speed
        self.radial_falloff_exponent = radial_falloff_exponent
        self.schedule_target = schedule_target
        self.schedule_before = schedule_before or list()
        self.schedule_after = schedule_after or list()

        if name is None:
            self.name = f'NewRadXBoundaryFn_{NewRadXBoundaryBatch._name_counter}'
            NewRadXBoundaryBatch._name_counter += 1
        else:
            self.name = name