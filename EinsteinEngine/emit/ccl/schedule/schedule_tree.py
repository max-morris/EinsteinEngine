#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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

from dataclasses import dataclass
from enum import auto, Enum
from typing import TypedDict, Optional
from typing import Union, List

from typing_extensions import Unpack

from EinsteinEngine.emit.tree import Node, Identifier, Integer, String, Language
from EinsteinEngine.util import ReprEnum, try_get


class ScheduleNode(Node):
    pass


@dataclass
class StorageDecl(ScheduleNode):
    group: Identifier
    time_levels: Union[Integer, Identifier]


@dataclass
class StorageLine(ScheduleNode):
    decls: List[StorageDecl]


class AtOrIn(ReprEnum):
    At = auto(), 'AT'  # For builtins
    In = auto(), 'IN'  # For user-defined groups


class GroupOrFunction(Enum):
    Function = auto()
    Group = auto()


class IntentRegion(ReprEnum):
    Interior = auto(), 'Interior'
    Boundary = auto(), 'Boundary'
    Everywhere = auto(), 'Everywhere'

    def consolidate(self, other: 'IntentRegion') -> 'IntentRegion':
        return self if self is other else IntentRegion.Everywhere


@dataclass(frozen=True)
class Intent(ScheduleNode):
    name: Identifier
    region: Optional[IntentRegion]


class ScheduleBlockOptionalArgs(TypedDict, total=False):
    alias: Identifier
    while_var: Identifier
    if_var: Identifier
    before: List[Identifier]
    after: List[Identifier]
    lang: Language
    storage: StorageLine
    trigger: List[Identifier]
    sync: List[Identifier]
    options: List[Identifier]
    reads: List[Intent]
    writes: List[Intent]


@dataclass(init=False)
class ScheduleBlock(ScheduleNode):
    group_or_function: GroupOrFunction
    name: Identifier
    at_or_in: AtOrIn
    schedule_bin: Identifier
    description: String
    alias: Optional[Identifier]
    while_var: Optional[Identifier]
    if_var: Optional[Identifier]
    before: Optional[List[Identifier]]
    after: Optional[List[Identifier]]
    lang: Optional[Language]
    storage: Optional[StorageLine]
    trigger: Optional[List[Identifier]]
    sync: Optional[List[Identifier]]
    options: Optional[List[Identifier]]
    reads: Optional[List[Intent]]
    writes: Optional[List[Intent]]

    def __init__(self, group_or_function: GroupOrFunction, name: Identifier, at_or_in: AtOrIn, schedule_bin: Identifier,
                 description: String, **kwargs: Unpack[ScheduleBlockOptionalArgs]):
        self.group_or_function = group_or_function
        self.name = name
        self.at_or_in = at_or_in
        self.schedule_bin = schedule_bin
        self.description = description
        self.alias = try_get(kwargs, 'alias')
        self.while_var = try_get(kwargs, 'while_var')
        self.if_var = try_get(kwargs, 'if_var')
        self.before = try_get(kwargs, 'before')
        self.after = try_get(kwargs, 'after')
        self.lang = try_get(kwargs, 'lang')
        self.storage = try_get(kwargs, 'storage')
        self.trigger = try_get(kwargs, 'trigger')
        self.sync = try_get(kwargs, 'sync')
        self.options = try_get(kwargs, 'options')
        self.reads = try_get(kwargs, 'reads')
        self.writes = try_get(kwargs, 'writes')

        if self.reads is not None and len(self.reads) == 0:
            self.reads = None

        if self.writes is not None and len(self.writes) == 0:
            self.writes = None

    def __hash__(self) -> int:
        return hash(self.name)


# todo: schedule.ccl supports else-if/else and more complex boolean predicates,
#       but this is sufficient for our current purposes. We ought to extend this functionality later.
@dataclass
class IfStatement(ScheduleNode):
    cond: Identifier
    then: List['ScheduleStatement']


ScheduleStatement = ScheduleBlock | StorageLine | IfStatement

@dataclass
class ScheduleRoot(ScheduleNode):
    statements: List[ScheduleStatement]
