from dataclasses import dataclass
from enum import auto, Enum
from typing import TypedDict, Optional
from typing_extensions import Unpack

from tree import Node, Identifier, Integer, String, Language
from util import ReprEnum, try_get


class ScheduleNode(Node):
    pass


@dataclass
class StorageDecl(ScheduleNode):
    group: Identifier
    time_levels: Integer | Identifier


@dataclass
class StorageLine(ScheduleNode):
    decls: list[StorageDecl]


@dataclass
class StorageSection(ScheduleNode):
    lines: list[StorageLine]


class AtOrIn(ReprEnum):
    At = auto(), 'AT'
    In = auto(), 'IN'


class GroupOrFunction(Enum):
    Function = auto()
    Group = auto()


class IntentRegion(ReprEnum):
    Interior = auto(), 'Interior'
    Boundary = auto(), 'Boundary'
    Everywhere = auto(), 'Everywhere'


@dataclass
class Intent(ScheduleNode):
    name: Identifier
    region: IntentRegion


class ScheduleBlockOptionalArgs(TypedDict, total=False):
    alias: Identifier
    while_var: Identifier
    if_var: Identifier
    before: list[Identifier]
    after: list[Identifier]
    lang: Language
    storage: StorageLine
    trigger: list[Identifier]
    sync: list[Identifier]
    options: list[Identifier]
    reads: list[Intent]
    writes: list[Intent]


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
    before: Optional[list[Identifier]]
    after: Optional[list[Identifier]]
    lang: Optional[Language]
    storage: Optional[StorageLine]
    trigger: Optional[list[Identifier]]
    sync: Optional[list[Identifier]]
    options: Optional[list[Identifier]]
    reads: Optional[list[Intent]]
    writes: Optional[list[Intent]]

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


@dataclass
class ScheduleSection(ScheduleNode):
    schedule_blocks: list[ScheduleBlock]


@dataclass
class ScheduleRoot(ScheduleNode):
    storage_section: StorageSection
    schedule_section: ScheduleSection
