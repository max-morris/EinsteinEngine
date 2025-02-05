from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import auto
from typing import TypedDict, Optional

from typing_extensions import Unpack

from EmitCactus.emit.tree import Node, Identifier, Verbatim, Integer, String, Language, Centering, CommonNode, Bool
from EmitCactus.util import try_get, ReprEnum


class InterfaceNode(Node):
    pass


@dataclass
class HeaderSection(InterfaceNode):
    implements: Identifier
    inherits: list[Identifier]
    friends: list[Identifier]


class IncludeType(ReprEnum):
    Header = auto(), 'HEADER'
    Source = auto(), 'SOURCE'


@dataclass
class UsesInclude(InterfaceNode):
    file_name: Verbatim
    typ: IncludeType = IncludeType.Header


@dataclass
class IncludeIn(InterfaceNode):
    file_name: Verbatim
    file_to_include: Verbatim
    typ: IncludeType = IncludeType.Header


@dataclass
class IncludeSection(InterfaceNode):
    directives: list[UsesInclude | IncludeIn]


class FunctionAliasReturnType(ReprEnum):
    Void = auto(), 'void'
    Int = auto(), 'CCTK_INT'
    Real = auto(), 'CCTK_REAL'
    Complex = auto(), 'CCTK_COMPLEX'
    Pointer = auto(), 'CCTK_POINTER'
    PointerToConst = auto(), 'CCTK_POINTER_TO_CONST'


class FunctionAliasArgType(ReprEnum):
    String = auto(), 'STRING'
    Int = auto(), 'CCTK_INT'
    Real = auto(), 'CCTK_REAL'
    Complex = auto(), 'CCTK_COMPLEX'
    Pointer = auto(), 'CCTK_POINTER'
    PointerToConst = auto(), 'CCTK_POINTER_TO_CONST'


class FunctionAliasArgIntent(ReprEnum):
    In = auto(), 'IN'
    Out = auto(), 'OUT'
    InOut = auto(), 'INOUT'


@dataclass
class FunctionAliasArg(InterfaceNode):
    arg_type: FunctionAliasArgType
    arg_intent: FunctionAliasArgIntent
    arg_name: Identifier
    is_array: bool = False


@dataclass
class FunctionAliasFpArg(InterfaceNode):
    fp_return_type: FunctionAliasArgType
    fp_intent: FunctionAliasArgIntent
    fp_name: Identifier
    fp_args: list[FunctionAliasArg]
    is_array: bool = False


@dataclass
class FunctionAlias(InterfaceNode):
    return_type: FunctionAliasReturnType
    alias: Identifier
    args: list[FunctionAliasArg | FunctionAliasFpArg]


@dataclass
class RequiresFunction(InterfaceNode):
    alias: Identifier


@dataclass
class UsesFunction(InterfaceNode):
    alias: Identifier


@dataclass
class ProvidesFunction(InterfaceNode):
    alias: Identifier
    provider: Identifier
    language: Language


@dataclass
class FunctionSection(InterfaceNode):
    declarations: list[FunctionAlias | RequiresFunction | UsesFunction | ProvidesFunction]


class Access(ReprEnum):
    Public = auto(), 'public'
    Protected = auto(), 'protected'
    Private = auto(), 'private'


class DataType(ReprEnum):
    Char = auto(), 'CHAR'
    Byte = auto(), 'BYTE'
    Int = auto(), 'INT'
    Real = auto(), 'REAL'
    Complex = auto(), 'COMPLEX'


class GroupType(ReprEnum):
    GF = auto(), 'GF'
    Array = auto(), 'ARRAY'
    Scalar = auto(), 'SCALAR'


class DistribType(ReprEnum):
    Default = auto(), 'DEFAULT'
    Constant = auto(), 'CONSTANT'


class Parity(ReprEnum):
    Positive = auto(), '+1'
    Negative = auto(), '-1'


@dataclass
class SingleIndexParity(InterfaceNode):
    x_parity: Parity
    y_parity: Parity
    z_parity: Parity


@dataclass
class TensorParity(InterfaceNode):
    parities: list[SingleIndexParity]


class TagPropertyNode(InterfaceNode):
    @abstractmethod
    def get_key(self) -> Identifier:
        ...

    @abstractmethod
    def get_value(self) -> CommonNode | InterfaceNode:
        ...


@dataclass
class CheckpointTag(TagPropertyNode):
    do_checkpoint: Bool

    def get_key(self) -> Identifier:
        return Identifier('checkpoint')

    def get_value(self) -> Bool:
        return self.do_checkpoint


@dataclass
class RhsTag(TagPropertyNode):
    rhs_name: String

    def get_key(self) -> Identifier:
        return Identifier('rhs')

    def get_value(self) -> String:
        return self.rhs_name


@dataclass
class ParityTag(TagPropertyNode):
    parity: TensorParity

    def get_key(self) -> Identifier:
        return Identifier('parities')

    def get_value(self) -> TensorParity:
        return self.parity


@dataclass
class GroupTags(InterfaceNode):
    tags: list[TagPropertyNode]


class VariableGroupOptionalArgs(TypedDict, total=False):
    vector_size: Integer
    group_type: GroupType
    dim: Integer
    array_size: list[Integer]
    array_distrib: DistribType
    time_levels: Integer
    array_ghost_size: Integer
    stagger_spec: String
    tags: GroupTags
    group_description: String
    centering: Centering


class VariableGroup(InterfaceNode):
    access: Access
    group_name: Identifier
    data_type: DataType
    variable_names: list[Identifier]
    vector_size: Optional[Integer]
    group_type: Optional[GroupType]
    dim: Optional[Integer]
    array_size: Optional[list[Integer]]
    array_distrib: Optional[DistribType]
    time_levels: Optional[Integer]
    array_ghost_size: Optional[Integer]
    stagger_spec: Optional[String]
    tags: Optional[GroupTags]
    group_description: Optional[String]
    centering: Optional[Centering]

    def __init__(self, access: Access, group_name: Identifier, data_type: DataType, variable_names: list[Identifier],
                 **kwargs: Unpack[VariableGroupOptionalArgs]):
        self.access = access
        self.group_name = group_name
        self.data_type = data_type
        self.variable_names = variable_names
        self.vector_size = try_get(kwargs, 'vector_size')
        self.group_type = try_get(kwargs, 'group_type')
        self.dim = try_get(kwargs, 'dim')
        self.array_size = try_get(kwargs, 'array_size')
        self.array_distrib = try_get(kwargs, 'array_distrib')
        self.time_levels = try_get(kwargs, 'time_levels')
        self.array_ghost_size = try_get(kwargs, 'array_ghost_size')
        self.stagger_spec = try_get(kwargs, 'stagger_spec')
        self.tags = try_get(kwargs, 'tags')
        self.group_description = try_get(kwargs, 'group_description')
        self.centering = try_get(kwargs, 'centering')


@dataclass
class VariableSection(InterfaceNode):
    variable_groups: list[VariableGroup]


@dataclass
class InterfaceRoot(InterfaceNode):
    header_section: HeaderSection
    include_section: IncludeSection
    function_section: FunctionSection
    variable_section: VariableSection
