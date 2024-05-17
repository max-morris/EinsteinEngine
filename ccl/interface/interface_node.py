from __future__ import annotations

from enum import Enum, auto
from typing import Any, TypedDict, Optional

from typing_extensions import Unpack

from node import Node, Identifier, Verbatim, Integer, String
from util import try_get


class InterfaceNode(Node):
    pass


class HeaderSection(InterfaceNode):
    implements: Identifier
    inherits: list[Identifier]
    friends: list[Identifier]

    def __init__(self, implements: Identifier, inherits: list[Identifier], friends: list[Identifier]):
        self.implements = implements
        self.inherits = inherits
        self.friends = friends


class IncludeType(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> IncludeType:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    Header = auto(), 'HEADER'
    Source = auto(), 'SOURCE'


class UsesInclude(InterfaceNode):
    typ: IncludeType
    file_name: Verbatim

    def __init__(self, file_name: Verbatim, typ: IncludeType = IncludeType.Header):
        self.typ = typ
        self.file_name = file_name


class IncludeIn(InterfaceNode):
    typ: IncludeType
    file_name: Verbatim
    file_to_include: Verbatim

    def __init__(self, file_name: Verbatim, file_to_include: Verbatim, typ: IncludeType = IncludeType.Header):
        self.typ = typ
        self.file_name = file_name
        self.file_to_include = file_to_include


class IncludeSection(InterfaceNode):
    directives: list[UsesInclude | IncludeIn]

    def __init__(self, directives: list[UsesInclude | IncludeIn]):
        self.directives = directives


class FunctionAliasReturnType(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> FunctionAliasReturnType:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    Void = auto(), 'void'
    Int = auto(), 'CCTK_INT'
    Real = auto(), 'CCTK_REAL'
    Complex = auto(), 'CCTK_COMPLEX'
    Pointer = auto(), 'CCTK_POINTER'
    PointerToConst = auto(), 'CCTK_POINTER_TO_CONST'


class FunctionAliasArgType(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> FunctionAliasArgType:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    String = auto(), 'STRING'
    Int = auto(), 'CCTK_INT'
    Real = auto(), 'CCTK_REAL'
    Complex = auto(), 'CCTK_COMPLEX'
    Pointer = auto(), 'CCTK_POINTER'
    PointerToConst = auto(), 'CCTK_POINTER_TO_CONST'


class FunctionAliasArgIntent(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> FunctionAliasArgIntent:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    In = auto(), 'IN'
    Out = auto(), 'OUT'
    InOut = auto(), 'INOUT'


class FunctionAliasArg(InterfaceNode):
    arg_type: FunctionAliasArgType
    arg_intent: FunctionAliasArgIntent
    is_array: bool
    arg_name: Identifier

    def __init__(self, arg_type: FunctionAliasArgType, arg_intent: FunctionAliasArgIntent, arg_name: Identifier,
                 is_array: bool = False):
        self.arg_type = arg_type
        self.arg_intent = arg_intent
        self.is_array = is_array
        self.arg_name = arg_name


class FunctionAliasFpArg(InterfaceNode):
    fp_return_type: FunctionAliasArgType
    fp_intent: FunctionAliasArgIntent
    is_array: bool
    fp_name: Identifier
    fp_args: list[FunctionAliasArg]

    # From Cactus reference manual: function pointers may not be nested.
    def __init__(self, fp_return_type: FunctionAliasArgType, fp_intent: FunctionAliasArgIntent, fp_name: Identifier, fp_args: list[FunctionAliasArg],
                 is_array: bool = False):
        self.fp_return_type = fp_return_type
        self.is_array = is_array
        self.fp_name = fp_name
        self.fp_args = fp_args
        self.fp_intent = fp_intent


class FunctionAlias(InterfaceNode):
    return_type: FunctionAliasReturnType
    alias: Identifier
    args: list[FunctionAliasArg | FunctionAliasFpArg]

    def __init__(self, return_type: FunctionAliasReturnType, alias: Identifier,
                 args: list[FunctionAliasArg | FunctionAliasFpArg]):
        self.return_type = return_type
        self.alias = alias
        self.args = args


class RequiresFunction(InterfaceNode):
    alias: Identifier

    def __init__(self, alias: Identifier):
        self.alias = alias


class UsesFunction(InterfaceNode):
    alias: Identifier

    def __init__(self, alias: Identifier):
        self.alias = alias


class ProvidingLanguage(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> ProvidingLanguage:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    C = auto(), 'C'
    Fortran = auto(), 'Fortran'


class ProvidesFunction(InterfaceNode):
    alias: Identifier
    provider: Identifier
    language: ProvidingLanguage

    def __init__(self, alias: Identifier, provider: Identifier, language: ProvidingLanguage):
        self.alias = alias
        self.provider = provider
        self.language = language


class FunctionSection(InterfaceNode):
    declarations: list[FunctionAlias | RequiresFunction | UsesFunction | ProvidesFunction]

    def __init__(self, declarations: list[FunctionAlias | RequiresFunction | UsesFunction | ProvidesFunction]):
        self.declarations = declarations


class Access(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> Access:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    Public = auto(), 'public'
    Protected = auto(), 'protected'
    Private = auto(), 'private'


class DataType(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> DataType:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    Char = auto(), 'CHAR'
    Byte = auto(), 'BYTE'
    Int = auto(), 'INT'
    Real = auto(), 'REAL'
    Complex = auto(), 'COMPLEX'


class GroupType(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> GroupType:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    GF = auto(), 'GF'
    Array = auto(), 'ARRAY'
    Scalar = auto(), 'SCALAR'


class DistribType(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> DistribType:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    Default = auto(), 'DEFAULT'
    Constant = auto(), 'CONSTANT'


class VariableGroupOptionalArgs(TypedDict, total=False):
    vector_size: Integer
    group_type: GroupType
    dim: Integer
    array_size: list[Integer]
    array_distrib: DistribType
    time_levels: Integer
    array_ghost_size: Integer
    stagger_spec: String
    tags: String
    group_description: String


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
    tags: Optional[String]
    group_description: Optional[String]

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


class VariableSection(InterfaceNode):
    variable_groups: list[VariableGroup]

    def __init__(self, variable_groups: list[VariableGroup]):
        self.variable_groups = variable_groups


class InterfaceRoot(InterfaceNode):
    header_section: HeaderSection
    include_section: IncludeSection
    function_section: FunctionSection
    variable_section: VariableSection

    def __init__(self, header_section: HeaderSection, include_section: IncludeSection,
                 function_section: FunctionSection, variable_section: VariableSection):
        self.header_section = header_section
        self.include_section = include_section
        self.function_section = function_section
        self.variable_section = variable_section
