from __future__ import annotations

from dataclasses import dataclass
from enum import auto
from util import ReprEnum, try_get
from typing import Optional, TypedDict
from typing_extensions import Unpack

from node import Node, String, Integer, Float, Identifier, Verbatim, LiteralExpression


class ParamNode(Node):
    pass


class ParamAccess(ReprEnum):
    Global = auto(), 'global'
    Restricted = auto(), 'restricted'
    Private = auto(), 'private'
    Shares = auto(), 'shares'


class ExtendsUses(ReprEnum):
    Extends = auto(), 'EXTENDS'
    Uses = auto(), 'USES'


class Steerability(ReprEnum):
    Never = auto(), 'NEVER',
    Always = auto(), 'ALWAYS'
    Recover = auto(), 'RECOVER'


class ParamType(ReprEnum):
    Int = auto(), 'INT'
    Real = auto(), 'REAL'
    Keyword = auto(), 'KEYWORD'
    String = auto(), 'STRING'
    Bool = auto(), 'BOOLEAN'


@dataclass
class IntParamDescWildcard(ParamNode):
    pass


@dataclass
class IntParamDescSingle(ParamNode):
    integer: Integer


@dataclass
class IntParamClosedLowerBound(ParamNode):
    integer: Integer


@dataclass
class IntParamOpenLowerBound(ParamNode):
    integer: Integer


@dataclass
class IntParamClosedUpperBound(ParamNode):
    integer: Integer


@dataclass
class IntParamOpenUpperBound(ParamNode):
    integer: Integer


IntParamLowerBound = IntParamDescWildcard | IntParamDescSingle | IntParamClosedLowerBound | IntParamOpenLowerBound
IntParamUpperBound = IntParamDescWildcard | IntParamDescSingle | IntParamClosedUpperBound | IntParamOpenUpperBound


@dataclass
class IntParamDescRange(ParamNode):
    lower_bound: IntParamLowerBound
    upper_bound: IntParamUpperBound


@dataclass
class IntParamDescRangeWithStep(ParamNode):
    lower_bound: IntParamLowerBound
    upper_bound: IntParamUpperBound
    step: Integer


IntParamDesc = IntParamDescWildcard | IntParamDescSingle | IntParamDescRange | IntParamDescRangeWithStep


@dataclass
class IntParamRange(ParamNode):
    range_desc: IntParamDesc
    comment: String


@dataclass
class RealParamDescWildcard(ParamNode):
    pass


@dataclass
class RealParamDescSingle(ParamNode):
    real: Float


@dataclass
class RealParamClosedLowerBound(ParamNode):
    real: Float


@dataclass
class RealParamOpenLowerBound(ParamNode):
    real: Float


@dataclass
class RealParamClosedUpperBound(ParamNode):
    real: Float


@dataclass
class RealParamOpenUpperBound(ParamNode):
    real: Float


RealParamLowerBound = RealParamDescWildcard | RealParamDescSingle | RealParamClosedLowerBound | RealParamOpenLowerBound
RealParamUpperBound = RealParamDescWildcard | RealParamDescSingle | RealParamClosedUpperBound | RealParamOpenUpperBound


@dataclass
class RealParamDescRange(ParamNode):
    lower_bound: RealParamLowerBound
    upper_bound: RealParamUpperBound


@dataclass
class RealParamDescRangeWithStep(ParamNode):
    lower_bound: RealParamLowerBound
    upper_bound: RealParamUpperBound
    step: Float


RealParamDesc = RealParamDescWildcard | RealParamDescSingle | RealParamDescRange | RealParamDescRangeWithStep


@dataclass
class RealParamRange(ParamNode):
    range_desc: RealParamDesc
    comment: String


@dataclass
class KeywordParamRange(ParamNode):
    values: list[String]
    comment: String


@dataclass
class StringParamRange(ParamNode):
    values: list[String]
    comment: String


ParamRange = IntParamRange | RealParamRange | KeywordParamRange | StringParamRange


class ParamOptionalArgs(TypedDict, total=False):
    extends_uses: ExtendsUses
    arr_len: Integer
    alias_name: Identifier
    steerability: Steerability
    accumulator: Verbatim
    accumulator_base: Identifier
    shares_with: Identifier


@dataclass(init=False)
class Param(ParamNode):
    param_access: ParamAccess
    param_type: ParamType
    param_name: Identifier
    param_desc: String
    range_descriptions: list[ParamRange]
    default_value: LiteralExpression
    extends_uses: Optional[ExtendsUses]
    arr_len: Optional[Integer]
    alias_name: Optional[Identifier]
    steerability: Optional[Steerability]
    accumulator: Optional[Verbatim]
    accumulator_base: Optional[Identifier]
    shares_with: Optional[Identifier]

    def __init__(self, param_access: ParamAccess, param_type: ParamType, param_name: Identifier, param_desc: String,
                 range_descriptions: list[ParamRange], default_value: LiteralExpression,
                 **kwargs: Unpack[ParamOptionalArgs]):
        self.param_access = param_access
        self.param_type = param_type
        self.param_name = param_name
        self.param_desc = param_desc
        self.range_descriptions = range_descriptions
        self.default_value = default_value
        self.extends_uses = try_get(kwargs, 'extends_uses')
        self.arr_len = try_get(kwargs, 'arr_len')
        self.alias_name = try_get(kwargs, 'alias_name')
        self.steerability = try_get(kwargs, 'steerability')
        self.accumulator = try_get(kwargs, 'accumulator')
        self.accumulator_base = try_get(kwargs, 'accumulator_base')
        self.shares_with = try_get(kwargs, 'shares_with')


@dataclass
class ParamRoot(ParamNode):
    params: list[Param]
