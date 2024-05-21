from abc import ABC
from dataclasses import dataclass
from enum import auto

from util import ReprEnum


class Node(ABC):
    pass


class CommonNode(Node):
    pass


@dataclass
class Identifier(CommonNode):
    identifier: str


@dataclass
class Verbatim(CommonNode):
    text: str


@dataclass
class String(CommonNode):
    text: str


@dataclass
class Integer(CommonNode):
    integer: int


@dataclass
class Float(CommonNode):
    fl: float


@dataclass
class Bool(CommonNode):
    b: bool


class Language(ReprEnum):
    C = auto(), 'C'
    Fortran = auto(), 'Fortran'


LiteralExpression = Verbatim | String | Integer | Float | Bool
