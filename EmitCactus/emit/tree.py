from typing import Union
from abc import ABC
from dataclasses import dataclass
from enum import auto

from EmitCactus.util import ReprEnum


class Node(ABC):
    pass


class CommonNode(Node):
    pass


@dataclass
class Identifier(CommonNode):
    identifier: str

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Identifier) and self.identifier.__eq__(__value.identifier)

    def __hash__(self) -> int:
        return self.identifier.__hash__()


@dataclass
class Verbatim(CommonNode):
    text: str


@dataclass(init=False)
class String(CommonNode):
    text: str
    single_quotes: bool

    def __init__(self, text: str, single_quotes: bool = False):
        self.text = text
        self.single_quotes = single_quotes


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


LiteralExpression = Union[Verbatim, String, Integer, Float, Bool]
