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

from typing import Union
from abc import ABC
from dataclasses import dataclass
from enum import auto

from EinsteinEngine.util import ReprEnum, CenteringEnum


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


class Centering(CenteringEnum):
    VVV = auto(), 'VVV', (0, 0, 0)
    CVV = auto(), 'CVV', (1, 0, 0)
    VCV = auto(), 'VCV', (0, 1, 0)
    VVC = auto(), 'VVC', (0, 0, 1)
    CCV = auto(), 'CCV', (1, 1, 0)
    VCC = auto(), 'VCC', (0, 1, 1)
    CVC = auto(), 'CVC', (1, 0, 1)
    CCC = auto(), 'CCC', (1, 1, 1)


LiteralExpression = Union[Verbatim, String, Integer, Float, Bool]
