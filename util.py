from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar, Optional


def get_class_name(x: Any) -> str:
    name = x.__class__.__name__
    assert isinstance(name, str)
    return name


T = TypeVar('T')


def try_get(d: Any, x: Any) -> Optional[T]:
    return d[x] if x in d else None


def indent(s: str, spaces: int = 4) -> str:
    ind = ' ' * spaces
    split = s.split('\n')

    for i in range(len(split) - 1):
        split[i] += '\n'

    return ''.join([f'{ind}{s}' for s in split])


class ReprEnum(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> ReprEnum:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    def __repr__(self) -> str:
        return self.representation


class CenteringEnum(Enum):
    string_repr: str
    int_repr: tuple[int, int, int]

    def __new__(cls, value: Any, string_repr: str, int_repr: tuple[int, int, int]) -> CenteringEnum:
        member = object.__new__(cls)
        member._value_ = value
        member.string_repr = string_repr
        member.int_repr = int_repr
        return member

    def __repr__(self) -> str:
        return self.string_repr
