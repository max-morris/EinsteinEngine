from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar, Optional, Iterable

from node import Node
from visitor import Visitor


def get_class_name(x: Any) -> str:
    name = x.__class__.__name__
    assert isinstance(name, str)
    return name


T = TypeVar('T')


def try_get(d: Any, x: Any) -> Optional[T]:
    return d[x] if x in d else None


def visit_each(v: Visitor[Any], nodes: Iterable[Node]) -> list[str]:
    return [v.visit(n) for n in nodes]


class ReprEnum(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> ReprEnum:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member
