from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, NoReturn, Iterable, Any

from EmitCactus.emit.tree import CommonNode
from EmitCactus import util

# noinspection PyUnresolvedReferences
# MyPy needs these
from EmitCactus.emit.tree import Node
from typing import TypeVar

N = TypeVar('N', bound=Node)


class Visitor(ABC, Generic[N]):

    @abstractmethod
    def visit(self, n: N | CommonNode) -> str:
        pass

    def not_implemented(self, n: N | CommonNode) -> NoReturn:
        raise NotImplementedError(f'visit({util.get_class_name(n)}) not implemented in {util.get_class_name(self)}')


class VisitorException(Exception):
    pass


def visit_each(v: Visitor[Any], nodes: Iterable[Node]) -> list[str]:
    return [v.visit(n) for n in nodes]
