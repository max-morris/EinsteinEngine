from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, NoReturn

from node import CommonNode
from util import *

# noinspection PyUnresolvedReferences
# MyPy needs these
from node import Node
from typing import TypeVar

N = TypeVar('N', bound=Node)


class Visitor(ABC, Generic[N]):

    @abstractmethod
    def visit(self, n: N | CommonNode) -> str:
        pass

    def not_implemented(self, n: N | CommonNode) -> NoReturn:
        raise NotImplementedError(f'visit({get_class_name(n)}) not implemented in {get_class_name(self)}')
