from enum import Enum
from functools import total_ordering


@total_ordering
class TempKind(Enum):
    Inline = 0
    Local = 1
    Tile = 2
    Global = 3

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TempKind):
            return NotImplemented

        return self.value == other.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TempKind):
            return NotImplemented

        return self.value < other.value

    def clamp(self, max_kind: 'TempKind', *, min_kind: 'TempKind|None' = None) -> 'TempKind':
        min_kind = min_kind or TempKind.Inline
        return max(min(self, max_kind), min_kind)

    def __repr__(self) -> str:
        return self.name.split('.')[1]
