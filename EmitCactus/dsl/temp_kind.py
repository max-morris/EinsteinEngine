from enum import Enum, auto


class TempKind(Enum):
    Local = auto()
    Tile = auto()
    Global = auto()