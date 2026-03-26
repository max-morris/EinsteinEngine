from dataclasses import dataclass

@dataclass
class DimensionSingleton:
    value: int = 3

_dimension_container = DimensionSingleton()

def set_dimension(d: int) -> None:
    _dimension_container.value = d

def get_dimension() -> int:
    return _dimension_container.value
