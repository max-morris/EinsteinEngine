from typing import Protocol, Optional

from EinsteinEngine import Centering


class SympyNameSubstitutionFn(Protocol):
    def __call__(self, name: str, in_stencil_args: bool) -> str: ...

class ShouldWrapWithAccessFn(Protocol):
    def __call__(self, name: str, in_stencil_args: bool) -> bool: ...

class VarCenteringFn(Protocol):
    def __call__(self, var_name: str) -> Optional[Centering]: ...