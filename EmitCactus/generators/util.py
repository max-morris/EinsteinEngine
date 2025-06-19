from typing import Protocol


class SympyNameSubstitutionFn(Protocol):
    def __call__(self, name: str, in_stencil_args: bool) -> str: ...