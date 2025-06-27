from functools import wraps, partial
from typing import overload, Optional, Callable, cast

from EmitCactus.dsl.dsl_exception import DslException

@overload
def require_baked[**P, T](func: Callable[P, T], /) -> Callable[P, T]: ...

@overload
def require_baked[**P, T](*, msg: str = "Cannot call this method before baking the object.") -> Callable[[Callable[P, T]], Callable[P, T]]: ...

def require_baked[**P, T](func: Optional[Callable[P, T]] = None,
                          /, *,
                          msg: str = "Cannot call this method before baking the object.") -> Callable[[Callable[P, T]], Callable[P, T]] | Callable[P, T]:
    if not func:
        return partial(require_baked, msg=msg)

    @wraps(func)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> T:  # type: ignore[no-untyped-def]
        if not self.been_baked:
            raise DslException(msg)
        return func(self, *args, **kwargs)

    return cast(Callable[[Callable[P, T]], Callable[P, T]] | Callable[P, T], wrapper)