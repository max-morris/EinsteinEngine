#  Copyright (C) 2024-2026 Max Morris and other Einstein Engine contributors.
#
#  This file is part of the Einstein Engine (EinsteinEngine).
#
#  EinsteinEngine is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  EinsteinEngine is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from functools import wraps, partial
from typing import overload, Optional, Callable, cast, Collection, List

from sympy import Function, Symbol, Expr

from EinsteinEngine.dsl.dsl_exception import DslException
from EinsteinEngine.dsl.functions import stencil
from EinsteinEngine.dsl.sympywrap import cse_return, cse


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


def cse_isolate(exprs: List[Expr], symbols_to_isolate: Optional[Collection[Symbol]] = None) -> cse_return:
    if symbols_to_isolate:
        dummy = Function('isolated_symbol_dummy')
        iso_map = {s: dummy(s) for s in symbols_to_isolate}
        inv_iso_map = {v: k for k, v in iso_map.items()}
        exprs_sub = [e.subs(iso_map) for e in exprs]  # type: ignore[no-untyped-call]

        # When calling `stencil()`, we must use the naked GF name as declared -- i.e., not wrapped in `access()` -- so don't pull those out.
        exprs_sub = [e.replace(
            lambda e: e.is_Function and e.func == stencil and e.args and e.args[0] in inv_iso_map,
            lambda e: stencil(inv_iso_map[e.args[0]], *e.args[1:])
        ) for e in exprs_sub]

        new_syms, new_exprs = cse(exprs_sub)

        # Restore original symbols
        new_syms = [(lhs, rhs.subs(inv_iso_map)) for lhs, rhs in new_syms]  # type: ignore[no-untyped-call]
        new_exprs = [e.subs(inv_iso_map) for e in new_exprs]  # type: ignore[no-untyped-call]

        return new_syms, new_exprs
    else:
        return cse(exprs)

def require[T](expr: Optional[T], msg: Callable[[], str] = lambda: "Expression is required.") -> T:
    if expr is None:
        raise DslException(msg())
    return expr
