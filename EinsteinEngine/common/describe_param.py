#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
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

"""A description of a recipe-facing param value that is stable across runs, used by the tuning probe and by reprs."""

import functools
import re
from enum import Enum
from types import BuiltinFunctionType, CellType, FunctionType, MethodType


_ADDRESS_RE = re.compile(r' at 0x[0-9a-fA-F]+')

#  How deep describe_param follows closures, partials and containers.
_MAX_DESCRIBE_DEPTH = 8


def describe_param(value: object) -> str:
    """A description of a recipe-facing param value that is stable across runs (and hash seeds).

    - Enums by class and member name; classes by qualified name.
    - Lists, tuples, dicts, sets and frozensets (exactly those types) element by element, with set elements and dict
      entries sorted by their descriptions.
    - ``functools.partial`` objects by their function, arguments and keywords (sorted).
    - Functions by their qualified name. Lambdas and nested functions (``<lambda>`` or ``<locals>`` in the qualified
      name) also by their defaults and the contents of their closure, if any, so that ``cartesian_product(a, b)`` and
      ``cartesian_product(b, a)`` differ. A function's code and the globals it reads are not described: two lambdas
      in one scope with the same defaults and closure describe the same way.
    - Wrappers with ``__wrapped__`` (e.g. ``functools.cache``) as what they wrap.
    - Bound methods by their function, and by their instance if its class defines ``__repr__``.
    - Anything else by its repr, with memory addresses removed. An object with the default repr is thus described by
      its class alone. Ordering functions made by ``add_eqn_order`` / ``add_eqn_key_order`` have a repr that lists
      their order or key.

    Recursion stops at a depth of ``_MAX_DESCRIBE_DEPTH`` (``...``) and at cycles (``<cycle>``).
    """
    return _describe(value, 0, frozenset())


def _describe(value: object, depth: int, enclosing: frozenset[int]) -> str:
    if depth > _MAX_DESCRIBE_DEPTH:
        return '...'
    if id(value) in enclosing:
        return '<cycle>'
    enclosing = enclosing | {id(value)}

    def inner(v: object) -> str:
        return _describe(v, depth + 1, enclosing)

    if isinstance(value, Enum):
        return f'{type(value).__qualname__}.{value.name}'
    if isinstance(value, type):
        return f'{value.__module__}.{value.__qualname__}'
    if type(value) is list:
        return f'[{", ".join(map(inner, value))}]'
    if type(value) is tuple:
        return f'({", ".join(map(inner, value))}{"," if len(value) == 1 else ""})'
    if type(value) is dict:
        return f'{{{", ".join(sorted(f"{inner(k)}: {inner(v)}" for k, v in value.items()))}}}'
    if isinstance(value, (set, frozenset)) and type(value) in (set, frozenset):
        if not value:
            return f'{type(value).__name__}()'
        items = ", ".join(sorted(map(inner, value)))
        return f'{{{items}}}' if isinstance(value, set) else f'frozenset({{{items}}})'
    if isinstance(value, functools.partial):
        keywords = [f'{key}={inner(arg)}' for key, arg in sorted(value.keywords.items())]
        return f'partial({", ".join([inner(value.func), *map(inner, value.args), *keywords])})'
    if isinstance(value, MethodType):
        owner = value.__self__
        if type(owner).__repr__ is object.__repr__:
            return inner(value.__func__)
        return f'{inner(value.__func__)} of {inner(owner)}'
    if isinstance(value, (FunctionType, BuiltinFunctionType)):
        module = getattr(value, '__module__', None)
        name = value.__qualname__ if module is None else f'{module}.{value.__qualname__}'
        # A function defined at module or class level is identified by its qualified name. A lambda or a nested
        # function is not, so it is also described by what it captured.
        if not isinstance(value, FunctionType) or '<' not in value.__qualname__:
            return name
        details = []
        if value.__defaults__:
            details.append(f'defaults={inner(value.__defaults__)}')
        if value.__kwdefaults__:
            details.append(f'kwdefaults={inner(value.__kwdefaults__)}')
        if value.__closure__:
            details.append(f'closure={inner(tuple(map(_cell_contents, value.__closure__)))}')
        return f'{name}[{", ".join(details)}]' if details else name
    if (wrapped := getattr(value, '__dict__', {}).get('__wrapped__')) is not None:
        return inner(wrapped)
    return _ADDRESS_RE.sub('', repr(value))


class _EmptyCell:
    def __repr__(self) -> str:
        return '<empty>'


def _cell_contents(cell: CellType) -> object:
    try:
        return cell.cell_contents
    except ValueError:  # the variable is not assigned yet
        return _EmptyCell()
