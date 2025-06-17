from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar, Optional, Callable, Generic, Iterator, Set

from types import TracebackType
from time import time, sleep


def get_class_name(x: Any) -> str:
    name = x.__class__.__name__
    assert isinstance(name, str)
    return name


def try_get[T](d: Any, x: Any) -> Optional[T]:
    return d[x] if x in d else None


def incr_and_get[K](d: dict[K, int], k: K) -> int:
    v = d[k] = d.get(k, 0) + 1
    return v


def get_or_compute[K, V](d: dict[K, V], k: K, f: Callable[[K], V]) -> V:
    if k in d:
        return d[k]
    else:
        v = f(k)
        d[k] = v
        return v


def consolidate[K, V](recipient: dict[K, V], donor: dict[K, V], f: Callable[[V, V], V]) -> None:
    for k, donor_v in donor.items():
        if k in recipient:
            recipient[k] = f(recipient[k], donor_v)
        else:
            recipient[k] = donor_v


def indent(s: str, spaces: int = 4) -> str:
    ind = ' ' * spaces
    split = s.split('\n')

    for i in range(len(split) - 1):
        split[i] += '\n'

    return ''.join([f'{ind}{s}' for s in split])


class ReprEnum(Enum):
    representation: str

    def __new__(cls, value: Any, representation: str) -> ReprEnum:
        member = object.__new__(cls)
        member._value_ = value
        member.representation = representation
        return member

    def __repr__(self) -> str:
        return self.representation


class CenteringEnum(Enum):
    string_repr: str
    int_repr: tuple[int, int, int]

    def __new__(cls, value: Any, string_repr: str, int_repr: tuple[int, int, int]) -> CenteringEnum:
        member = object.__new__(cls)
        member._value_ = value
        member.string_repr = string_repr
        member.int_repr = int_repr
        return member

    def __repr__(self) -> str:
        return self.string_repr


class ScheduleBinEnum(Enum):
    generic_name: str
    is_builtin: bool

    def __new__(cls, value: Any, generic_name: str, is_builtin: bool) -> ScheduleBinEnum:
        member = object.__new__(cls)
        member._value_ = value
        member.generic_name = generic_name
        member.is_builtin = is_builtin
        return member

    def __repr__(self) -> str:
        return self.generic_name


T0 = TypeVar('T0')


class OrderedSet(Set[T0], Generic[T0]):
    def __iter__(self) -> Iterator[T0]:
        r = set.__iter__(self)
        return sorted(list(r), key=lambda a: repr(a)).__iter__()


class ProgressBarImpl:
    def __init__(self, n_items: int, name: str, bar_size: int) -> None:
        self.n_items = n_items
        self.name = name
        self.bar_size = bar_size
        self.n = 0
        self.t0 = time()

    def __call__(self) -> None:
        self.n += 1
        n = self.n
        nt = self.n_items
        t0 = self.t0
        tn = time()
        frac = n / nt
        delt = tn - t0
        tt = delt / frac
        tr = tt - delt
        n_star = int(self.bar_size * n / nt)
        bar = ("*" * n_star) + (" " * (self.bar_size - n_star))
        if delt >= 1.5 and tt >= 3.0:
            print("%s: %s %d/%d (%.2f%%) time: (remaining: %.2fs, total: %.2fs)   " % (
                self.name, bar, n, nt, 100 * n / nt, tr, tt), end='\r')


class ProgressBar:
    def __init__(self, n_items: int, name: str = "progress", bar_size: int = 40) -> None:
        self.n_items = n_items
        self.bar_size = bar_size
        self.name = name

    def __enter__(self) -> ProgressBarImpl:
        return ProgressBarImpl(self.n_items, self.name, self.bar_size)

    def __exit__(self, ty: Optional[type[BaseException]], val: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        print()
