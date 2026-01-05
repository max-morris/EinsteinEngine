from bisect import bisect_left
from functools import lru_cache
from typing import Protocol, Any, Callable

from sympy import Symbol

from EmitCactus.dsl.dsl_exception import DslException
from EmitCactus.dsl.temp_kind import TempKind



class TemporaryPromotionPredicate(Protocol):
    """
    A TemporaryPromotionPredicate is a function or callable object that takes a Symbol and returns a TempKind indicating
    the "highest" level of temporary promotion allowed for that symbol, according to some TemporaryPromotionStrategy.
    """

    def __init__(self, complexities: dict[Symbol, int], /, **kwargs: Any) -> None:
        ...

    def __call__(self, temp_name: Symbol, /) -> TempKind:
        ...


def _get_complexity_ordered_symbols(complexities: dict[Symbol, int]) -> list[tuple[str, int]]:
    return [(str(sym), cx) for sym, cx in sorted(complexities.items(), key=lambda kv: kv[1])]


def _find_symbol(complexity_ordered_symbols: list[tuple[str, int]], needle: Symbol, needle_complexity: int) -> int:
    needle_name = str(needle)
    start = bisect_left(complexity_ordered_symbols, needle_complexity, key=lambda kv: kv[1])

    for i, (symbol_name, _) in enumerate(complexity_ordered_symbols[start:]):
        if symbol_name == needle_name:
            return i + start
    else:
        raise ValueError(f"symbol {needle} not found in {complexity_ordered_symbols}")


class PercentilePromotionPredicate:
    complexities: dict[Symbol, int]
    percentile: float
    complexity_ordered_symbols: list[tuple[str, int]]

    def __init__(self, complexities: dict[Symbol, int], /, *, percentile: float) -> None:
        self.complexities = complexities
        self.percentile = percentile
        if not 0.0 <= self.percentile <= 1.0:
            raise DslException(f"percentile must be between 0.0 and 1.0, got {percentile}")
        self.complexity_ordered_symbols = _get_complexity_ordered_symbols(complexities)

    @lru_cache
    def __call__(self, temp_name: Symbol, /) -> TempKind:
        if _find_symbol(self.complexity_ordered_symbols, temp_name, self.complexities[temp_name]) / len(self.complexity_ordered_symbols) >= self.percentile:
            return TempKind.Global
        else:
            return TempKind.Local


class ThresholdPromotionPredicate:
    complexities: dict[Symbol, int]
    threshold: int

    def __init__(self, complexities: dict[Symbol, int], /, *, threshold: int) -> None:
        self.complexities = complexities
        self.threshold = threshold
        if self.threshold < 0:
            raise DslException(f"threshold must be at least 0, got {self.threshold}")

    def __call__(self, temp_name: Symbol, /) -> TempKind:
        if self.complexities[temp_name] >= self.threshold:
            return TempKind.Global
        else:
            return TempKind.Local


class RankPromotionPredicate:
    complexities: dict[Symbol, int]
    max_promotions: int
    complexity_ordered_symbols: list[tuple[str, int]]

    def __init__(self, complexities: dict[Symbol, int], /, *, max_promotions: int):
        self.complexities = complexities
        self.max_promotions = max_promotions
        if self.max_promotions < 1:
            raise DslException(f"max_promotions must be at least 1, got {max_promotions}")
        self.complexity_ordered_symbols = _get_complexity_ordered_symbols(complexities)

    @lru_cache
    def __call__(self, temp_name: Symbol, /) -> TempKind:
        if len(self.complexity_ordered_symbols) - _find_symbol(self.complexity_ordered_symbols, temp_name, self.complexities[temp_name]) <= self.max_promotions:
            return TempKind.Global
        else:
            return TempKind.Local


class TruePromotionPredicate:
    def __init__(self, _complexities: dict[Symbol, int], /):
        pass

    # noinspection PyMethodMayBeStatic
    def __call__(self, _temp_name: Symbol, /) -> TempKind:
        return TempKind.Global


class FalsePromotionPredicate:
    def __init__(self, _complexities: dict[Symbol, int], /):
        pass

    # noinspection PyMethodMayBeStatic
    def __call__(self, _temp_name: Symbol, /) -> TempKind:
        return TempKind.Local


TemporaryPromotionStrategy = Callable[[dict[Symbol, int]], TemporaryPromotionPredicate]

class TemporaryPromotionStrategyFactory(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> TemporaryPromotionStrategy:
        ...


def promote_all() -> TemporaryPromotionStrategy:
    return lambda cx: TruePromotionPredicate(cx)

def promote_none() -> TemporaryPromotionStrategy:
    return lambda cx: FalsePromotionPredicate(cx)

def promote_percentile(percentile: float) -> TemporaryPromotionStrategy:
    return lambda cx: PercentilePromotionPredicate(cx, percentile=percentile)

def promote_threshold(threshold: int) -> TemporaryPromotionStrategy:
    return lambda cx: ThresholdPromotionPredicate(cx, threshold=threshold)

def promote_rank(max_promotions: int) -> TemporaryPromotionStrategy:
    return lambda cx: RankPromotionPredicate(cx, max_promotions=max_promotions)