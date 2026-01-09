from abc import ABC, abstractmethod
from bisect import bisect_left
from functools import lru_cache
from typing import Protocol, Any, Optional

from sympy import Symbol

from EmitCactus.dsl.dsl_exception import DslException
from EmitCactus.dsl.temp_kind import TempKind


# region Promotion Predicates

class TemporaryPromotionPredicate(Protocol):
    """
    A TemporaryPromotionPredicate is a function or callable object that takes a Symbol and returns a TempKind indicating
    the "highest" level of temporary promotion allowed for that symbol, according to some TemporaryPromotionStrategy.
    """

    def __init__(self, complexities: dict[Symbol, int], /, **kwargs: Any) -> None:
        ...

    def __call__(self, temp_name: Symbol, /) -> TempKind:
        ...


def _get_complexity_ordered_symbols(complexities: dict[Symbol, int]) -> list[tuple[Symbol, int]]:
    return [(sym, cx) for sym, cx in sorted(complexities.items(), key=lambda kv: kv[1])]


def _find_symbol(complexity_ordered_symbols: list[tuple[Symbol, int]], needle: Symbol, needle_complexity: int) -> int:
    if (r := _try_find_symbol(complexity_ordered_symbols, needle, needle_complexity)) is not None:
        return r
    else:
        raise ValueError(
            f"Could not find symbol {needle} in complexity ordered list with complexity {needle_complexity}. "
            f"Complexity ordered list: {complexity_ordered_symbols}"
        )
    

def _try_find_symbol(complexity_ordered_symbols: list[tuple[Symbol, int]], needle: Symbol, needle_complexity: int) -> Optional[int]:
    needle_name = str(needle)
    start = bisect_left(complexity_ordered_symbols, needle_complexity, key=lambda kv: kv[1])

    for i, (sym, _) in enumerate(complexity_ordered_symbols[start:]):
        if str(sym) == needle_name:
            return i + start
    else:
        return None


class PercentilePromotionPredicate:
    complexities: dict[Symbol, int]
    temp_kinds: dict[Symbol, TempKind]
    percentile: float
    complexity_ordered_global_symbols: list[tuple[Symbol, int]]

    def __init__(self, complexities: dict[Symbol, int], temp_kinds: dict[Symbol, TempKind], /, *, percentile: float) -> None:
        self.complexities = complexities
        self.temp_kinds = temp_kinds
        self.percentile = percentile
        if not 0.0 <= self.percentile <= 1.0:
            raise DslException(f"percentile must be between 0.0 and 1.0, got {percentile}")
        self.complexity_ordered_global_symbols = list(filter(lambda kv: self.temp_kinds.get(kv[0]) is TempKind.Global, _get_complexity_ordered_symbols(complexities)))

    @lru_cache
    def __call__(self, temp_name: Symbol, /) -> TempKind:
        if (sym_pos := _try_find_symbol(self.complexity_ordered_global_symbols, temp_name, self.complexities[temp_name])) is None:
            return TempKind.Local

        if sym_pos / len(self.complexity_ordered_global_symbols) >= self.percentile:
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
    temp_kinds: dict[Symbol, TempKind]
    max_promotions: int
    complexity_ordered_global_symbols: list[tuple[Symbol, int]]

    def __init__(self, complexities: dict[Symbol, int], temp_kinds: dict[Symbol, TempKind], /, *, max_promotions: int):
        self.complexities = complexities
        self.temp_kinds = temp_kinds
        self.max_promotions = max_promotions
        if self.max_promotions < 1:
            raise DslException(f"max_promotions must be at least 1, got {max_promotions}")
        self.complexity_ordered_global_symbols = list(filter(lambda kv: self.temp_kinds.get(kv[0]) is TempKind.Global, _get_complexity_ordered_symbols(complexities)))

    @lru_cache
    def __call__(self, temp_name: Symbol, /) -> TempKind:
        if (sym_pos := _try_find_symbol(self.complexity_ordered_global_symbols, temp_name, self.complexities[temp_name])) is None:
            return TempKind.Local
        
        if len(self.complexity_ordered_global_symbols) - sym_pos <= self.max_promotions:
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


# endregion
# region Promotion Strategies

class OnePassTemporaryPromotionStrategy(ABC):
    @abstractmethod
    def __call__(self, complexities: dict[Symbol, int]) -> TemporaryPromotionPredicate:
        ...
    
class TwoPassTemporaryPromotionStrategy(ABC):
    @abstractmethod
    def __call__(self, complexities: dict[Symbol, int], temp_kinds: dict[Symbol, TempKind]) -> TemporaryPromotionPredicate:
        ...
    
TemporaryPromotionStrategy = OnePassTemporaryPromotionStrategy | TwoPassTemporaryPromotionStrategy


class _AllPromotionStrategy(OnePassTemporaryPromotionStrategy):
    def __call__(self, complexities: dict[Symbol, int]) -> TruePromotionPredicate:
        return TruePromotionPredicate(complexities)


class _NonePromotionStrategy(OnePassTemporaryPromotionStrategy):
    def __call__(self, complexities: dict[Symbol, int]) -> FalsePromotionPredicate:
        return FalsePromotionPredicate(complexities)


class _PercentilePromotionStrategy(TwoPassTemporaryPromotionStrategy):
    def __init__(self, percentile: float) -> None:
        self.percentile = percentile

    def __call__(self, complexities: dict[Symbol, int], temp_kinds: dict[Symbol, TempKind]) -> PercentilePromotionPredicate:
        return PercentilePromotionPredicate(complexities, temp_kinds, percentile=self.percentile)


class _ThresholdPromotionStrategy(OnePassTemporaryPromotionStrategy):
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def __call__(self, complexities: dict[Symbol, int]) -> ThresholdPromotionPredicate:
        return ThresholdPromotionPredicate(complexities, threshold=self.threshold)


class _RankPromotionStrategy(TwoPassTemporaryPromotionStrategy):
    def __init__(self, max_promotions: int) -> None:
        self.max_promotions = max_promotions

    def __call__(self, complexities: dict[Symbol, int], temp_kinds: dict[Symbol, TempKind]) -> RankPromotionPredicate:
        return RankPromotionPredicate(complexities, temp_kinds, max_promotions=self.max_promotions)

        

# endregion
# region Promotion Strategy Factories for DSL

def promote_all() -> TemporaryPromotionStrategy:
    """
    Allow all temporaries to be promoted as far as Global.
    """

    return _AllPromotionStrategy()


def promote_none() -> TemporaryPromotionStrategy:
    """
    Force all temporaries to be Local.
    """

    return _NonePromotionStrategy()


def promote_percentile(percentile: float) -> TemporaryPromotionStrategy:
    """
    Of all temporaries aspiring to become Global, promote those whose complexities fall above the given percentile.
    """

    return _PercentilePromotionStrategy(percentile)


def promote_threshold(threshold: int) -> TemporaryPromotionStrategy:
    """
    Allow all temporaries whose complexity falls at or above the given threshold to be promoted as far as Global.
    """

    return _ThresholdPromotionStrategy(threshold)


def promote_rank(max_promotions: int) -> TemporaryPromotionStrategy:
    """
    Of all temporaries aspiring to become Global, promote at most the given number with the highest complexities.
    """

    return _RankPromotionStrategy(max_promotions)


#endregion
