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
    global_percentile: float
    complexity_ordered_global_symbols: list[tuple[Symbol, int]]
    tile_percentile: Optional[float]
    local_percentile: float
    complexity_ordered_tile_symbols: list[tuple[Symbol, int]]
    complexity_ordered_local_symbols: list[tuple[Symbol, int]]
    _promoted_globals: set[Symbol]
    _promoted_tiles: set[Symbol]
    _promoted_locals: set[Symbol]

    def __init__(
        self,
        complexities: dict[Symbol, int],
        temp_kinds: dict[Symbol, TempKind],
        /,
        *,
        global_percentile: float,
        tile_percentile: Optional[float] = None,
        local_percentile: float = 0.0
    ) -> None:
        self.complexities = complexities
        self.temp_kinds = temp_kinds
        self.global_percentile = global_percentile
        self.tile_percentile = tile_percentile
        self.local_percentile = local_percentile
        if not 0.0 <= self.global_percentile <= 1.0:
            raise DslException(f"global_percentile must be between 0.0 and 1.0, got {global_percentile}")
        if self.tile_percentile is not None and not 0.0 <= self.tile_percentile <= 1.0:
            raise DslException(f"tile_percentile must be between 0.0 and 1.0, got {self.tile_percentile}")
        if not 0.0 <= self.local_percentile <= 1.0:
            raise DslException(f"local_percentile must be between 0.0 and 1.0, got {self.local_percentile}")
        ordered_symbols = _get_complexity_ordered_symbols(complexities)
        self.complexity_ordered_global_symbols = list(filter(lambda kv: self.temp_kinds.get(kv[0]) is TempKind.Global, ordered_symbols))
        self.complexity_ordered_tile_symbols = list(filter(lambda kv: self.temp_kinds.get(kv[0]) is TempKind.Tile, ordered_symbols))
        self.complexity_ordered_local_symbols = list(filter(lambda kv: self.temp_kinds.get(kv[0]) is TempKind.Local, ordered_symbols))
        order_index = {sym: i for i, (sym, _) in enumerate(ordered_symbols)}

        def _top_by_percentile(symbols: list[Symbol], percentile: Optional[float]) -> set[Symbol]:
            if percentile is None:
                return set()
            if len(symbols) == 0:
                return set()
            cutoff = int(percentile * len(symbols))
            if (percentile * len(symbols)) % 1 != 0:
                cutoff += 1
            return set(sorted(symbols, key=lambda s: order_index[s])[cutoff:])

        global_symbols = [sym for sym, _ in self.complexity_ordered_global_symbols]
        self._promoted_globals = _top_by_percentile(global_symbols, self.global_percentile)

        tile_candidates = [sym for sym, _ in self.complexity_ordered_tile_symbols]
        tile_candidates.extend(sym for sym in global_symbols if sym not in self._promoted_globals)
        self._promoted_tiles = _top_by_percentile(tile_candidates, self.tile_percentile)

        local_candidates = [sym for sym, _ in self.complexity_ordered_local_symbols]
        local_candidates.extend(sym for sym in tile_candidates if sym not in self._promoted_tiles)
        self._promoted_locals = _top_by_percentile(local_candidates, self.local_percentile)

    @lru_cache
    def __call__(self, temp_name: Symbol, /) -> TempKind:
        if temp_name in self._promoted_globals:
            return TempKind.Global
        if temp_name in self._promoted_tiles:
            return TempKind.Tile
        if temp_name in self._promoted_locals:
            return TempKind.Local
        return TempKind.Inline


class ThresholdPromotionPredicate:
    complexities: dict[Symbol, int]
    global_threshold: int
    local_threshold: int
    tile_threshold: Optional[int]

    def __init__(
            self,
            complexities: dict[Symbol, int],
            /, *,
            global_threshold: int,
            local_threshold: int = 0,
            tile_threshold: Optional[int] = None
    ) -> None:
        self.complexities = complexities
        self.local_threshold = local_threshold
        self.tile_threshold = tile_threshold
        self.global_threshold = global_threshold

        if self.global_threshold < 0:
            raise DslException(f"threshold must be at least 0, got {self.global_threshold}")
        if self.local_threshold < 0:
            raise DslException(f"local_threshold must be at least 0, got {self.local_threshold}")
        if self.tile_threshold is not None:
            if self.tile_threshold < 0:
                raise DslException(f"tile_threshold must be at least 0, got {self.tile_threshold}")
            if self.tile_threshold < self.local_threshold:
                raise DslException(
                    "tile_threshold must be at least local_threshold "
                    f"({self.local_threshold}), got {self.tile_threshold}"
                )
            if self.tile_threshold > self.global_threshold:
                raise DslException(
                    "tile_threshold must be at most global_threshold "
                    f"({self.global_threshold}), got {self.tile_threshold}"
                )

    def __call__(self, temp_name: Symbol, /) -> TempKind:
        if self.complexities[temp_name] >= self.global_threshold:
            return TempKind.Global
        elif self.tile_threshold is not None and self.complexities[temp_name] >= self.tile_threshold:
            return TempKind.Tile
        elif self.complexities[temp_name] >= self.local_threshold:
            return TempKind.Local
        else:
            return TempKind.Inline


class RankPromotionPredicate:
    complexities: dict[Symbol, int]
    temp_kinds: dict[Symbol, TempKind]
    max_promotions: Optional[int]
    tile_max_promotions: Optional[int]
    local_max_promotions: Optional[int]
    complexity_ordered_global_symbols: list[tuple[Symbol, int]]
    complexity_ordered_tile_symbols: list[tuple[Symbol, int]]
    complexity_ordered_local_symbols: list[tuple[Symbol, int]]
    _promoted_globals: set[Symbol]
    _promoted_tiles: set[Symbol]
    _promoted_locals: set[Symbol]

    def __init__(
        self,
        complexities: dict[Symbol, int],
        temp_kinds: dict[Symbol, TempKind],
        /,
        *,
        max_promotions: Optional[int],
        tile_max_promotions: Optional[int] = None,
        local_max_promotions: Optional[int] = None
    ):
        self.complexities = complexities
        self.temp_kinds = temp_kinds
        self.max_promotions = max_promotions
        self.tile_max_promotions = tile_max_promotions
        self.local_max_promotions = local_max_promotions
        if self.max_promotions is not None and self.max_promotions < 1:
            raise DslException(f"max_promotions must be at least 1, got {max_promotions}")
        if self.tile_max_promotions is not None and self.tile_max_promotions < 1:
            raise DslException(f"tile_max_promotions must be at least 1, got {self.tile_max_promotions}")
        if self.local_max_promotions is not None and self.local_max_promotions < 1:
            raise DslException(f"local_max_promotions must be at least 1, got {self.local_max_promotions}")
        ordered_symbols = _get_complexity_ordered_symbols(complexities)
        self.complexity_ordered_global_symbols = list(filter(lambda kv: self.temp_kinds.get(kv[0]) is TempKind.Global, ordered_symbols))
        self.complexity_ordered_tile_symbols = list(filter(lambda kv: self.temp_kinds.get(kv[0]) is TempKind.Tile, ordered_symbols))
        self.complexity_ordered_local_symbols = list(filter(lambda kv: self.temp_kinds.get(kv[0]) is TempKind.Local, ordered_symbols))
        order_index = {sym: i for i, (sym, _) in enumerate(ordered_symbols)}

        def _top_by_complexity(symbols: list[Symbol], count: Optional[int]) -> set[Symbol]:
            if count is None:
                return set(symbols)
            if count <= 0:
                return set()
            return set(sorted(symbols, key=lambda s: order_index[s])[-count:])

        global_symbols = [sym for sym, _ in self.complexity_ordered_global_symbols]
        self._promoted_globals = _top_by_complexity(global_symbols, self.max_promotions)

        tile_candidates = [sym for sym, _ in self.complexity_ordered_tile_symbols]
        tile_candidates.extend(sym for sym in global_symbols if sym not in self._promoted_globals)
        self._promoted_tiles = _top_by_complexity(tile_candidates, self.tile_max_promotions)

        local_candidates = [sym for sym, _ in self.complexity_ordered_local_symbols]
        local_candidates.extend(sym for sym in tile_candidates if sym not in self._promoted_tiles)
        self._promoted_locals = _top_by_complexity(local_candidates, self.local_max_promotions)

    @lru_cache
    def __call__(self, temp_name: Symbol, /) -> TempKind:
        if temp_name in self._promoted_globals:
            return TempKind.Global
        if temp_name in self._promoted_tiles:
            return TempKind.Tile
        if temp_name in self._promoted_locals:
            return TempKind.Local
        return TempKind.Inline


class TruePromotionPredicate:
    def __init__(self, _complexities: dict[Symbol, int], /, *, highest: TempKind):
        self.highest = highest

    def __call__(self, _temp_name: Symbol, /) -> TempKind:
        return self.highest


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
    def __init__(self, highest: TempKind) -> None:
        self.highest = highest

    def __call__(self, complexities: dict[Symbol, int]) -> TruePromotionPredicate:
        return TruePromotionPredicate(complexities, highest=self.highest)


class _NonePromotionStrategy(OnePassTemporaryPromotionStrategy):
    def __call__(self, complexities: dict[Symbol, int]) -> FalsePromotionPredicate:
        return FalsePromotionPredicate(complexities)


class _PercentilePromotionStrategy(TwoPassTemporaryPromotionStrategy):
    def __init__(
        self,
        global_percentile: float,
        tile_percentile: Optional[float] = None,
        local_percentile: float = 0.0
    ) -> None:
        self.global_percentile = global_percentile
        self.tile_percentile = tile_percentile
        self.local_percentile = local_percentile

    def __call__(self, complexities: dict[Symbol, int], temp_kinds: dict[Symbol, TempKind]) -> PercentilePromotionPredicate:
        return PercentilePromotionPredicate(
            complexities,
            temp_kinds,
            global_percentile=self.global_percentile,
            tile_percentile=self.tile_percentile,
            local_percentile=self.local_percentile
        )


class _ThresholdPromotionStrategy(OnePassTemporaryPromotionStrategy):
    def __init__(self, global_threshold: int, local_threshold: int = 0, tile_threshold: Optional[int] = None) -> None:
        self.global_threshold = global_threshold
        self.local_threshold = local_threshold
        self.tile_threshold = tile_threshold

    def __call__(self, complexities: dict[Symbol, int]) -> ThresholdPromotionPredicate:
        return ThresholdPromotionPredicate(
            complexities,
            global_threshold=self.global_threshold,
            local_threshold=self.local_threshold,
            tile_threshold=self.tile_threshold
        )


class _RankPromotionStrategy(TwoPassTemporaryPromotionStrategy):
    def __init__(
        self,
        max_promotions: Optional[int],
        tile_max_promotions: Optional[int] = None,
        local_max_promotions: Optional[int] = None
    ) -> None:
        self.max_promotions = max_promotions
        self.tile_max_promotions = tile_max_promotions
        self.local_max_promotions = local_max_promotions

    def __call__(self, complexities: dict[Symbol, int], temp_kinds: dict[Symbol, TempKind]) -> RankPromotionPredicate:
        return RankPromotionPredicate(
            complexities,
            temp_kinds,
            max_promotions=self.max_promotions,
            tile_max_promotions=self.tile_max_promotions,
            local_max_promotions=self.local_max_promotions
        )

        

# endregion
# region Promotion Strategy Factories for DSL

def promote_all(highest: TempKind = TempKind.Global) -> TemporaryPromotionStrategy:
    """
    Promote every temporary to the same maximum TempKind.

    - highest: the maximum TempKind to allow for all temps (default: Global).

    Examples:
    - `promote_all()` -> every temp may become Global
    - `promote_all(TempKind.Tile)` -> cap all temps at Tile
    """

    return _AllPromotionStrategy(highest=highest)


def promote_none() -> TemporaryPromotionStrategy:
    """
    Force all temporaries to be Local.
    """

    return _NonePromotionStrategy()


def promote_percentile(
    global_percentile: float,
    tile_percentile: Optional[float] = None,
    local_percentile: float = 0.0
) -> TemporaryPromotionStrategy:
    """
    Percentile-based promotion within each TempKind candidate pool.

    For each pool (Global, Tile, Local), temps are ordered by complexity and promoted if
    their percentile rank is at or above the configured threshold. Candidates fall through
    from Global -> Tile -> Local as needed.

    - global_percentile: Global pool threshold (required).
    - tile_percentile: Tile pool threshold; if None, Tile candidates are demoted to Local.
    - local_percentile: Local pool threshold; defaults to 0.0 (keeps all Local temps as Local).

    Any Local candidates below local_percentile are demoted to Inline.

    Example:
    - promote_percentile(0.9, tile_percentile=0.8, local_percentile=0.5)
      Promotes the top ~10% Global, top ~20% Tile, and top ~50% Local candidates.
    """

    return _PercentilePromotionStrategy(global_percentile, tile_percentile, local_percentile)


def promote_threshold(global_threshold: int, local_threshold: int = 0, tile_threshold: Optional[int] = None) -> TemporaryPromotionStrategy:
    """
    Promote each temporary based on absolute complexity thresholds.

    The highest threshold met determines the target TempKind:
    Inline < Local < Tile < Global

    - global_threshold: complexity required for Global promotion.
    - tile_threshold: complexity required for Tile promotion; if None, Tile is not used.
    - local_threshold: complexity required for Local promotion; below this becomes Inline.

    Thresholds must be ordered: local_threshold <= tile_threshold <= global_threshold
    (when tile_threshold is provided).
    """

    return _ThresholdPromotionStrategy(global_threshold, local_threshold, tile_threshold)


def promote_rank(
    *,
    max_global: Optional[int] = None,
    max_tile: Optional[int] = None,
    max_local: Optional[int] = None
) -> TemporaryPromotionStrategy:
    """
    Rank-based promotion with fall-through across TempKind tiers.

    The algorithm walks in descending complexity and applies caps per tier:
    - Promote up to max_global of the temps whose current kind is Global.
    - Remaining Global temps are treated as Tile candidates.
    - Promote up to max_tile of Tile candidates (including those demoted from Global).
    - Remaining Tile candidates are treated as Local candidates.
    - Promote up to max_local of Local candidates (including those demoted from Tile).
    - Any remaining candidates become Inline.

    Defaults:
    - max_global: None (keeps all Global candidates as Global unless overridden)
    - max_tile: None (keeps all Tile candidates as Tile unless overridden)
    - max_local: None (keeps all Local candidates as Local unless overridden)
    """

    return _RankPromotionStrategy(max_global, max_tile, max_local)


#endregion
