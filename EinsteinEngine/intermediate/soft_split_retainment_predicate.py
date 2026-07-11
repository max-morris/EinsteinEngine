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

from math import ceil
from typing import Callable, Protocol, Any

from sympy import Symbol  # type: ignore[import-untyped]

from EinsteinEngine.frontend.dsl.dsl_exception import DslException

SoftSplitRetainmentPredicate = Callable[[Symbol], bool]


class SoftSplitRetainmentStrategy(Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def __call__(self, complexity: dict[Symbol, int], /) -> SoftSplitRetainmentPredicate:
        ...


def retain_none() -> SoftSplitRetainmentStrategy:
    """
    Retain no symbols.
    """

    def strategy(_complexity: dict[Symbol, int]) -> SoftSplitRetainmentPredicate:
        return lambda _sym: False

    return strategy


def retain_all() -> SoftSplitRetainmentStrategy:
    """
    Retain all symbols.
    """

    def strategy(_complexity: dict[Symbol, int]) -> SoftSplitRetainmentPredicate:
        return lambda _sym: True

    return strategy


def retain_percentile(percentile: float) -> SoftSplitRetainmentStrategy:
    """
    Retain symbols whose complexity is in the configured percentile.

    Complexities are ordered ascending and the percentile selects a complexity cutoff.
    Symbols with complexity at or above that cutoff are retained.

    - percentile: value in [0.0, 1.0].
    """

    if not 0.0 <= percentile <= 1.0:
        raise DslException(f"percentile must be between 0.0 and 1.0, got {percentile}")

    def strategy(complexity: dict[Symbol, int]) -> SoftSplitRetainmentPredicate:
        if len(complexity) == 0:
            return lambda _sym: False

        sorted_complexities = sorted(complexity.values())
        threshold_idx = max(0, ceil(percentile * len(sorted_complexities)) - 1)
        threshold = sorted_complexities[threshold_idx]
        retained_symbols = {sym for sym, cx in complexity.items() if cx >= threshold}
        return lambda sym: sym in retained_symbols

    return strategy


def retain_rank(max_retained: int) -> SoftSplitRetainmentStrategy:
    """
    Retain symbols by complexity rank from the highest complexity downward.

    Rank is based on distinct complexity values. A symbol is retained when its complexity
    falls within the top `max_retained` ranks.

    - max_retained: number of highest ranks to retain; must be >= 0.
    """

    if max_retained < 0:
        raise DslException(f"max_retained must be at least 0, got {max_retained}")

    def strategy(complexity: dict[Symbol, int]) -> SoftSplitRetainmentPredicate:
        if max_retained == 0 or len(complexity) == 0:
            return lambda _sym: False

        ranked_complexities = sorted(set(complexity.values()))
        threshold = ranked_complexities[max(0, len(ranked_complexities) - max_retained)]
        retained_symbols = {sym for sym, cx in complexity.items() if cx >= threshold}
        return lambda sym: sym in retained_symbols

    return strategy


def retain_threshold(threshold: int) -> SoftSplitRetainmentStrategy:
    """
    Retain symbols whose complexity is at or above a fixed threshold.

    - threshold: minimum complexity required for retainment.
    """

    def strategy(complexity: dict[Symbol, int]) -> SoftSplitRetainmentPredicate:
        retained_symbols = {sym for sym, cx in complexity.items() if cx >= threshold}
        return lambda sym: sym in retained_symbols

    return strategy
