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

"""
Factories for CSE optimizations (see `CseOptimization`): each returns one sympy (preprocessor, postprocessor) pair.
"""

from dataclasses import dataclass
from enum import Enum
from typing import cast

from sympy import Expr, factor_terms
from sympy.simplify.cse_opts import sub_post, sub_pre

from EinsteinEngine.common.collect import DEFAULT_MAX_ITER, collect_greedy
from EinsteinEngine.common.sympywrap import CseOptimization

__all__ = ["cse_basic", "cse_collect_greedy", "CsePreset"]


def _basic_pre(expr: Expr) -> Expr:
    return factor_terms(sub_pre(expr))  # type: ignore[no-untyped-call]


def _basic_post(expr: Expr) -> Expr:
    return cast(Expr, sub_post(expr))  # type: ignore[no-untyped-call]


def cse_basic() -> CseOptimization:
    """
    sympy's "basic" CSE optimizations as one pair. Before the CSE, subtractions are rewritten so that they match as
    subexpressions, and common factors are pulled out of sums (factor_terms); after it, the subtractions are restored.
    This is exactly what ``sympy.cse(..., optimizations="basic")`` does with its two pairs.
    """
    return _basic_pre, _basic_post


@dataclass(frozen=True)
class _CollectGreedy:
    max_iter: int

    def __call__(self, expr: Expr) -> Expr:
        return collect_greedy(expr, max_iter=self.max_iter)


def cse_collect_greedy(max_iter: int = DEFAULT_MAX_ITER) -> CseOptimization:
    """
    Greedy subset-factoring of each expression before the CSE (see `collect_greedy`), which exposes sharing that
    syntactic CSE cannot see, e.g. a common factor in two of three terms. Each expression gets at most ``max_iter``
    improving moves.

    Preprocessors run in list order, so ``[cse_collect_greedy(), cse_basic()]`` factors before cse_basic()'s rewriting,
    and ``[cse_basic(), cse_collect_greedy()]`` after it.
    """
    return _CollectGreedy(max_iter), None


class CsePreset(Enum):
    """
    Named lists of CSE optimizations, for the cse_optimizations bake option. Each member is a factory:
    ``CsePreset.CollectBeforeBasic(max_iter=50)`` gives its list, in which ``max_iter`` is the budget of its
    cse_collect_greedy (the presets without one ignore it). A tuner can pick a preset by an index into
    ``list(CsePreset)``, whose order does not change.

    - Off: no optimizations (the bake option's default).
    - Basic: sympy's basic optimizations (`cse_basic`).
    - CollectBefore: greedy subset-factoring of each expression before the CSE (`cse_collect_greedy`).
    - CollectBeforeBasic: the factoring, then sympy's basic optimizations.
    """
    Off = "off"
    Basic = "basic"
    CollectBefore = "collect-before"
    CollectBeforeBasic = "collect-before-basic"

    def __call__(self, max_iter: int = DEFAULT_MAX_ITER) -> list[CseOptimization]:
        match self:
            case CsePreset.Off:
                return []
            case CsePreset.Basic:
                return [cse_basic()]
            case CsePreset.CollectBefore:
                return [cse_collect_greedy(max_iter)]
            case CsePreset.CollectBeforeBasic:
                return [cse_collect_greedy(max_iter), cse_basic()]
