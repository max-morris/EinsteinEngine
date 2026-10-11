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

__all__ = ["cse_basic", "cse_collect_greedy", "CsePreset", "default_cse_optimizations"]


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


def cse_collect_greedy(max_iter: int = DEFAULT_MAX_ITER, after_cse: bool = False) -> CseOptimization:
    """
    Greedy subset-factoring (see `collect_greedy`), with at most ``max_iter`` improving moves per expression.

    By default it factors each expression before the CSE, which exposes sharing that syntactic CSE cannot see, e.g. a
    common factor in two of three terms. Each expression only gets cheaper, but a factored form can break up a
    subexpression that another expression shared, so the CSE as a whole can get worse.

    With ``after_cse``, it factors each replacement and reduced expression after the CSE instead. Every shared
    subexpression is a temporary by then, so factoring one expression cannot break any sharing, but it cannot expose
    any either.

    Preprocessors run in list order, so ``[cse_collect_greedy(), cse_basic()]`` factors before cse_basic()'s rewriting.
    Postprocessors run in reverse list order, so ``[cse_collect_greedy(after_cse=True), cse_basic()]`` factors after
    cse_basic() has restored the subtractions.
    """
    collect = _CollectGreedy(max_iter)
    return (None, collect) if after_cse else (collect, None)


class CsePreset(Enum):
    """
    Named lists of CSE optimizations, for the cse_optimizations bake option. Each member is a factory:
    ``CsePreset.CollectBeforeBasic(max_iter=50)`` gives its list, in which ``max_iter`` is the budget of its
    cse_collect_greedy (the presets without one ignore it). A tuner can pick a preset by an index into
    ``list(CsePreset)``, whose order does not change.

    - Off: no optimizations.
    - Basic: sympy's basic optimizations (`cse_basic`).
    - CollectBefore: greedy subset-factoring of each expression before the CSE (`cse_collect_greedy`).
    - CollectBeforeBasic: the factoring, then sympy's basic optimizations.
    - CollectAfter: greedy subset-factoring of each expression after the CSE (`cse_collect_greedy` with ``after_cse``).
    - BasicCollectAfter: sympy's basic optimizations, then the factoring after the CSE (the bake option's default,
      `default_cse_optimizations`).
    """
    Off = "off"
    Basic = "basic"
    CollectBefore = "collect-before"
    CollectBeforeBasic = "collect-before-basic"
    CollectAfter = "collect-after"
    BasicCollectAfter = "basic-collect-after"

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
            case CsePreset.CollectAfter:
                return [cse_collect_greedy(max_iter, after_cse=True)]
            case CsePreset.BasicCollectAfter:
                # Listed first, so that its postprocessor runs last, on the restored subtractions.
                return [cse_collect_greedy(max_iter, after_cse=True), cse_basic()]


def default_cse_optimizations() -> list[CseOptimization]:
    """
    The cse_optimizations bake option's default: CsePreset.BasicCollectAfter. Of the presets, it ran the unsplit Z4c RHS
    fastest, about 11% faster than no optimizations (224.1 s against 251.4 s for 50 iterations, every preset on one
    V100S in one job), and adds under a minute to a Z4c bake. A recipe that exposes cse_optimizations as a tuning knob
    should default it to this.
    """
    return CsePreset.BasicCollectAfter()
