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

from typing import cast

from sympy import Expr, factor_terms
from sympy.simplify.cse_opts import sub_post, sub_pre

from EinsteinEngine.common.sympywrap import CseOptimization

__all__ = ["cse_basic"]


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
