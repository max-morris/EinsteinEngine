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
Tests for the CSE optimizations (EinsteinEngine.common.cse_optimizations).
Run as a plain script.
"""

from sympy import Rational, cse as sympy_cse

from EinsteinEngine.common.cse_optimizations import cse_basic
from EinsteinEngine.common.sympywrap import cse, mk_symbol, sqrt

a, b, c, x = (mk_symbol(n) for n in "abcx")


def test_cse_basic_matches_sympy() -> None:
    # cse_basic() folds sympy's two "basic" pairs into one, which must not change what CSE produces.
    for exprs in [[(a - b) * c + (a - b) ** 2, c * (a - b)],
                  [x * a - x * b + (a - b) / c, sqrt(a - b) * (b - a)],
                  [a - b, -(a - b) * Rational(3, 2) + c * (b - a)]]:
        assert cse(exprs, [cse_basic()]) == sympy_cse(exprs, optimizations="basic"), exprs  # type: ignore[no-untyped-call]
        assert cse(exprs) == sympy_cse(exprs), exprs  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    tests = [
        test_cse_basic_matches_sympy,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print("All CSE optimization tests passed.")
