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

"""The transcendental surcharge must apply to applied calls, not only to classes."""

import unittest

import sympy as sy

from EinsteinEngine.generators.sympy_complexity import SympyComplexityVisitor


def _visitor() -> SympyComplexityVisitor:
    return SympyComplexityVisitor(lambda _s: False)


class TestTranscendentalSurcharge(unittest.TestCase):
    def test_listed_functions_cost_fifteen_plus_arguments(self) -> None:
        v = _visitor()
        x = sy.Symbol("x")
        # A local symbol costs 1. The surcharge is 15.
        for fn in (sy.sin, sy.cos, sy.exp, sy.log):
            with self.subTest(fn=fn):
                self.assertEqual(v.complexity(fn(x)), 16)

    def test_unlisted_function_has_no_surcharge(self) -> None:
        v = _visitor()
        self.assertEqual(v.complexity(sy.tan(sy.Symbol("x"))), 1)


if __name__ == "__main__":
    unittest.main()
