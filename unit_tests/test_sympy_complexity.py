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

"""The transcendental surcharge must apply to applied calls, not only to classes.

sqrt and cbrt are built as powers, so they are matched by exponent instead.
"""

import unittest

import sympy as sy

from EinsteinEngine.common.sympywrap import cbrt, mk_symbol, sqrt
from EinsteinEngine.generators.sympy_complexity import SympyComplexityVisitor


def _visitor() -> SympyComplexityVisitor:
    return SympyComplexityVisitor(lambda _s: False)


class TestTranscendentalSurcharge(unittest.TestCase):
    def test_listed_functions_cost_fifteen_plus_arguments(self) -> None:
        v = _visitor()
        x = mk_symbol("x")
        # A local symbol costs 1. The surcharge is 15.
        for fn in (sy.sin, sy.cos, sy.exp, sy.log):
            with self.subTest(fn=fn):
                self.assertEqual(v.complexity(fn(x)), 16)

    def test_unlisted_function_has_no_surcharge(self) -> None:
        v = _visitor()
        self.assertEqual(v.complexity(sy.tan(mk_symbol("x"))), 1)

    def test_sqrt_and_cbrt_cost_fifteen_plus_base(self) -> None:
        v = _visitor()
        x = mk_symbol("x")
        # These are x**(1/2) and x**(1/3). The exponent is not charged.
        for fn in (sqrt, cbrt):
            with self.subTest(fn=fn):
                self.assertEqual(v.complexity(fn(x)), 16)

    def test_other_fractional_powers_keep_the_general_cost(self) -> None:
        v = _visitor()
        x = mk_symbol("x")
        # These are emitted as pow(), so the exponent is still charged.
        for power in (sy.Rational(-1, 2), sy.Rational(3, 2), sy.Rational(2, 3)):
            with self.subTest(power=power):
                self.assertEqual(v.complexity(x ** power), 17)


if __name__ == "__main__":
    unittest.main()
