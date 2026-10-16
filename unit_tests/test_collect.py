#  Copyright (C) 2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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
Tests for the greedy subset-factoring search in EinsteinEngine.common.collect.
Run as a plain script.
"""

import os
import subprocess
import sys
import time

from sympy import Dummy, Expr, Function, IndexedBase, cancel, expand
from sympy.parsing.sympy_parser import parse_expr

from EinsteinEngine.common.collect import collect_greedy, op_count
from EinsteinEngine.common.sympywrap import mk_symbol

a, b, c, d = (mk_symbol(n) for n in "abcd")
w, x, y, z = (mk_symbol(n) for n in "wxyz")


def assert_exact_and_never_worse(before: Expr, after: Expr) -> None:
    assert op_count(after) <= op_count(before), (before, after)
    assert cancel(after - before) == 0, (before, after)  # type: ignore[no-untyped-call]


def test_shared_factor() -> None:
    # x97 occurs in two of the three terms, which syntactic CSE cannot exploit.
    x206, x97, x544, x88, x546, x530 = (mk_symbol(n) for n in ("x206", "x97", "x544", "x88", "x546", "x530"))
    kernel = x530 * (x206 * x97 + x544 * x88 + x546 * x97)
    assert op_count(kernel) == 6
    improved = collect_greedy(kernel)
    assert op_count(improved) == 5, improved
    assert_exact_and_never_worse(kernel, improved)


def test_exact_and_never_worse() -> None:
    assert collect_greedy(x**2 + 2 * x + 1) == (x + 1) ** 2
    for probe in [x + y, (x**2 - 1) / (x - 1), x**4 + 4 * x**3 + 6 * x**2 + 4 * x + 1,
                  a * x + a * y + b * x + b * y + c]:
        assert_exact_and_never_worse(probe, collect_greedy(probe))


def test_wrapped_grid_variables() -> None:
    # cse_isolate wraps each grid variable in an undefined function, and collect() only factors out plain Symbols, so
    # these atoms stand in as Symbols during the search.
    iso = Function("isolated_symbol_dummy")
    gx, gy = iso(x), iso(y)
    e = gx * a + gx * b + c
    out = collect_greedy(e)
    assert out == c + (a + b) * gx, out
    assert_exact_and_never_worse(e, out)
    g = IndexedBase("g")  # type: ignore[no-untyped-call]
    e = g[0] * a + g[0] * b + g[1] * c
    out = collect_greedy(e)
    assert out == c * g[1] + (a + b) * g[0], out
    # The stand-ins are named after the atoms' sorted order, not sympy's global Dummy counter.
    e = gx * a * z + gy * b * z + gx * b * w + gy * a * w + c * gx
    first = collect_greedy(e)
    _ = [Dummy() for _ in range(1000)]  # type: ignore[no-untyped-call]
    assert collect_greedy(e) == first
    assert_exact_and_never_worse(e, first)
    # Restoring the atoms must not distribute a factored-out number again, or the move would be thrown away.
    e = 3 * gx + 3 * a * b
    out = collect_greedy(e)
    assert op_count(out) == op_count(e) - 1, out
    assert_exact_and_never_worse(e, out)


def test_iteration_budget() -> None:
    assert collect_greedy(x + y, max_iter=100) == x + y
    # A zero budget returns the input untouched.
    assert collect_greedy(x**2 + 2 * x + 1, max_iter=0) == x**2 + 2 * x + 1
    # Each iteration makes at most one move, so a larger budget is never worse than a smaller one.
    e = a * x * z + b * x * w + a * y * w + b * y * z + c * x + c * y + d * z
    costs = [op_count(collect_greedy(e, max_iter=n)) for n in range(5)]
    assert costs == sorted(costs, reverse=True), costs


# A post-CSE expression from Z4c: a sum of degree 3 in 87 symbols. Factoring it took 866 s.
MANY_SYMBOLS = (
    "x196*(x829*x932 + x834*x931 - x836*x899 + x926*x927 + x928*x935 + x930*x936) + x210*(-x829*x924 + x8"
    "30*x901 + x831*x923 + x917*x918 + x919*x920 + x921*x922) + x44*(-x829*x899 + x830*x932 + x831*x931 +"
    " x925*x926 + x927*x928 + x929*x930) + x44*(x829*x901 + x834*x923 - x836*x924 + x918*x919 + x920*x933"
    " + x922*x934) + x48*x943 + x48*(x840*x923 + x920*x934 + x922*x945 - x944 + x946) + x59*x949 + x59*(x"
    "831*x932 + x840*x931 + x926*x929 + x928*x936 + x930*x950 - x951) + x677*x915 + x681*x916 + x689*x870"
    " + x69*x953 + x725*x914 + x737*x847 + x766*x858 - x796*x879 - x802*x877 - x804*x875 + x860*x897 + x8"
    "60*x906 - x860*x907 + x861*x895 + x861*x903 + x861*x905 + x863*x893 - x863*x899 + x863*x901 - x864*x"
    "911 - x867*x910 - x870*x909 + x880*x881 + x882*x883 + x884*x885 + x886*x887 + x888*x889 + x890*x891"
)


def test_many_symbols_are_not_factored() -> None:
    e = parse_expr(MANY_SYMBOLS)
    start = time.monotonic()
    out = collect_greedy(e)
    assert time.monotonic() - start < 30, "factor was tried on an expression with many symbols"
    assert_exact_and_never_worse(e, out)


def test_large_input_terminates() -> None:
    # Large inputs skip the slow whole-expression moves (sympy's heuristic polynomial GCD); the cheap moves must still
    # terminate on this within the default budget.
    big = expand((x + y + z + w) ** 6)  # type: ignore[no-untyped-call]
    assert_exact_and_never_worse(big, collect_greedy(big))


DETERMINISM_SCRIPT = """
from EinsteinEngine.common.collect import collect_greedy
from EinsteinEngine.common.sympywrap import mk_symbol
from sympy import Function
a, b, c, d, w, x, y, z = (mk_symbol(n) for n in "abcdwxyz")
iso = Function("isolated_symbol_dummy")
# Each of these has several moves of equal cost, so the result depends on the order in which candidates are tried.
for e in [a*x + a*y + b*x + b*y + c,
          a*x*z + b*x*w + a*y*w + b*y*z + c*x + c*y + d*z,
          (a*x + b*y + c*z)*(a*w + d*x) + a*b*x + c*d*y,
          iso(x)*a*z + iso(y)*b*z + iso(x)*b*w + iso(y)*a*w + c*iso(x)]:
    print(collect_greedy(e))
"""


def test_deterministic_across_hash_seeds() -> None:
    outputs = set()
    for seed in ("0", "1", "2", "3", "4", "5", "6", "7"):
        result = subprocess.run([sys.executable, "-c", DETERMINISM_SCRIPT], capture_output=True, text=True,
                                env={**os.environ, "PYTHONHASHSEED": seed})
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout)
    assert len(outputs) == 1, outputs


if __name__ == "__main__":
    tests = [
        test_shared_factor,
        test_exact_and_never_worse,
        test_wrapped_grid_variables,
        test_iteration_budget,
        test_large_input_terminates,
        test_many_symbols_are_not_factored,
        test_deterministic_across_hash_seeds,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print("All collect tests passed.")
