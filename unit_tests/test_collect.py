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

from sympy import Dummy, Expr, Function, IndexedBase, cancel, expand

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


def test_iteration_budget() -> None:
    assert collect_greedy(x + y, max_iter=100) == x + y
    # A zero budget returns the input untouched.
    assert collect_greedy(x**2 + 2 * x + 1, max_iter=0) == x**2 + 2 * x + 1
    # Each iteration makes at most one move, so a larger budget is never worse than a smaller one.
    e = a * x * z + b * x * w + a * y * w + b * y * z + c * x + c * y + d * z
    costs = [op_count(collect_greedy(e, max_iter=n)) for n in range(5)]
    assert costs == sorted(costs, reverse=True), costs


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
        test_deterministic_across_hash_seeds,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print("All collect tests passed.")
