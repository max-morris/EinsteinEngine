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
Tests that EE's built-in centered first derivative D(f, la) groups every pair of mirror points with opposite weights,
w * (f(+k) - f(-k)), at every stencil width. The terms are sorted and neighbors of equal |weight| are grouped, so the
sort key must put equal |weight| next to each other. Also checks that the weights are unchanged, and the evaluation
error in double against 60-digit arithmetic on the same double inputs (it excludes the rounding of the inputs
themselves). Run as a plain script.
"""

import math
import random
from fractions import Fraction
from typing import Callable

import mpmath  # type: ignore[import-untyped]
import sympy as sy

from EinsteinEngine.frontend.definitions import DXI, DYI, DZI, D
from EinsteinEngine.frontend.dsl.finite_difference import ApplyDivN
from EinsteinEngine.frontend.dsl.indices import l0, l1, l2
from EinsteinEngine.common.sympywrap import mk_symbol

f = mk_symbol("f")
H = 1.0 / 64
WIDTHS = (3, 5, 7, 9)
AXES = ((l0, DXI), (l1, DYI), (l2, DZI))


def offset(e: sy.Basic) -> int:
    """The one nonzero offset of a stencil point (0 for f itself)."""
    if e == f:
        return 0
    assert isinstance(e, sy.Function) and getattr(e.func, 'name', None) == 'stencil', e
    nonzero = [int(a) for a in e.args[1:] if int(a) != 0]
    assert len(nonzero) == 1, e
    return nonzero[0]


def evaluate(e: sy.Basic, value: Callable[[int], float], num: Callable[[object], object], inv_h: object) -> object:
    """Evaluates the stencil expression bottom up in the number type of num, in the expression's own order."""
    if isinstance(e, (sy.Add, sy.Mul)):
        acc = None
        for a in e.args:
            v = evaluate(a, value, num, inv_h)
            acc = v if acc is None else (acc + v if isinstance(e, sy.Add) else acc * v)  # type: ignore[operator]
        return acc
    if isinstance(e, sy.Rational):
        return num(e.p) / num(e.q)  # type: ignore[operator]
    if e in (DXI, DYI, DZI):
        return inv_h
    if isinstance(e, sy.Function) and getattr(e.func, 'name', None) == 'noop':
        return evaluate(e.args[0], value, num, inv_h)
    return num(value(offset(e)))


def derivative(width: int, la: sy.Idx = l0) -> sy.Basic:
    return ApplyDivN(width, {}, {}, {}, 3).apply(D(f, la))


def fd_weights(width: int) -> list[Fraction]:
    col = ApplyDivN(width, {}, {}, {}, 3).fd_matrix.col(1)
    return [Fraction(int(sy.fraction(c)[0]), int(sy.fraction(c)[1])) for c in col]


def test_pairs_grouped() -> None:
    """Every mirror pair is one grouped difference: width // 2 noop terms, each f(+k) - f(-k) (in either order)."""
    for width in WIDTHS:
        for la, _ in AXES:
            groups = [g for g in sy.preorder_traversal(derivative(width, la))
                      if isinstance(g, sy.Function) and getattr(g.func, 'name', None) == 'noop']
            pairs = sorted(sorted(offset(t.as_coeff_Mul()[1]) for t in sy.Add.make_args(g.args[0])) for g in groups)
            expected = sorted([-k, k] for k in range(1, width // 2 + 1))
            assert pairs == expected, (width, la, pairs)


def test_weights() -> None:
    for width in WIDTHS:
        c = fd_weights(width)
        half = width // 2
        expected = {k: c[half + k] for k in range(-half, half + 1) if c[half + k] != 0}
        for la, _ in AXES:
            e = derivative(width, la)
            got: dict[int, Fraction] = {}
            for k in range(-half, half + 1):
                v = evaluate(e, lambda p: 1.0 if p == k else 0.0, Fraction, Fraction(1))
                if v != 0:
                    got[k] = v  # type: ignore[assignment]
            assert got == expected, (width, la)


def errors(e: sy.Basic, width: int, shift: float, samples: int = 2000) -> tuple[float, float]:
    """
    (rms error of e in double, rms error of sum_k w_k (f(k) - f(-k)) in double), each divided by the rms of e in
    60-digit arithmetic on the same double inputs, on smooth random data f = shift + 0.3 sin(k x + phi) at spacing H.
    (Dividing each sample by its own exact value instead lets the samples where f' passes through zero dominate.)
    """
    rng = random.Random(7)
    mpmath.mp.dps = 60
    c = [float(x) for x in fd_weights(width)]
    half = width // 2
    sq_e, sq_r, sq_x = 0.0, 0.0, 0.0
    for _ in range(samples):
        k, phi, x0 = rng.uniform(0.5, 2), rng.uniform(0, 2 * math.pi), rng.uniform(-1, 1)
        vals = {a: shift + 0.3 * math.sin(k * (x0 + a * H) + phi) for a in range(-half, half + 1)}
        exact = evaluate(e, lambda a: vals[a], lambda x: mpmath.mpf(x), mpmath.mpf(1) / mpmath.mpf(H))
        got = evaluate(e, lambda a: vals[a], float, 1.0 / H)
        ref = sum(c[half + a] * (vals[a] - vals[-a]) for a in range(1, half + 1)) * (1.0 / H)
        sq_e += float(got - exact) ** 2  # type: ignore[operator]
        sq_r += float(ref - exact) ** 2  # type: ignore[operator]
        sq_x += float(exact) ** 2  # type: ignore[arg-type]
    return math.sqrt(sq_e / sq_x), math.sqrt(sq_r / sq_x)


def test_accuracy() -> None:
    # A term added singly carries the data's level times its weight; the difference of two such terms cancels only
    # in the final sum and loses bits, which a grouped pair avoids. Grouped and reference forms differ only in the
    # order of the pair sums, so their errors agree within a small factor (1.0-1.7 measured); ungrouped pairs are
    # worse by orders of magnitude, more so the larger the shift (46x to 6.4e4x the reference with the former key).
    # Width 3 has one pair and is exact either way.
    for shift in (1.0, 1e3):
        for width in WIDTHS[1:]:
            err, ref = errors(derivative(width), width, shift)
            print(f"  shift {shift:g}, width {width}: rms error / rms value {err:.2e}, "
                  f"mirror pairs differenced first {ref:.2e}")
            assert err <= 3 * ref, (shift, width, err, ref)


if __name__ == "__main__":
    for test in [test_weights, test_pairs_grouped, test_accuracy]:
        test()
        print(f"ok: {test.__name__}")
    print("All first-derivative pairing tests passed.")
