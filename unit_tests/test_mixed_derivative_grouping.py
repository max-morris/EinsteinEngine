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
Tests that EE's built-in mixed second derivative D(f, la, lb) is as accurate in double precision as the
one-direction-first form (nested one-dimensional differences), on smooth data at every stencil width. Pairing the
equal-weight corners of the outer product instead, f(a, b) + f(-a, -b), adds nearly equal values before the
cancellation; on smooth data that loses several bits (found by the AthenaK level-1 validation, 2026-10-20).
The generated expression is evaluated in double with its own grouping (noop) and argument order, and in 60-digit
arithmetic on the same double inputs. Also checks that the weights are unchanged and that D(f, l0, l1) and
D(f, l1, l0) are the same expression. Run as a plain script.
"""

import math
import random
from fractions import Fraction
from typing import Callable

import mpmath  # type: ignore[import-untyped]
import sympy as sy

from EinsteinEngine.frontend.definitions import DXI, DYI, DZI, D
from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.frontend.dsl.finite_difference import ApplyDivN
from EinsteinEngine.frontend.dsl.indices import l0, l1, l2
from EinsteinEngine.common.sympywrap import mk_symbol

f = mk_symbol("f")
H = 1.0 / 64


def evaluate(e: sy.Basic, value: Callable[[int, int], float], num: Callable[[object], object],
             inv_h: object, axes: tuple[int, int] = (0, 1)) -> object:
    """Evaluates the stencil expression bottom up in the number type of num, in the expression's own order; value
    takes the offsets along the two axes of the derivative (0, 1, 2 for x, y, z)."""
    if isinstance(e, sy.Add):
        acc = None
        for a in e.args:
            v = evaluate(a, value, num, inv_h, axes)
            acc = v if acc is None else acc + v  # type: ignore[operator]
        return acc
    if isinstance(e, sy.Mul):
        acc = None
        for a in e.args:
            v = evaluate(a, value, num, inv_h, axes)
            acc = v if acc is None else acc * v  # type: ignore[operator]
        return acc
    if isinstance(e, sy.Rational):
        return num(e.p) / num(e.q)  # type: ignore[operator]
    if e in (DXI, DYI, DZI):
        return inv_h
    if isinstance(e, sy.Function) and getattr(e.func, 'name', None) == 'noop':
        return evaluate(e.args[0], value, num, inv_h, axes)
    if isinstance(e, sy.Function) and getattr(e.func, 'name', None) == 'stencil':
        other = ({0, 1, 2} - set(axes)).pop()
        assert int(e.args[1 + other]) == 0, e  # no offset along the third axis
        return num(value(int(e.args[1 + axes[0]]), int(e.args[1 + axes[1]])))
    if e == f:
        return num(value(0, 0))
    raise ValueError(f"unexpected {e!r}")


def weights(e: sy.Basic, axes: tuple[int, int] = (0, 1)) -> dict[tuple[int, int], Fraction]:
    """The exact weight of each point (the expression evaluated in rational arithmetic on unit impulses)."""
    out = {}
    for p in [(a, b) for a in range(-4, 5) for b in range(-4, 5)]:
        v = evaluate(e, lambda a, b: 1.0 if (a, b) == p else 0.0, Fraction, Fraction(1), axes)
        if v != 0:
            out[p] = v  # type: ignore[assignment]
    return out


def first_derivative_weights(width: int) -> list[Fraction]:
    col = ApplyDivN(width, {}, {}, {}, 3).fd_matrix.col(1)
    return [Fraction(int(sy.fraction(c)[0]), int(sy.fraction(c)[1])) for c in col]


def errors(e: sy.Basic, width: int, samples: int = 400, seed: int = 7) -> tuple[float, float]:
    """(rms error of e in double, rms error of the one-direction-first reference in double), each divided by the rms
    of e in 60-digit arithmetic on the same double inputs, over smooth random data f = 1 + 0.3 sin(kx x + ky y + phi)
    at spacing H. (Dividing each sample by its own exact value instead lets the samples where the derivative passes
    through zero dominate.)"""
    rng = random.Random(seed)
    mpmath.mp.dps = 60
    c = [float(x) for x in first_derivative_weights(width)]
    half = width // 2
    sq_e, sq_r, sq_x = 0.0, 0.0, 0.0
    for _ in range(samples):
        kx, ky, phi = rng.uniform(0.5, 2), rng.uniform(0.5, 2), rng.uniform(0, 2 * math.pi)
        x0, y0 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        vals = {(a, b): 1 + 0.3 * math.sin(kx * (x0 + a * H) + ky * (y0 + b * H) + phi)
                for a in range(-half, half + 1) for b in range(-half, half + 1)}
        exact = evaluate(e, lambda a, b: vals[(a, b)], lambda x: mpmath.mpf(x), mpmath.mpf(1) / mpmath.mpf(H))
        got = evaluate(e, lambda a, b: vals[(a, b)], float, 1.0 / H)
        # Reference: at each a, the 1-D derivative across b with the differences first; then the same across a.
        inner = {a: sum(c[half + b] * (vals[(a, b)] - vals[(a, -b)]) for b in range(1, half + 1))
                 for a in range(-half, half + 1)}
        ref = sum(c[half + a] * (inner[a] - inner[-a]) for a in range(1, half + 1)) * (1.0 / H) * (1.0 / H)
        sq_e += float(got - exact) ** 2  # type: ignore[operator]
        sq_r += float(ref - exact) ** 2  # type: ignore[operator]
        sq_x += float(exact) ** 2  # type: ignore[arg-type]
    return math.sqrt(sq_e / sq_x), math.sqrt(sq_r / sq_x)


def mixed(width: int, la: sy.Idx = l0, lb: sy.Idx = l1) -> sy.Basic:
    return ApplyDivN(width, {}, {}, {}, 3).apply(D(f, la, lb))


def test_accuracy() -> None:
    # The bound: the nested form and the reference differ only in the order of the final sums, so their errors agree
    # within a small factor (up to about 2.1 over seeds 7-26); the corner pairing of the former code is 400 to 1500 times
    # the reference at widths 5-9, and at width 3 the reference is exact while that pairing is not. 3x separates
    # the two with margin on both sides.
    for width in (3, 5, 7, 9):
        err, ref = errors(mixed(width), width)
        print(f"  width {width}: rms error / rms value {err:.2e}, one-direction-first {ref:.2e}")
        assert err <= 3 * ref, (width, err, ref)


def test_only_two_indices() -> None:
    """With dimensionality 4 the applier also sees D with three index arguments: it must not be read as mixed."""
    for args in ((l0, l1, l2), (l0, l1, l0)):
        try:
            ApplyDivN(5, {}, {}, {}, 4).apply(D(f, *args))
        except Exception as ex:  # the applier's own rejection: a bare Exception or a DslException
            assert type(ex) is Exception or isinstance(ex, DslException), repr(ex)
            continue
        raise AssertionError(f"D(f, {args}) was accepted")


AXIS_PAIRS = ((l0, l1, (0, 1)), (l0, l2, (0, 2)), (l1, l2, (1, 2)))


def test_symmetric_and_weights() -> None:
    """For every axis pair: D(f, a, b) == D(f, b, a), and the weights are the outer product of the 1-D ones."""
    for width in (3, 5, 7, 9):
        c = first_derivative_weights(width)
        half = width // 2
        expected = {(a, b): c[half + a] * c[half + b] for a in range(-half, half + 1) for b in range(-half, half + 1)
                    if c[half + a] * c[half + b] != 0}
        for la, lb, axes in AXIS_PAIRS:
            assert mixed(width, la, lb) == mixed(width, lb, la), (width, la, lb)
            assert weights(mixed(width, la, lb), axes) == expected, (width, la, lb)


if __name__ == "__main__":
    for test in [test_only_two_indices, test_symmetric_and_weights, test_accuracy]:
        test()
        print(f"ok: {test.__name__}")
    print("All mixed-derivative grouping tests passed.")
