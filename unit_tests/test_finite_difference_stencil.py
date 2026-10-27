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
Tests finite_difference_stencil(accuracy, ndivs, offset, la): applied to x**m, the stencil's weights (times the
DDI(la)**ndivs factor, with spacing 1) must give the exact ndivs-th derivative at 0, ndivs! for m = ndivs and 0
otherwise, for every m below ndivs + order, where order is the documented order of accuracy; and, for these cases,
the first m past the order must not be exact (the documented lower bound is attained). The points used must be the
documented ones, every term must carry DDI(la)**ndivs, and an odd accuracy (even width) or an ndivs outside
1..accuracy must be refused. Before the fix, ndivs >= 2 gave f^(ndivs)/ndivs!. Run as a plain script.
"""

from math import factorial

import sympy as sy

from EinsteinEngine.frontend.definitions import DDI, finite_difference_stencil, noop, stencil
from EinsteinEngine.common.sympywrap import mk_idx

la = mk_idx("la")


def weights(expr: sy.Expr, ndivs: int) -> dict[int, sy.Rational]:
    """{offset k: weight} of the stencil; every term must carry exactly DDI(la)**ndivs."""
    expr = sy.expand(expr.replace(noop, lambda x: x))
    out: dict[int, sy.Rational] = {}
    for term in sy.Add.make_args(expr):
        assert term.as_powers_dict().get(DDI(la), 0) == ndivs, f"term {term} lacks DDI(la)**{ndivs}"
        coef, rest = term.subs(DDI(la), 1).as_coeff_Mul()
        assert rest.func == stencil, f"unexpected term {term}"
        k = int(sy.simplify(rest.args[0] / la))
        out[k] = out.get(k, 0) + coef
    return out


def order(width: int, ndivs: int, offset: int) -> int:
    """The documented order of accuracy."""
    if offset == 0 and width % 2 == 1:
        return 2 * ((width - ndivs + 1) // 2)
    return width - ndivs


def check(accuracy: int, ndivs: int, offset: int) -> None:
    w = weights(finite_difference_stencil(accuracy, ndivs, offset, la), ndivs)
    width = accuracy + 1
    # The points, with their sign: a mirrored offset would pass the monomial checks for even ndivs.
    allowed = set(range(offset - width // 2, offset + width // 2 + 1))
    assert set(w) <= allowed, f"({accuracy}, {ndivs}, {offset}) points {sorted(w)}"
    p = order(width, ndivs, offset)
    assert p == {(4, 1, 0): 4, (4, 2, 0): 4, (4, 1, 1): 4, (4, 2, 1): 3, (4, 3, 0): 2}.get((accuracy, ndivs, offset), p)
    for m in range(ndivs + p):
        got = sum(c * sy.Integer(k) ** m for k, c in w.items())
        want = factorial(ndivs) if m == ndivs else 0
        assert got == want, f"({accuracy}, {ndivs}, {offset}) on x**{m}: {got}, want {want}"
    m = ndivs + p
    assert sum(c * sy.Integer(k) ** m for k, c in w.items()) != (factorial(ndivs) if m == ndivs else 0), \
        f"({accuracy}, {ndivs}, {offset}) is exact beyond order {p}"


def main() -> None:
    cases = [(a, n, o) for a in (2, 4, 6, 8) for n in (1, 2, 3) for o in (-1, 0, 1) if n <= a]
    for c in cases:
        check(*c)
    for a, n in ((1, 1), (3, 1), (5, 2), (4, 0), (4, 5)):
        try:
            finite_difference_stencil(a, n, 0, la)
        except ValueError:
            continue
        raise AssertionError(f"({a}, {n}, 0) was accepted: an even width or ndivs outside 1..accuracy")
    print(f"finite_difference_stencil: {len(cases)} (accuracy, ndivs, offset) cases exact to their order; "
          "even widths and out-of-range ndivs refused")


if __name__ == "__main__":
    main()
