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

from functools import reduce
from typing import Sequence

import sympy as sy
from sympy import Expr, Symbol

import numpy as np
from scipy.optimize import brentq

from EinsteinEngine.common.sympywrap import mk_symbol, sympify

cos = sy.cos
x = mk_symbol('x')

def sum_of_cosines(outer_coefficients: Sequence[float],
                   inner_coefficients: Sequence[float]) -> Expr:
    assert len(outer_coefficients) == len(inner_coefficients)

    return reduce(
        sy.Add,
        (
            outer * cos(inner * x) ** order
            for order, (inner, outer) in enumerate(zip(inner_coefficients, outer_coefficients), start=1)
        ),
        sympify(0)
    )

class OwnsZero:
    def __init__(self, fn: Expr, arg_sym: Symbol = x, epsilon: float = 1e-10, n_samples: int = 200) -> None:
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.f = sy.lambdify(arg_sym, fn, 'numpy')
        self.df = sy.lambdify(arg_sym, sy.diff(fn, arg_sym), 'numpy')

    def __call__(self, n: int) -> bool:
        f, df = self.f, self.df
        lo, hi = n - 0.5, n + 0.5
        n_is_even = (n % 2 == 0)

        xs = np.linspace(lo, hi, self.n_samples)
        fs = f(xs)

        # Check endpoint zeros (even n only)
        if n_is_even and (abs(fs[0]) < self.epsilon or abs(fs[-1]) < self.epsilon):
            return True

        # Sign changes in F -> zero in interior
        sign_changes = np.where(np.diff(np.sign(fs)))[0]
        if len(sign_changes) > 0:
            return True

        # Tangential zeros: find critical points (zeros of F'), check F there
        fps = df(xs)
        cp_indices = np.where(np.diff(np.sign(fps)))[0]
        for i in cp_indices:
            x_crit = brentq(df, xs[i], xs[i + 1])
            # Reject boundary critical points for odd n
            if not n_is_even and (abs(x_crit - lo) < 1e-12 or abs(x_crit - hi) < 1e-12):
                continue
            if abs(f(x_crit)) < self.epsilon:
                return True

        return False
