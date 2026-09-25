#  Copyright (C) 2026 Steven R. Brandt and other Einstein Engine contributors.
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

"""Search the Z4c RHS splitting AND the order the equations are added in.

The splitting half is CombinatorialSplitTuner unchanged, so this is directly
comparable with split_tuning_checkpt.jsonl (best 5.332 s), which searched the
same split space with the equations in recipe order.

The new half is one float per equation. Z4c.py buffers its add_eqn calls and
replays them sorted by these keys, so the function behaves as though the
equations had been written in that order -- including where the split
predicates fire, since those run off position in the replayed sequence.

Floats rather than a permutation on purpose: the optimizer never has to sample
a valid permutation, every draw is feasible, and ties are measure-zero (and
fall back to source order anyway).

Note the two knobs interact. With no split between them, a dependency landing
after its use is repaired inside the loop, so any order is safe. Once a split
falls between them it is not repaired -- _calc_tile_temps asserts and the trial
fails, scoring -inf. Some combinations are therefore expected to fail; that is
the search learning the constraint, not a bug.
"""

from typing import Any, Callable

from EinsteinEngine.intermediate.soft_split_retainment_predicate import (
    SoftSplitRetainmentStrategy, retain_percentile)
from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.tuning.tuning import Tuner

#  Z4c.py makes exactly 13 add_eqn calls on fun_z4c_rhs.
N_VARS = 13


class SplitAndOrderTuner(Tuner):
    def get_experiment(self) -> Experiment:
        e = Experiment()

        #  --- splitting: identical to CombinatorialSplitTuner ---
        for n in range(1, N_VARS + 1):
            e.add_in_param(f'split_{n}', (0, 2))
            e.add_in_param(f'soft_retain_percentile_{n}', (0.0, 1.0),
                           lambda params, n=n: params[f'split_{n}'] == 1)

        #  --- ordering: one sort key per equation ---
        for n in range(1, N_VARS + 1):
            e.add_in_param(f'order_{n}', (0.0, 1.0))

        def get_hard_split_predicate(params: dict[str, Any]) -> Callable[[int], bool]:
            return lambda i: int(params[f'split_{i}']) == 2

        def get_soft_split_predicate(params: dict[str, Any]) -> Callable[[int], bool | SoftSplitRetainmentStrategy]:
            def f(i: int) -> bool | SoftSplitRetainmentStrategy:
                if int(params[f'split_{i}']) != 1:
                    return False
                return retain_percentile(params[f'soft_retain_percentile_{i}'])
            return f

        def get_order_key(params: dict[str, Any]) -> Callable[[int], float]:
            #  Called with the equation's original 1-based position.
            return lambda i: float(params[f'order_{i}'])

        e.add_out_param('auto_hard_split_predicate', get_hard_split_predicate)
        e.add_out_param('auto_soft_split_predicate', get_soft_split_predicate)
        e.add_out_param('auto_order_key', get_order_key)

        return e


def get_tuner() -> Tuner:
    return SplitAndOrderTuner()
