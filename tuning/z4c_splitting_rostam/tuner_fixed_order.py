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

"""Ordinary split search, run on a base equation order derived beforehand.

The order comes from ordertest/derived_order.json, produced by running the RHS
unsplit under bare prioritize_rare_symbols and ranking the 13 author-level
add_eqn calls by where their scalar equations landed. It is pinned here, not
searched: the point is to spend the whole budget on splits while starting from
the order the sorter prefers, rather than asking the optimizer to find an order
and a split pattern at once from 39 knobs.

The split half is CombinatorialSplitTuner unchanged, so this is directly
comparable with split_tuning_checkpt.jsonl (best 5.332 s), which searched the
identical split space in recipe order. The base order is the only difference.
"""

import json
import pathlib
from typing import Any, Callable

from EinsteinEngine.intermediate.soft_split_retainment_predicate import (
    SoftSplitRetainmentStrategy, retain_percentile)
from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.tuning.tuning import Tuner

N_VARS = 13
ORDER_FILE = pathlib.Path(__file__).parent / "ordertest" / "derived_order.json"


def _load_rank() -> dict[int, float]:
    if not ORDER_FILE.exists():
        raise RuntimeError(
            f"{ORDER_FILE} not found. Run ordertest/run_derive.sh first; this "
            "tuner pins the order it produces rather than searching one.")
    data = json.loads(ORDER_FILE.read_text())
    rank = {int(k): float(v) for k, v in data["rank"].items()}
    missing = set(range(1, N_VARS + 1)) - set(rank)
    if missing:
        raise RuntimeError(f"derived order is missing add_eqn positions {sorted(missing)}")
    print(f"[tuner_fixed_order] base order from {data.get('derived_from', '?')}: "
          f"{data.get('order')}")
    return rank


class FixedOrderSplitTuner(Tuner):
    def __init__(self) -> None:
        self.rank = _load_rank()

    def get_experiment(self) -> Experiment:
        e = Experiment()

        for n in range(1, N_VARS + 1):
            e.add_in_param(f'split_{n}', (0, 2))
            e.add_in_param(f'soft_retain_percentile_{n}', (0.0, 1.0),
                           lambda params, n=n: params[f'split_{n}'] == 1)

        def get_hard_split_predicate(params: dict[str, Any]) -> Callable[[int], bool]:
            return lambda i: int(params[f'split_{i}']) == 2

        def get_soft_split_predicate(params: dict[str, Any]) -> Callable[[int], bool | SoftSplitRetainmentStrategy]:
            def f(i: int) -> bool | SoftSplitRetainmentStrategy:
                if int(params[f'split_{i}']) != 1:
                    return False
                return retain_percentile(params[f'soft_retain_percentile_{i}'])
            return f

        rank = self.rank

        def get_order_key(_params: dict[str, Any]) -> Callable[[int], float]:
            #  Constant for the whole search: the order is an input, not a knob.
            return lambda i: rank[i]

        e.add_out_param('auto_hard_split_predicate', get_hard_split_predicate)
        e.add_out_param('auto_soft_split_predicate', get_soft_split_predicate)
        e.add_out_param('auto_order_key', get_order_key)

        return e


def get_tuner() -> Tuner:
    return FixedOrderSplitTuner()
