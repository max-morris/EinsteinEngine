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

"""A tuner over a recipe's CSE optimizations: which CsePreset, and with which cse_collect_greedy budget.

It drives the ``cse_optimizations`` out-param, which the recipe must read through ``get_optional_tuning_param`` and pass
to the bake, e.g. ``cse_optimizations=get_optional_tuning_param('cse_optimizations', default_cse_optimizations())``.
It provides nothing else, so the recipe's other knobs, its split predicates included, must be optional too; they then
keep their defaults.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from EinsteinEngine.common.collect import DEFAULT_MAX_ITER
from EinsteinEngine.common.cse_optimizations import CsePreset
from EinsteinEngine.common.describe_param import describe_param
from EinsteinEngine.common.sympywrap import CseOptimization
from EinsteinEngine.tuning.experiment import Discrete, Experiment
from EinsteinEngine.tuning.tuning import Tuner

DEFAULT_BUDGETS = (5, 10, 25, 50, 100)
"""The cse_collect_greedy budgets (``max_iter``) a CseOptimizationTuner searches by default."""


class CseOptimizationTuner(Tuner):
    """Search the CSE optimizations of a recipe.

    Params: ``cse_preset``, an index into ``presets``, and ``cse_collect_max_iter``, one of ``budgets``, which is only
    sampled when the preset runs cse_collect_greedy (see ``CsePreset.collects``). ``presets`` defaults to
    ``list(CsePreset)``, whose order does not change; a checkpoint records indices, so resume it with the same
    ``presets``.
    """

    def __init__(self, presets: Sequence[CsePreset] = tuple(CsePreset),
                 budgets: Collection[int] = DEFAULT_BUDGETS) -> None:
        if not presets:
            raise ValueError("CseOptimizationTuner needs at least one preset")
        if len(set(presets)) != len(presets):
            raise ValueError(f"CseOptimizationTuner's presets must be distinct, got {list(presets)}")
        if not budgets or any(budget < 1 for budget in budgets):
            raise ValueError(f"CseOptimizationTuner's budgets must be positive, got {list(budgets)}")
        self.presets = list(presets)
        self.budgets = sorted(set(budgets))

    def describe_space(self) -> Mapping[str, str]:
        return {'cse_preset': describe_param(self.presets), 'cse_collect_max_iter': describe_param(self.budgets)}

    def _preset(self, params: dict[str, Any]) -> CsePreset:
        return self.presets[int(params['cse_preset'])]

    def _optimizations(self, params: dict[str, Any]) -> list[CseOptimization]:
        return self._preset(params)(max_iter=int(params.get('cse_collect_max_iter', DEFAULT_MAX_ITER)))

    def get_experiment(self) -> Experiment:
        e = Experiment()
        e.add_in_param('cse_preset', Discrete(range(len(self.presets))))
        if any(preset.collects for preset in self.presets):
            e.add_in_param('cse_collect_max_iter', Discrete(self.budgets),
                           lambda params: self._preset(params).collects)
        e.add_out_param('cse_optimizations', self._optimizations)
        return e
