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

"""Tests for CseOptimizationTuner and CsePreset.collects."""

from EinsteinEngine.common.collect import DEFAULT_MAX_ITER
from EinsteinEngine.common.cse_optimizations import CsePreset
from EinsteinEngine.tuning.tune_cse import DEFAULT_BUDGETS, CseOptimizationTuner


class _ScriptedTrial:
    """A TrialObj that returns caller-chosen coordinates (or the low end)."""

    def __init__(self, picks: dict[str, int]) -> None:
        self._picks = picks

    def suggest_int(self, name: str, lo: int, hi: int, /) -> int:
        v = self._picks.get(name, lo)
        assert lo <= v <= hi, (name, v, lo, hi)
        return v

    def suggest_float(self, name: str, lo: float, hi: float, /) -> float:
        raise AssertionError(f"unexpected float param {name}")


def _raises_value_error(fn: object) -> bool:
    try:
        fn()  # type: ignore[operator]
    except ValueError:
        return True
    return False


def test_collects() -> None:
    collecting = {p for p in CsePreset if p.collects}
    assert collecting == {CsePreset.CollectBefore, CsePreset.CollectBeforeBasic, CsePreset.CollectAfter,
                          CsePreset.BasicCollectAfter}, collecting


def test_every_preset_and_budget() -> None:
    e = CseOptimizationTuner().get_experiment()
    for i, preset in enumerate(CsePreset):
        for j, budget in enumerate(DEFAULT_BUDGETS):
            result = e.suggest_params(_ScriptedTrial({'cse_preset': i, 'cse_collect_max_iter': j}))
            expected_params = {'cse_preset': i, 'cse_collect_max_iter': j} if preset.collects else {'cse_preset': i}
            assert dict(result.realized_params) == expected_params, (preset, dict(result.realized_params))
            assert dict(result.recipe_facing_args) == {'cse_optimizations': preset(max_iter=budget)}, preset


def test_budget_is_passed_to_collect_greedy() -> None:
    e = CseOptimizationTuner(budgets=[7, 3]).get_experiment()
    i = list(CsePreset).index(CsePreset.BasicCollectAfter)
    result = e.suggest_params(_ScriptedTrial({'cse_preset': i, 'cse_collect_max_iter': 1}))
    assert result.recipe_facing_args['cse_optimizations'] == CsePreset.BasicCollectAfter(max_iter=7)
    assert result.recipe_facing_args['cse_optimizations'] != CsePreset.BasicCollectAfter(max_iter=DEFAULT_MAX_ITER)


def test_presets_without_collecting_have_no_budget_param() -> None:
    tuner = CseOptimizationTuner(presets=[CsePreset.Basic, CsePreset.Off])
    e = tuner.get_experiment()
    assert list(e.in_params) == ['cse_preset']
    result = e.suggest_params(_ScriptedTrial({'cse_preset': 1}))
    assert dict(result.recipe_facing_args) == {'cse_optimizations': []}


def test_rejects_bad_arguments() -> None:
    assert _raises_value_error(lambda: CseOptimizationTuner(presets=[]))
    assert _raises_value_error(lambda: CseOptimizationTuner(presets=[CsePreset.Basic, CsePreset.Basic]))
    assert _raises_value_error(lambda: CseOptimizationTuner(budgets=[]))
    assert _raises_value_error(lambda: CseOptimizationTuner(budgets=[10, 0]))


if __name__ == '__main__':
    # Runnable without pytest; pytest still collects the test_* functions.
    for _name, _fn in sorted(dict(globals()).items()):
        if _name.startswith('test_') and callable(_fn):
            _fn()
            print(f'PASS {_name}')
    print('all tests passed')
