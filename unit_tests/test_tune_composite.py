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

"""Tests for CompositeTuner and DiscreteKnobTuner: merging, refusals, probing through build_experiment, and checkpoint
replay as CheckpointedOptimizer and generate_best do it."""

import json
import os
import tempfile
import textwrap
from collections.abc import Collection
from typing import Any, Callable

from EinsteinEngine.intermediate.split_locus import SplitLocus
from EinsteinEngine.tuning import probe, tuning
from EinsteinEngine.tuning.checkpointed_optimizer import FAILED_VALUE, read_checkpoint
from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.tuning.generate_best import _FixedTrial, load_best_params
from EinsteinEngine.tuning.probe import EMPTY_PROBE, ProbeResult
from EinsteinEngine.tuning.tune_composite import CompositeTuner, DiscreteKnobTuner
from EinsteinEngine.tuning.tune_cse import CseOptimizationTuner
from EinsteinEngine.tuning.tune_splitting import CombinatorialSplitTuner
from EinsteinEngine.tuning.tuning import Tuner, build_experiment, get_tuning_param, probe_sidecar_path

HARD = 'auto_hard_split_predicate'
SOFT = 'auto_soft_split_predicate'
LOCUS = 'auto_split_locus'


def expect_error[E: BaseException](kind: type[E], fn: Callable[[], Any], *fragments: str) -> E:
    try:
        fn()
    except kind as e:
        for fragment in fragments:
            assert fragment in str(e), f"expected {fragment!r} in error message {str(e)!r}"
        return e
    raise AssertionError(f"expected {kind.__name__}")


def realize(e: Experiment, in_params: dict[str, Any]) -> dict[str, Any]:
    """The recipe-facing args for a fixed set of in_params."""
    return dict(e.suggest_params(_FixedTrial(in_params)).recipe_facing_args)


class FakeProbingTuner(Tuner):
    """A tuner with one probe target, whose Experiment has one in-param per probed position."""

    def __init__(self, target: str, params: dict[str, Any] | None = None, prefix: str = '') -> None:
        self.target = target
        self.params = dict(params or {})
        self.prefix = prefix
        self.seen: list[ProbeResult] = []

    def probe_targets(self) -> Collection[str]:
        return (self.target,)

    def probe_params(self) -> dict[str, Any]:
        return dict(self.params)

    def get_experiment(self) -> Experiment:
        raise RuntimeError('probes')

    def get_probed_experiment(self, probe_result: ProbeResult) -> Experiment:
        self.seen.append(probe_result)
        e = Experiment()
        for i in range(probe_result[self.target]):
            e.add_in_param(f'{self.prefix}x_{i}', (0, 1))
        e.add_out_param(self.target, lambda params: sum(int(params.get(f'{self.prefix}x_{i}', 0))
                                                        for i in range(probe_result[self.target])))
        for name, value in self.params.items():
            e.add_out_param(name, lambda _params, value=value: value)
        return e


def test_discrete_knob() -> None:
    knob = DiscreteKnobTuner('layout', ['zeta', 'alpha', 'mu'])
    e = knob.get_experiment()
    assert list(e.in_params) == ['layout'] and list(e.out_params) == ['layout']
    #  The index follows the order given, not the sorted order.
    assert [realize(e, {'layout': i})['layout'] for i in range(3)] == ['zeta', 'alpha', 'mu']
    renamed = DiscreteKnobTuner('bounds_index', [(128, 1), (256, 2)], out_param='launch_bounds').get_experiment()
    assert list(renamed.in_params) == ['bounds_index'] and list(renamed.out_params) == ['launch_bounds']
    assert realize(renamed, {'bounds_index': 1}) == {'launch_bounds': (256, 2)}
    #  Unhashable or equal choices are fine: the domain is the index.
    assert realize(DiscreteKnobTuner('k', [[1], [1]]).get_experiment(), {'k': 1}) == {'k': [1]}
    expect_error(ValueError, lambda: DiscreteKnobTuner('k', []), "'k'", 'at least one choice')
    assert not knob.probe_targets() and not knob.probe_params()


def test_merge() -> None:
    composite = CompositeTuner(CseOptimizationTuner(), DiscreteKnobTuner('layout', ['a', 'b']),
                               DiscreteKnobTuner('bounds', [None, (128, 1)], out_param='launch_bounds'))
    assert not composite.probe_targets() and not composite.probe_params()
    e = composite.get_experiment()
    assert list(e.in_params) == ['cse_preset', 'cse_collect_max_iter', 'layout', 'bounds']
    assert list(e.out_params) == ['cse_optimizations', 'layout', 'launch_bounds']
    assert list(build_experiment(composite, None, 'unused.jsonl').in_params) == list(e.in_params)
    args = realize(e, {'cse_preset': 0, 'layout': 1, 'bounds': 1})
    assert args['layout'] == 'b' and args['launch_bounds'] == (128, 1)
    assert args['cse_optimizations'] == CseOptimizationTuner()._optimizations({'cse_preset': 0})
    #  A composite of one is its tuner.
    single = CompositeTuner(DiscreteKnobTuner('layout', ['a', 'b'])).get_experiment()
    assert list(single.in_params) == ['layout'] and realize(single, {'layout': 0}) == {'layout': 'a'}
    expect_error(ValueError, CompositeTuner, 'at least one tuner')


def test_name_collisions() -> None:
    same_in = CompositeTuner(DiscreteKnobTuner('k', [1, 2]), DiscreteKnobTuner('k', [1, 2], out_param='other'))
    expect_error(ValueError, same_in.get_experiment, "in-param 'k'", 'tuners 0 and 1')
    same_out = CompositeTuner(DiscreteKnobTuner('a', [1, 2], out_param='o'), DiscreteKnobTuner('b', [1], out_param='o'))
    expect_error(ValueError, same_out.get_experiment, "out-param 'o'", 'tuners 0 and 1')
    #  A knob may not provide an out-param that another tuner provides as a probe param, even with the same value:
    #  only a probe param provided by both is shared.
    knob_locus = CompositeTuner(FakeProbingTuner('p', {LOCUS: SplitLocus.Early}),
                                DiscreteKnobTuner('locus_index', [SplitLocus.Early], out_param=LOCUS))
    expect_error(ValueError, lambda: knob_locus.get_probed_experiment(ProbeResult({'p': 1})), f"out-param '{LOCUS}'")


def test_probe_targets_and_params() -> None:
    expect_error(ValueError, lambda: CompositeTuner(CombinatorialSplitTuner(), CombinatorialSplitTuner()),
                 f"probe target '{HARD}'", 'tuners 0 and 1', 'prefix')
    #  Different values for one probe param are refused at construction.
    expect_error(ValueError, lambda: CompositeTuner(FakeProbingTuner('p', {LOCUS: SplitLocus.Early}),
                                                    FakeProbingTuner('q', {LOCUS: SplitLocus.PrePopulation}, 'q')),
                 f"probe param '{LOCUS}'", 'tuner 0', 'tuner 1')

    def fn_a(_: Any) -> None: ...

    def fn_b(_: Any) -> None: ...

    #  Functions are compared by identity, not by description.
    expect_error(ValueError, lambda: CompositeTuner(FakeProbingTuner('p', {'ordering_fn': fn_a}),
                                                    FakeProbingTuner('q', {'ordering_fn': fn_b}, 'q')),
                 "probe param 'ordering_fn'")

    #  The same value is shared: one out-param, and each sub-tuner sees only its own counts.
    p = FakeProbingTuner('p', {LOCUS: SplitLocus.Early, 'ordering_fn': fn_a})
    q = FakeProbingTuner('q', {LOCUS: SplitLocus.Early}, 'q')
    composite = CompositeTuner(p, q, DiscreteKnobTuner('layout', ['a']))
    assert list(composite.probe_targets()) == ['p', 'q']
    assert composite.probe_params() == {LOCUS: SplitLocus.Early, 'ordering_fn': fn_a}
    expect_error(RuntimeError, composite.get_experiment, 'build_experiment')
    expect_error(RuntimeError, lambda: composite.get_probed_experiment(ProbeResult({'p': 2})), "['q']")
    e = composite.get_probed_experiment(ProbeResult({'p': 2, 'q': 1, 'unrelated': 5}))
    assert dict(p.seen[-1]) == {'p': 2} and dict(q.seen[-1]) == {'q': 1}
    assert list(e.in_params) == ['x_0', 'x_1', 'qx_0', 'layout']
    assert list(e.out_params) == ['p', LOCUS, 'ordering_fn', 'q', 'layout']
    assert realize(e, {'x_0': 1, 'x_1': 1, 'qx_0': 0, 'layout': 0}) == \
        {'p': 2, LOCUS: SplitLocus.Early, 'ordering_fn': fn_a, 'q': 0, 'layout': 'a'}
    #  A tuner without probe targets gets the empty probe.
    knob_only = FakeProbingTuner('r')
    CompositeTuner(knob_only).get_probed_experiment(ProbeResult({'r': 0}))
    assert dict(knob_only.seen[-1]) == {'r': 0}
    assert CompositeTuner(DiscreteKnobTuner('k', [1])).get_probed_experiment(EMPTY_PROBE).in_params.keys() == {'k'}


def test_build_experiment_and_replay(directory: str) -> None:
    """A split tuner, the CSE tuner and a knob in one study: probed through build_experiment with a sidecar, then
    resumed from the sidecar alone, with checkpoint entries replayed as CheckpointedOptimizer and generate_best do."""
    recipe = os.path.join(directory, 'recipe.py')
    with open(recipe, 'w') as fh:
        fh.write(textwrap.dedent(f"""\
            from EinsteinEngine.tuning.tuning import get_tuning_param
            from EinsteinEngine.tuning.probe import report_split_positions, bake_finished
            get_tuning_param('{LOCUS}', None)
            report_split_positions(get_tuning_param('{HARD}', None), 'rhs', 3)
            bake_finished()
            """))
    checkpoint = os.path.join(directory, 'composite.jsonl')
    composite = CompositeTuner(CombinatorialSplitTuner(), CseOptimizationTuner(), DiscreteKnobTuner('layout', ['a', 'b']))
    e = build_experiment(composite, recipe, checkpoint, record_probe=True)
    with open(probe_sidecar_path(checkpoint)) as fh:
        assert json.load(fh)['counts'] == {HARD: 3}
    split_names = [name for i in range(3) for name in (f'split_{i}', f'soft_retain_percentile_{i}')]
    assert list(e.in_params) == split_names + ['cse_preset', 'cse_collect_max_iter', 'layout']
    assert list(e.out_params) == [HARD, SOFT, LOCUS, 'cse_optimizations', 'layout']
    assert probe.active_params() is None and tuning._tuning_params is None
    assert get_tuning_param('anything', 'default') == 'default'

    resumed = build_experiment(composite, None, checkpoint)
    assert list(resumed.in_params) == list(e.in_params)
    best = {'split_0': 2, 'split_1': 1, 'soft_retain_percentile_1': 0.25, 'split_2': 0, 'cse_preset': 1,
            'layout': 1}
    worse = {'split_0': 0, 'split_1': 0, 'split_2': 0, 'cse_preset': 0, 'layout': 0}
    with open(checkpoint, 'w') as fh:
        for target, params in ((-2.0, worse), (-1.0, best), (FAILED_VALUE, worse)):
            fh.write(json.dumps({'target': target, 'params': params}) + '\n')
    entries = read_checkpoint(resumed, checkpoint)
    assert [target for target, _, _ in entries] == [-2.0, -1.0, FAILED_VALUE]
    best_target, best_params = load_best_params(checkpoint)
    assert best_target == -1.0 and best_params == best
    resumed.check_declared(best_params, 'best')
    args = realize(resumed, best_params)
    assert args[LOCUS] == SplitLocus.Early and args['layout'] == 'b'
    assert [args[HARD](i) for i in range(3)] == [True, False, False]
    assert args[SOFT](0) is False and args[SOFT](1) is not False and args[SOFT](2) is False
    assert args['cse_optimizations'] == CseOptimizationTuner()._optimizations({'cse_preset': 1})
    #  The same study with a sub-tuner left out does not replay its checkpoint.
    expect_error(Exception, lambda: read_checkpoint(build_experiment(
        CompositeTuner(CombinatorialSplitTuner(), CseOptimizationTuner()), None, checkpoint), checkpoint), 'layout')


if __name__ == '__main__':
    test_discrete_knob()
    test_merge()
    test_name_collisions()
    test_probe_targets_and_params()
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            test_build_experiment_and_replay(tmp)
        finally:
            os.chdir(cwd)
    print("All composite tuner tests passed.")
