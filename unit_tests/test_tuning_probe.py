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

"""Tests for recipe probing, the probe sidecar, the split tuners and strict checkpoint replay.

Most tests run tiny fake recipes that call the probe hooks directly, so they do not depend on the engine. The
test_real_recipe_* tests probe real ThornDef recipes at each locus, and check each count against the positions the
engine actually queries when the recipe runs with a recording predicate.
"""

import functools
import json
import os
import runpy
import subprocess
import sys
import tempfile
import textwrap
import warnings
from typing import Any, Callable

from EinsteinEngine.frontend.dsl.cactus.carpetx import ExplicitSyncBatch, NewRadXBoundaryBatch
from EinsteinEngine.intermediate.eqn_ordering import add_eqn_key_order, add_eqn_order, maximize_symbol_reuse, \
    prioritize_rare_symbols
from EinsteinEngine.intermediate.split_locus import SplitLocus
from EinsteinEngine.tuning import probe, tuning
from EinsteinEngine.tuning.checkpointed_optimizer import CheckpointedOptimizer
from EinsteinEngine.tuning.experiment import Experiment, MissingParamError, UndeclaredParamError
from EinsteinEngine.tuning.checkpointed_optimizer import FAILED_VALUE
from EinsteinEngine.tuning.generate_best import _FixedTrial, load_best_params
from EinsteinEngine.tuning.remote_feedback import TrialFailed
from EinsteinEngine.tuning.probe import EMPTY_PROBE, ProbeResult, describe_param, derive_add_eqn_order, probe_recipe, \
    rank_by_post_population
from EinsteinEngine.tuning.tune_composite import CompositeTuner
from EinsteinEngine.tuning.tune_splitting import CombinatorialSplitTuner, CutPositionSplitTuner
from EinsteinEngine.tuning.tuning import ProbeRecord, Tuner, build_experiment, get_optional_tuning_param, \
    get_tuning_param, probe_sidecar_path

HARD = 'auto_hard_split_predicate'
SOFT = 'auto_soft_split_predicate'
LOCUS = 'auto_split_locus'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIFT_SCRIPT = os.path.join(REPO_ROOT, 'scripts', 'shift_checkpoint_indices.py')

#  Prepended to every fake recipe.
FAKE_RECIPE_PRELUDE = """\
from EinsteinEngine.tuning.tuning import get_tuning_param
from EinsteinEngine.tuning.probe import report_split_positions, bake_finished
"""


class PlainTuner(Tuner):
    """A tuner without probe targets, that defines only get_experiment."""

    def get_experiment(self) -> Experiment:
        e = Experiment()
        e.add_in_param('a', (0, 1))
        return e


def expect_error[E: BaseException](kind: type[E], fn: Callable[[], Any], *fragments: str) -> E:
    try:
        fn()
    except kind as e:
        for fragment in fragments:
            assert fragment in str(e), f"expected {fragment!r} in error message {str(e)!r}"
        return e
    raise AssertionError(f"expected {kind.__name__}")


class Recipes:
    def __init__(self, directory: str) -> None:
        self.directory = directory

    def write(self, name: str, body: str, prelude: str = FAKE_RECIPE_PRELUDE) -> str:
        path = os.path.join(self.directory, f'{name}.py')
        with open(path, 'w') as fh:
            fh.write(prelude + textwrap.dedent(body))
        return path

    def marker(self, name: str) -> str:
        return os.path.join(self.directory, f'{name}.marker')

    def path(self, name: str) -> str:
        return os.path.join(self.directory, name)


def realize(e: Experiment, in_params: dict[str, Any]) -> dict[str, Any]:
    """The recipe-facing args for a fixed set of in_params."""
    return e.suggest_params(_FixedTrial(in_params)).recipe_facing_args


def assert_idle() -> None:
    assert probe.active_params() is None
    assert tuning._tuning_params is None
    assert get_tuning_param('anything', 'dflt') == 'dflt'


def write_jsonl(path: str, entries: list[dict[str, Any]]) -> None:
    with open(path, 'w') as fh:
        fh.writelines(json.dumps(entry) + '\n' for entry in entries)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_probe_counts(r: Recipes) -> None:
    marker = r.marker('basic')
    path = r.write('basic', f"""\
        import sys
        assert sys.argv == [__file__], sys.argv
        assert __name__ == '__main__'
        hard = get_tuning_param('{HARD}', None)
        assert get_tuning_param('{SOFT}', 'soft default') == 'soft default'
        assert get_tuning_param('unrelated', 42) == 42
        report_split_positions(hard, 'rhs', 13)
        report_split_positions(object(), 'rhs', 99)  # not a placeholder: ignored
        assert hard(0) is False and hard(12) is False
        bake_finished()
        open({marker!r}, 'w').close()  # never reached: bake_finished stops the recipe
        """)
    saved_argv = sys.argv
    result = probe_recipe(path, [HARD])
    assert result == ProbeResult({HARD: 13}), result
    assert not os.path.exists(marker), "the recipe ran past bake_finished"
    assert sys.argv is saved_argv
    assert_idle()


def test_probe_extra_params_and_normal_completion(r: Recipes) -> None:
    path = r.write('extra', f"""\
        from EinsteinEngine.intermediate.split_locus import SplitLocus
        locus = get_tuning_param('{LOCUS}', SplitLocus.Early)
        report_split_positions(get_tuning_param('{HARD}', None), 'rhs', 7 if locus is SplitLocus.PrePopulation else 3)
        # No bake_finished: finishing normally is fine too.
        """)
    assert probe_recipe(path, [HARD]) == {HARD: 3}
    assert probe_recipe(path, [HARD], {LOCUS: SplitLocus.PrePopulation}) == {HARD: 7}
    expect_error(ValueError, lambda: probe_recipe(path, [HARD], {HARD: None}), 'both probed')
    assert probe_recipe(path, []) is EMPTY_PROBE  # nothing to probe: the recipe is not run
    assert_idle()


def test_probe_errors(r: Recipes) -> None:
    never_evaluated = r.write('never_evaluated', f"""\
        get_tuning_param('{HARD}', None)
        bake_finished()
        """)
    expect_error(RuntimeError, lambda: probe_recipe(never_evaluated, [HARD]), 'never evaluated', HARD, LOCUS)
    assert_idle()

    #  The recipe must read every provided name: the probed predicates, and the params that change the count.
    never_read = r.write('never_read', """\
        bake_finished()
        """)
    expect_error(RuntimeError, lambda: probe_recipe(never_read, [HARD]), 'never read', HARD,
                 'before the probe stopped it')
    ignores_locus = r.write('ignores_locus', f"""\
        report_split_positions(get_tuning_param('{HARD}', None), 'rhs', 3)
        bake_finished()
        """)
    expect_error(RuntimeError, lambda: probe_recipe(ignores_locus, [HARD], {LOCUS: SplitLocus.Early,
                                                                            'early_ordering_fn': None}),
                 'never read', f"['{LOCUS}', 'early_ordering_fn']", 'get_optional_tuning_param')
    assert_idle()

    conflicting = r.write('conflicting', f"""\
        hard = get_tuning_param('{HARD}', None)
        report_split_positions(hard, 'rhs_a', 13)
        report_split_positions(hard, 'rhs_b', 5)
        bake_finished()
        """)
    expect_error(RuntimeError, lambda: probe_recipe(conflicting, [HARD]), 'rhs_a (13)', 'rhs_b (5)')
    assert_idle()

    agreeing = r.write('agreeing', f"""\
        hard = get_tuning_param('{HARD}', None)
        report_split_positions(hard, 'rhs_a', 4)
        report_split_positions(hard, 'rhs_b', 4)
        bake_finished()
        """)
    assert probe_recipe(agreeing, [HARD]) == {HARD: 4}

    failing = r.write('failing', """\
        raise ValueError('boom')
        """)
    e = expect_error(RuntimeError, lambda: probe_recipe(failing, [HARD]), 'failed while being probed')
    assert isinstance(e.__cause__, ValueError)
    assert_idle()


def test_bake_finished_waits_for_every_placeholder(r: Recipes) -> None:
    marker = r.marker('two_bakes')
    path = r.write('two_bakes', f"""\
        report_split_positions(get_tuning_param('{HARD}', None), 'thorn_a_rhs', 2)
        bake_finished()  # the soft placeholder is still unreported, so this is a no-op
        open({marker!r}, 'w').close()
        report_split_positions(get_tuning_param('{SOFT}', None), 'thorn_b_rhs', 5)
        bake_finished()
        raise AssertionError('not reached')
        """)
    assert probe_recipe(path, [HARD, SOFT]) == {HARD: 2, SOFT: 5}
    assert os.path.exists(marker)


def test_probe_result() -> None:
    p = ProbeResult({'b': 2, 'a': 1})
    assert dict(p) == {'a': 1, 'b': 2} and list(p) == ['a', 'b'] and len(p) == 2
    assert ProbeResult.from_counts({'a': 1, 'b': 2}) == p
    assert p == {'a': 1, 'b': 2} and p != ProbeResult({'a': 1, 'b': 3})
    assert not hasattr(p, '__setitem__')
    expect_error(ValueError, lambda: ProbeResult.from_counts([1]), 'object')
    expect_error(ValueError, lambda: ProbeResult.from_counts({'a': -1}), 'non-negative')
    expect_error(ValueError, lambda: ProbeResult.from_counts({'a': True}), 'non-negative')

    record = ProbeRecord(p, {LOCUS: 'SplitLocus.Early'})
    assert ProbeRecord.from_json(record.to_json()) == record
    assert json.loads(record.to_json()) == {'counts': {'a': 1, 'b': 2}, 'probe_params': {LOCUS: 'SplitLocus.Early'}}
    expect_error(ValueError, lambda: ProbeRecord.from_json('{"counts": {"a": 1}}'), 'probe_params')

    assert repr(p) == "ProbeResult({'a': 1, 'b': 2})"

    #  Derived orders: written to the sidecar only when there are any, and validated on reading.
    record = ProbeRecord(ProbeResult({'a': 1}), {}, {'early_ordering_fn': (2, 0, 1)})
    assert json.loads(record.to_json()) == {'counts': {'a': 1}, 'probe_params': {},
                                            'derived_orders': {'early_ordering_fn': [2, 0, 1]}}
    assert ProbeRecord.from_json(record.to_json()) == record
    expect_error(ValueError, lambda: ProbeRecord.from_json(
        '{"counts": {}, "probe_params": {}, "derived_orders": {"early_ordering_fn": [0, "1"]}}'), 'derived_orders')
    assert record.differences(ProbeRecord(ProbeResult({'a': 1}), {}, {'early_ordering_fn': (0, 1, 2)})) == [
        "the add_eqn order derived for 'early_ordering_fn' was [2, 0, 1] and is now [0, 1, 2]"]


def _module_level_key(i: int) -> float:
    return -i


def test_describe_param() -> None:
    order = [8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11]
    assert repr(add_eqn_order(order)) == f'add_eqn_order({order!r})'
    assert describe_param(add_eqn_order(order)) == describe_param(add_eqn_order(tuple(order)))
    assert describe_param(add_eqn_key_order(_module_level_key)) == \
        f'add_eqn_key_order({__name__}._module_level_key)'
    assert describe_param(SplitLocus.PrePopulation) == 'SplitLocus.PrePopulation'
    assert describe_param(maximize_symbol_reuse) == \
        'EinsteinEngine.intermediate.eqn_ordering.maximize_symbol_reuse'
    assert describe_param(functools.partial(prioritize_rare_symbols, consider_frequency=False)) == \
        'partial(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols, consider_frequency=False)'
    assert describe_param(None) == 'None'

    class Opaque:
        pass
    assert ' at 0x' in repr(Opaque()) and ' at 0x' not in describe_param(Opaque())
    assert describe_param(Opaque()) == describe_param(Opaque())

    #  Containers element by element; sets and dict entries sorted, so that the description does not depend on the
    #  hash seed (see test_describe_param_hash_seed).
    assert describe_param([SplitLocus.Early, (1,), {'b': 2, 'a': {3, 1, 2}}, frozenset(), set()]) == \
        "[SplitLocus.Early, (1,), {'a': {1, 2, 3}, 'b': 2}, frozenset(), set()]"
    assert describe_param(functools.partial(dict, b=frozenset({'y', 'x'}), a=1)) == \
        "partial(builtins.dict, a=1, b=frozenset({'x', 'y'}))"

    #  Functions with their defaults and closure, so that different closures of one function differ.
    def make(key: Callable[[int], float], scale: int = 2) -> Callable[[int], float]:
        def scaled(i: int, offset: int = scale) -> float:
            return key(i) * offset
        return scaled
    assert describe_param(make(_module_level_key)) == \
        f'{__name__}.test_describe_param.<locals>.make.<locals>.scaled[defaults=(2,), ' \
        f'closure=({__name__}._module_level_key,)]'
    assert describe_param(make(_module_level_key)) != describe_param(make(abs))
    assert describe_param(make(_module_level_key, 3)) != describe_param(make(_module_level_key))

    #  Wrappers as what they wrap; bound methods with their instance only if it has its own repr.
    assert describe_param(functools.cache(_module_level_key)) == describe_param(_module_level_key)

    class Plain:
        def key(self, i: int) -> float:
            return i

    class Named(Plain):
        def __repr__(self) -> str:
            return 'Named()'
    assert describe_param(Plain().key) == f'{__name__}.test_describe_param.<locals>.Plain.key'
    assert describe_param(Named().key) == f'{__name__}.test_describe_param.<locals>.Plain.key of Named()'

    #  Cycles and deep nesting terminate.
    def recursive(i: int) -> int:
        return i if i <= 0 else recursive(i - 1)
    assert describe_param(recursive).endswith('closure=(<cycle>,)]'), describe_param(recursive)
    nested: list[Any] = []
    for _ in range(20):
        nested = [nested]
    assert describe_param(nested).count('[') == 9 and '...' in describe_param(nested)


#  Printed by a subprocess under several hash seeds.
HASH_SEED_SCRIPT = """\
import functools
from sympy import Symbol
from EinsteinEngine.intermediate.eqn_ordering import cartesian_product, lexicographical_order, prioritize_rare_symbols
from EinsteinEngine.tuning.probe import describe_param
names = {'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'}
symbols = frozenset(Symbol(name) for name in names)
def closure(i):
    return i in names
print(describe_param([
    names, symbols, {'s': names, 'f': symbols}, closure,
    functools.partial(prioritize_rare_symbols, extra=names, more=[symbols]),
    cartesian_product(prioritize_rare_symbols, lexicographical_order),
]))
"""


def test_describe_param_hash_seed(r: Recipes) -> None:
    script = r.path('describe_hash_seed.py')
    with open(script, 'w') as fh:
        fh.write(HASH_SEED_SCRIPT)
    outputs = set()
    for seed in ('0', '1', '2', '3', '12345'):
        result = subprocess.run([sys.executable, script], capture_output=True, text=True,
                                env={**os.environ, 'PYTHONHASHSEED': seed})
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout)
    assert len(outputs) == 1, outputs
    assert "{'alpha', 'beta', 'delta', 'epsilon', 'gamma', 'zeta'}" in outputs.pop()


def test_tuning_params() -> None:
    assert get_optional_tuning_param('x', 1) == 1
    tuning._tuning_params = {'x': 2}
    try:
        assert get_tuning_param('x', 1) == 2
        assert get_optional_tuning_param('x', 1) == 2
        assert get_optional_tuning_param('y', 3) == 3
        expect_error(RuntimeError, lambda: get_tuning_param('y', 3), 'y')
    finally:
        tuning._tuning_params = None


def test_trial_failures() -> None:
    """A trial whose configuration fails is scored as failed; any other error (rsync, ssh, Slurm) stops the tuning."""
    def remote_run(error: Exception) -> Callable[..., float]:
        def run(*_args: Any) -> float:
            raise error
        return run

    saved = getattr(tuning, 'do_remote_run')
    try:
        setattr(tuning, 'do_remote_run', remote_run(TrialFailed('bad split')))
        assert tuning.do_tuning_run(None, x=1) == -float('inf')  # type: ignore[arg-type]
        setattr(tuning, 'do_remote_run', remote_run(RuntimeError('rsync failed')))
        expect_error(RuntimeError, lambda: tuning.do_tuning_run(None, x=1), 'rsync failed')  # type: ignore[arg-type]
    finally:
        setattr(tuning, 'do_remote_run', saved)
    assert_idle()


def test_load_best_params(r: Recipes) -> None:
    """Failed runs are skipped, whether recorded as the sentinel or as a non-finite target."""
    path = r.path('best.jsonl')
    write_jsonl(path, [{'target': -5.0, 'params': {'a': 0}}, {'target': FAILED_VALUE, 'params': {'a': 1}},
                       {'target': -4.0, 'params': {'a': 2}}])
    assert load_best_params(path) == (-4.0, {'a': 2})
    with open(path, 'w') as fh:
        fh.write(json.dumps({'target': FAILED_VALUE, 'params': {'a': 1}}) + '\n' + '{"target": -Infinity, "params": {}}\n')
    expect_error(RuntimeError, lambda: load_best_params(path), 'No successful run')


def test_tuner_api() -> None:
    #  A tuner without probe targets defines only get_experiment(self); build_experiment still works.
    assert list(PlainTuner().get_probed_experiment(EMPTY_PROBE).in_params) == ['a']

    probing = CombinatorialSplitTuner()
    expect_error(RuntimeError, probing.get_experiment, 'build_experiment')
    expect_error(RuntimeError, lambda: probing.get_probed_experiment(EMPTY_PROBE), HARD)
    fixed = CombinatorialSplitTuner(n_vars=2)
    assert list(fixed.probe_targets()) == []
    assert list(fixed.get_experiment().in_params) == list(fixed.get_probed_experiment(EMPTY_PROBE).in_params) == \
        ['split_0', 'soft_retain_percentile_0', 'split_1', 'soft_retain_percentile_1']
    expect_error(RuntimeError, CutPositionSplitTuner(1, locus=SplitLocus.Early).get_experiment, 'build_experiment')


def test_sidecar(r: Recipes) -> None:
    count_file = r.path('count.txt')
    path = r.write('sidecar', f"""\
        get_tuning_param('{LOCUS}', None)
        report_split_positions(get_tuning_param('{HARD}', None), 'rhs', int(open({count_file!r}).read()))
        bake_finished()
        """)

    def set_count(n: int) -> None:
        with open(count_file, 'w') as fh:
            fh.write(str(n))

    checkpoint = r.path('checkpt.jsonl')
    sidecar = probe_sidecar_path(checkpoint)
    assert sidecar == checkpoint + '.probe.json'
    tuner = CombinatorialSplitTuner()

    def recorded() -> dict[str, Any]:
        """The sidecar, without its (random) study id, which must be there."""
        with open(sidecar) as fh:
            loaded: dict[str, Any] = json.load(fh)
            assert len(loaded.pop('study')) == 32 and loaded.pop('origin') == os.path.abspath(checkpoint)
            return loaded

    set_count(3)
    #  Without record_probe: probe, but do not write a sidecar.
    assert len(build_experiment(tuner, path, checkpoint).in_params) == 6
    assert not os.path.exists(sidecar)
    #  No sidecar and no recipe: nothing to go on.
    expect_error(RuntimeError, lambda: build_experiment(tuner, None, checkpoint), 'no probe file')

    #  record_probe (remote_tuner): probe and write the sidecar, with the counts and the probe params.
    e = build_experiment(tuner, path, checkpoint, record_probe=True)
    assert list(e.in_params)[:2] == ['split_0', 'soft_retain_percentile_0'] and len(e.in_params) == 6
    assert recorded() == {'counts': {HARD: 3}, 'probe_params': {LOCUS: 'SplitLocus.Early'},
                          'space': {'split_*': 'CombinatorialSplitTuner(positions=probed)'}, 'recipe_args': []}
    build_experiment(tuner, path, checkpoint, record_probe=True)
    #  A sidecar written before the search space was recorded is accepted (the space is not checked), and
    #  record_probe adds the space.
    with open(sidecar) as fh:
        study = json.load(fh)['study']
    old = {k: v for k, v in recorded().items() if k != 'space'}
    with open(sidecar, 'w') as fh:
        json.dump({**old, 'study': study}, fh)
    build_experiment(tuner, None, checkpoint)
    build_experiment(tuner, path, checkpoint, record_probe=True)
    assert 'space' in recorded()
    with open(sidecar) as fh:
        assert json.load(fh)['study'] == study  # kept when the sidecar is rewritten

    #  While the checkpoint is empty, a changed probe is not an error, and record_probe replaces the sidecar.
    set_count(4)
    assert len(build_experiment(tuner, path, checkpoint, record_probe=True).in_params) == 8
    assert recorded()['counts'] == {HARD: 4}
    set_count(3)
    build_experiment(tuner, path, checkpoint, record_probe=True)
    assert recorded()['counts'] == {HARD: 3}

    #  Once the checkpoint has entries, a changed probe is an error, with or without record_probe.
    write_jsonl(checkpoint, [{'target': -1.0, 'params': {'split_0': 0, 'split_1': 2, 'split_2': 0}}])
    set_count(4)
    for record_probe in (True, False):
        expect_error(RuntimeError, lambda: build_experiment(tuner, path, checkpoint, record_probe=record_probe),
                     'does not match', f"the number of split positions for '{HARD}' was 3 and is now 4",
                     'misinterpreted')
    assert recorded()['counts'] == {HARD: 3}
    #  Without a recipe, the sidecar's counts are used.
    assert len(build_experiment(tuner, None, checkpoint).in_params) == 6

    #  A changed locus or ordering function is detected from the sidecar alone.
    expect_error(RuntimeError, lambda: build_experiment(CombinatorialSplitTuner(locus=SplitLocus.PrePopulation),
                                                        None, checkpoint),
                 f"the probe param '{LOCUS}' was 'SplitLocus.Early' and is now 'SplitLocus.PrePopulation'")
    expect_error(RuntimeError, lambda: build_experiment(CombinatorialSplitTuner(early_ordering_fn=add_eqn_order([1, 0])),
                                                        None, checkpoint),
                 "the probe param 'early_ordering_fn' was absent and is now 'add_eqn_order([1, 0])'")

    #  The sidecar is written only if the checkpoint loads against the new experiment.
    os.remove(sidecar)
    set_count(3)
    write_jsonl(checkpoint, [{'target': -1.0, 'params': {'split_1': 0, 'split_2': 2, 'split_3': 0}}])  # 1-based
    expect_error(UndeclaredParamError, lambda: build_experiment(tuner, path, checkpoint, record_probe=True),
                 'split_3')
    assert not os.path.exists(sidecar)
    #  Also when a coordinate is out of range (which CheckpointedOptimizer would refuse).
    write_jsonl(checkpoint, [{'target': -1.0, 'params': {'split_0': 7, 'split_1': 0, 'split_2': 0}}])
    expect_error(ValueError, lambda: build_experiment(tuner, path, checkpoint, record_probe=True),
                 f'{checkpoint}:1', 'split_0 = 7', '[0, 2]')
    assert not os.path.exists(sidecar)
    write_jsonl(checkpoint, [{'target': -1.0, 'params': {'split_0': 0, 'split_1': 2, 'split_2': 0}}])
    build_experiment(tuner, path, checkpoint, record_probe=True)
    assert recorded()['counts'] == {HARD: 3}

    #  A sidecar recorded for other targets.
    with open(sidecar, 'w') as fh:
        fh.write(ProbeRecord(ProbeResult({SOFT: 3}), {LOCUS: 'SplitLocus.Early'}).to_json())
    expect_error(RuntimeError, lambda: build_experiment(tuner, None, checkpoint), 'records')

    #  A corrupt sidecar.
    with open(sidecar, 'w') as fh:
        fh.write('{not json')
    expect_error(RuntimeError, lambda: build_experiment(tuner, path, checkpoint), f'probe file {sidecar}', 'not valid')

    #  A tuner without probe targets needs no sidecar, but record_probe writes one (with the study id batches need).
    os.remove(sidecar)
    os.remove(checkpoint)
    assert list(build_experiment(CombinatorialSplitTuner(n_vars=2), None, checkpoint).in_params) == \
        ['split_0', 'soft_retain_percentile_0', 'split_1', 'soft_retain_percentile_1']
    assert list(build_experiment(PlainTuner(), None, checkpoint).in_params) == ['a']
    assert not os.path.exists(sidecar)
    assert list(build_experiment(PlainTuner(), None, checkpoint, record_probe=True).in_params) == ['a']
    assert recorded() == {'counts': {}, 'probe_params': {}, 'space': {}, 'recipe_args': []}


class LocusTuner(Tuner):
    """Probes at the post-population locus; its Experiment declares only the given out-params."""

    def __init__(self, *out_params: str) -> None:
        self.out_params = out_params

    def probe_targets(self) -> tuple[str, ...]:
        return (HARD,)

    def probe_params(self) -> dict[str, Any]:
        return {LOCUS: SplitLocus.PostPopulation}

    def get_experiment(self) -> Experiment:
        raise RuntimeError('probes')

    def get_probed_experiment(self, probe: ProbeResult) -> Experiment:
        e = Experiment()
        e.add_in_param('cut', (0, probe[HARD] - 1))
        values: dict[str, Any] = {HARD: lambda _i: False, LOCUS: SplitLocus.PostPopulation}
        for name in self.out_params:
            e.add_out_param(name, lambda _params, value=values[name]: value)
        return e


def test_derived_order_sidecar(r: Recipes) -> None:
    """rank_by_post_population is derived once, recorded in the sidecar, and reused: by the trials (through the
    Experiment's out-param), and by later builds of the same checkpoint, which do not derive it again."""
    path = write_real_recipe(r, 'derived_sidecar')
    checkpoint = r.path('derived.jsonl')
    sidecar = probe_sidecar_path(checkpoint)
    tuner = CombinatorialSplitTuner(early_ordering_fn=rank_by_post_population(prioritize_rare_symbols))
    _, derived = derive_add_eqn_order(path, prioritize_rare_symbols, {LOCUS: SplitLocus.Early})
    assert sorted(derived) == [0, 1, 2, 3]
    assert_idle()

    def recorded() -> dict[str, Any]:
        """The sidecar, without its (random) study id."""
        with open(sidecar) as fh:
            loaded: dict[str, Any] = json.load(fh)
            loaded.pop('study', None)
            loaded.pop('origin', None)
            return loaded

    def trial_early_ordering_fn(e: Experiment) -> str:
        return repr(realize(e, {f'split_{i}': 0 for i in range(4)})['early_ordering_fn'])

    e = build_experiment(tuner, path, checkpoint, record_probe=True)
    assert trial_early_ordering_fn(e) == f'add_eqn_order({list(derived)})'
    assert recorded() == {
        'counts': {HARD: 4},
        'probe_params': {LOCUS: 'SplitLocus.Early',
                         'early_ordering_fn': 'rank_by_post_population('
                                              'EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols)'},
        'derived_orders': {'early_ordering_fn': list(derived)},
        'space': {'split_*': 'CombinatorialSplitTuner(positions=probed)'}, 'recipe_args': []}

    #  A recorded order is reused, not derived again, with or without the recipe.
    frozen = [3, 2, 1, 0]
    edited = {**recorded(), 'derived_orders': {'early_ordering_fn': frozen}}
    with open(sidecar, 'w') as fh:
        json.dump(edited, fh)
    def derive_again(*_args: Any) -> None:
        raise AssertionError('derived again')

    saved = getattr(tuning, 'derive_add_eqn_order')
    setattr(tuning, 'derive_add_eqn_order', derive_again)
    try:
        for recipe in (path, None):
            assert trial_early_ordering_fn(build_experiment(tuner, recipe, checkpoint)) == f'add_eqn_order({frozen})'
    finally:
        setattr(tuning, 'derive_add_eqn_order', saved)

    #  A different rule describes differently: while the checkpoint is empty it is derived anew and recorded; once the
    #  checkpoint has entries, it is an error.
    other = CombinatorialSplitTuner(early_ordering_fn=rank_by_post_population(maximize_symbol_reuse))
    build_experiment(other, path, checkpoint, record_probe=True)
    assert recorded()['probe_params']['early_ordering_fn'].endswith('maximize_symbol_reuse)')
    write_jsonl(checkpoint, [{'target': -1.0, 'params': {'split_0': 0, 'split_1': 2, 'split_2': 0, 'split_3': 0}}])
    expect_error(RuntimeError, lambda: build_experiment(tuner, path, checkpoint), 'does not match', 'misinterpreted')

    #  Without a recipe, the sidecar must record the order.
    edited = {'counts': {HARD: 4}, 'probe_params': recorded()['probe_params']}
    with open(sidecar, 'w') as fh:
        json.dump(edited, fh)
    expect_error(RuntimeError, lambda: build_experiment(other, None, checkpoint), 'records no derived add_eqn order')

    #  rank_by_post_population only resolves an early ordering function.
    bad = CombinatorialSplitTuner(ordering_fn=rank_by_post_population(prioritize_rare_symbols))
    expect_error(RuntimeError, lambda: build_experiment(bad, path, r.path('bad.jsonl')), 'only be the early_ordering_fn')
    assert_idle()


def test_probed_names_must_be_out_params(r: Recipes) -> None:
    path = r.write('out_params', f"""\
        from EinsteinEngine.intermediate.split_locus import SplitLocus
        locus = get_tuning_param('{LOCUS}', SplitLocus.Early)
        report_split_positions(get_tuning_param('{HARD}', None), 'rhs', 100 if locus is SplitLocus.PostPopulation else 3)
        bake_finished()
        """)
    checkpoint = r.path('out_params.jsonl')
    #  Tuned without the locus, the trials would run at the recipe's default locus, not the probed one.
    expect_error(RuntimeError, lambda: build_experiment(LocusTuner(HARD), path, checkpoint, record_probe=True),
                 'LocusTuner', f"['{LOCUS}']", 'no out-param')
    expect_error(RuntimeError, lambda: build_experiment(LocusTuner(), path, checkpoint), f"['{HARD}', '{LOCUS}']")
    assert not os.path.exists(probe_sidecar_path(checkpoint))
    e = build_experiment(LocusTuner(HARD, LOCUS), path, checkpoint, record_probe=True)
    assert realize(e, {'cut': 99})[LOCUS] is SplitLocus.PostPopulation


def test_probe_restores_name_counters(r: Recipes) -> None:
    names_file = r.path('names.txt')
    path = r.write('name_counters', f"""\
        from EinsteinEngine.frontend.dsl.cactus.carpetx import ExplicitSyncBatch, NewRadXBoundaryBatch
        sync = ExplicitSyncBatch([], 'Evolve')
        radx = NewRadXBoundaryBatch(None, 0, 0, 0, 'Evolve')
        open({names_file!r}, 'w').write(sync.name + ' ' + radx.name)
        report_split_positions(get_tuning_param('{HARD}', None), 'rhs', 1)
        bake_finished()
        """)
    before = ExplicitSyncBatch._name_counter, NewRadXBoundaryBatch._name_counter
    assert probe_recipe(path, [HARD]) == {HARD: 1}
    with open(names_file) as fh:
        assert fh.read() == f'DummySyncFn_{before[0]} NewRadXBoundaryFn_{before[1]}'
    assert (ExplicitSyncBatch._name_counter, NewRadXBoundaryBatch._name_counter) == before
    assert ExplicitSyncBatch([], 'Evolve').name == f'DummySyncFn_{before[0]}'
    ExplicitSyncBatch._name_counter = before[0]


def test_combinatorial_tuner() -> None:
    tuner = CombinatorialSplitTuner()
    assert list(tuner.probe_targets()) == [HARD]
    assert tuner.probe_params() == {LOCUS: SplitLocus.Early}
    assert list(CombinatorialSplitTuner(n_vars=3).probe_targets()) == []

    e = tuner.get_probed_experiment(ProbeResult({HARD: 3}))
    assert list(e.in_params) == ['split_0', 'soft_retain_percentile_0', 'split_1', 'soft_retain_percentile_1',
                                 'split_2', 'soft_retain_percentile_2']
    assert list(e.out_params) == [HARD, SOFT, LOCUS]

    args = realize(e, {'split_0': 2, 'split_1': 1, 'soft_retain_percentile_1': 0.5, 'split_2': 0})
    hard, soft = args[HARD], args[SOFT]
    assert args[LOCUS] is SplitLocus.Early
    assert [hard(i) for i in range(-1, 6)] == [False, True, False, False, False, False, False]
    assert [soft(i) is False for i in range(-1, 6)] == [True, True, False, True, True, True, True]
    assert callable(soft(1))

    order = add_eqn_order([1, 0])
    with_order = CombinatorialSplitTuner(locus=SplitLocus.PrePopulation, early_ordering_fn=order,
                                         ordering_fn=maximize_symbol_reuse)
    expected = {LOCUS: SplitLocus.PrePopulation, 'early_ordering_fn': order, 'ordering_fn': maximize_symbol_reuse}
    assert with_order.probe_params() == expected
    e = with_order.get_probed_experiment(ProbeResult({HARD: 1}))
    assert list(e.out_params) == [HARD, SOFT, LOCUS, 'early_ordering_fn', 'ordering_fn']
    args = realize(e, {'split_0': 0})
    assert {k: args[k] for k in expected} == expected

    #  Too many positions for two params each: warn, pointing to CutPositionSplitTuner.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tuner.get_probed_experiment(ProbeResult({HARD: 64}))
        assert not caught
        tuner.get_probed_experiment(ProbeResult({HARD: 65}))
    assert len(caught) == 1 and 'CutPositionSplitTuner' in str(caught[0].message), caught
    assert os.path.basename(caught[0].filename) == os.path.basename(__file__), caught[0].filename  # the caller


def test_cut_position_tuner() -> None:
    tuner = CutPositionSplitTuner(3, locus=SplitLocus.PostPopulation)
    assert list(tuner.probe_targets()) == [HARD]
    e = tuner.get_probed_experiment(ProbeResult({HARD: 1000}))
    assert list(e.in_params) == ['cut_position_0', 'cut_position_1', 'cut_position_2',
                                 'cut_kind_0', 'cut_retain_percentile_0', 'cut_kind_1', 'cut_retain_percentile_1',
                                 'cut_kind_2', 'cut_retain_percentile_2']
    assert e.in_params['cut_kind_0'].resolve_domain({}).coord_distribution({})[1:] == (0, 2)

    args = realize(e, {'cut_position_0': 10, 'cut_position_1': 500, 'cut_position_2': 999,
                       'cut_kind_0': 2, 'cut_kind_1': 1, 'cut_retain_percentile_1': 0.25, 'cut_kind_2': 0})
    hard, soft = args[HARD], args[SOFT]
    assert args[LOCUS] is SplitLocus.PostPopulation
    assert [i for i in range(-5, 1100) if hard(i)] == [10]
    assert [i for i in range(-5, 1100) if soft(i) is not False] == [500]

    strict = CutPositionSplitTuner(1, locus=SplitLocus.Early, allow_unused_cuts=False)
    e = strict.get_probed_experiment(ProbeResult({HARD: 5}))
    assert e.in_params['cut_kind_0'].resolve_domain({}).coord_distribution({})[1:] == (1, 2)
    expect_error(RuntimeError, lambda: CutPositionSplitTuner(6, locus=SplitLocus.Early).get_probed_experiment(
        ProbeResult({HARD: 5})), '6 cuts', '5 positions')
    expect_error(ValueError, lambda: CutPositionSplitTuner(0, locus=SplitLocus.Early), 'n_cuts')


def test_prefix() -> None:
    """A prefix names every in-param and out-param of a split tuner, so one tuner per function fits in one study."""
    tuner = CombinatorialSplitTuner(prefix='rhs_', ordering_fn=maximize_symbol_reuse)
    assert list(tuner.probe_targets()) == ['rhs_' + HARD]
    assert tuner.probe_params() == {'rhs_' + LOCUS: SplitLocus.Early, 'rhs_ordering_fn': maximize_symbol_reuse}
    expect_error(RuntimeError, lambda: tuner.get_probed_experiment(ProbeResult({HARD: 2})), f"'rhs_{HARD}'")
    e = tuner.get_probed_experiment(ProbeResult({'rhs_' + HARD: 2}))
    assert list(e.in_params) == ['rhs_split_0', 'rhs_soft_retain_percentile_0', 'rhs_split_1',
                                 'rhs_soft_retain_percentile_1']
    assert list(e.out_params) == ['rhs_' + HARD, 'rhs_' + SOFT, 'rhs_' + LOCUS, 'rhs_ordering_fn']
    args = realize(e, {'rhs_split_0': 1, 'rhs_soft_retain_percentile_0': 0.5, 'rhs_split_1': 2})
    assert [args['rhs_' + HARD](i) for i in range(3)] == [False, True, False]
    assert [args['rhs_' + SOFT](i) is False for i in range(3)] == [False, True, True]
    assert args['rhs_' + LOCUS] is SplitLocus.Early and args['rhs_ordering_fn'] is maximize_symbol_reuse

    cuts = CutPositionSplitTuner(1, locus=SplitLocus.PostPopulation, prefix='bnd_')
    e = cuts.get_probed_experiment(ProbeResult({'bnd_' + HARD: 10}))
    assert list(e.in_params) == ['bnd_cut_position_0', 'bnd_cut_kind_0', 'bnd_cut_retain_percentile_0']
    assert list(e.out_params) == ['bnd_' + HARD, 'bnd_' + SOFT, 'bnd_' + LOCUS]
    args = realize(e, {'bnd_cut_position_0': 7, 'bnd_cut_kind_0': 2})
    assert [i for i in range(10) if args['bnd_' + HARD](i)] == [7]
    #  Without a prefix, the names are as before.
    assert list(CombinatorialSplitTuner(n_vars=1).get_experiment().in_params) == ['split_0', 'soft_retain_percentile_0']


#  Two functions, each reading its own prefixed split params and ordering functions.
TWO_FUNCTION_RECIPE = """\
from EinsteinEngine import *
from EinsteinEngine.intermediate.split_locus import SplitLocus
from EinsteinEngine.tuning.tuning import get_optional_tuning_param, get_tuning_param

gf = ThornDef("TestProbe", "PROBETWOFUNCTIONS")
v = gf.decl("v", [li])
w = gf.decl("w", [li])
a, b, c, d, e, src = (gf.decl(name, []) for name in ("a", "b", "c", "d", "e", "src"))
options = {{}}
for name in ("f1", "f2"):
    gf.create_function(name, ScheduleBin.Evolve,
                       auto_hard_split_predicate=get_tuning_param(f'{{name}}_auto_hard_split_predicate', None),
                       auto_soft_split_predicate=get_tuning_param(f'{{name}}_auto_soft_split_predicate', None),
                       auto_split_locus=get_optional_tuning_param(f'{{name}}_auto_split_locus', SplitLocus.Early))
    options[name] = {{'ordering_fn': get_optional_tuning_param(f'{{name}}_ordering_fn', maximize_symbol_reuse),
                     'early_ordering_fn': get_optional_tuning_param(f'{{name}}_early_ordering_fn', None)}}
f1, f2 = gf.functions["f1"], gf.functions["f2"]
f1.add_eqn(a, sin(src) * cos(src) + pull_out(src * src))
f1.add_eqn(v[li], [a * sin(src), a * cos(src), sin(src) * cos(src)])
f1.add_eqn(b, v[l0] + v[l1] * sin(src))
f1.add_eqn(c, b + v[l2] * sin(src) * cos(src))
f2.add_eqn(w[li], [sin(src), cos(src), sin(src) * cos(src)])
f2.add_eqn(d, w[l0] * w[l1] + cos(src))
f2.add_eqn(e, d + w[l2])
gf.bake(do_cse=True, temporary_promotion_strategy=promote_all(), functions=options)
open({marker!r}, 'w').close()
"""


def test_prefixed_study(r: Recipes) -> None:
    """One study over two functions, one prefixed split tuner each: probed, with a prefixed rank_by_post_population
    early ordering function derived for one of them, recorded in the sidecar, and replayed."""
    path = r.write('two_functions', TWO_FUNCTION_RECIPE.format(marker=r.marker('two_functions')), prelude='')
    checkpoint = r.path('two_functions.jsonl')
    study = CompositeTuner(
        CombinatorialSplitTuner(prefix='f1_', early_ordering_fn=rank_by_post_population(prioritize_rare_symbols)),
        CombinatorialSplitTuner(prefix='f2_', locus=SplitLocus.PrePopulation))
    e = build_experiment(study, path, checkpoint, record_probe=True)
    assert_idle()
    with open(probe_sidecar_path(checkpoint)) as fh:
        recorded = json.load(fh)
    #  f1 has its 4 add_eqn groups at the early locus; f2 has at least its 5 scalar equations at the pre-population one.
    assert recorded['counts']['f1_' + HARD] == 4 and recorded['counts']['f2_' + HARD] >= 5, recorded
    assert set(recorded['counts']) == {'f1_' + HARD, 'f2_' + HARD}
    assert set(recorded['probe_params']) == {'f1_' + LOCUS, 'f1_early_ordering_fn', 'f2_' + LOCUS}
    assert list(recorded['derived_orders']) == ['f1_early_ordering_fn']
    assert sorted(recorded['derived_orders']['f1_early_ordering_fn']) == [0, 1, 2, 3]
    n2 = recorded['counts']['f2_' + HARD]
    assert [name for name in e.in_params if name.startswith('f1_split_')] == [f'f1_split_{i}' for i in range(4)]
    assert [name for name in e.in_params if name.startswith('f2_split_')] == [f'f2_split_{i}' for i in range(n2)]
    args = realize(e, {**{f'f1_split_{i}': 0 for i in range(4)}, **{f'f2_split_{i}': 0 for i in range(n2)}})
    assert repr(args['f1_early_ordering_fn']) == f"add_eqn_order({recorded['derived_orders']['f1_early_ordering_fn']})"
    assert args['f2_' + LOCUS] is SplitLocus.PrePopulation
    #  Resumed from the sidecar alone.
    assert list(build_experiment(study, None, checkpoint).in_params) == list(e.in_params)

    #  Each function queries its own predicate, at as many positions as were probed for it.
    calls: dict[str, list[int]] = {'f1': [], 'f2': []}

    def recorder(name: str) -> Callable[[int], bool]:
        def hard(i: int) -> bool:
            calls[name].append(i)
            return False
        return hard
    tuning._tuning_params = {**args, 'f1_' + HARD: recorder('f1'), 'f2_' + HARD: recorder('f2')}
    try:
        runpy.run_path(path, run_name='__main__')
    finally:
        tuning._tuning_params = None
    assert calls == {'f1': list(range(4)), 'f2': list(range(n2))}, calls
    assert os.path.exists(r.marker('two_functions'))


#  Two thorns, each with one function that reads its own prefixed split params, baked one after the other.
TWO_THORN_RECIPE = """\
from EinsteinEngine import *
from EinsteinEngine.intermediate.split_locus import SplitLocus
from EinsteinEngine.tuning.tuning import get_optional_tuning_param, get_tuning_param

def thorn(thorn_name, name):
    gf = ThornDef("TestProbe", thorn_name)
    v = gf.decl("v", [li])
    a, b, c, src = (gf.decl(n, []) for n in ("a", "b", "c", "src"))
    gf.create_function(name, ScheduleBin.Evolve,
                       auto_hard_split_predicate=get_tuning_param(f'{name}_auto_hard_split_predicate', None),
                       auto_soft_split_predicate=get_tuning_param(f'{name}_auto_soft_split_predicate', None),
                       auto_split_locus=get_optional_tuning_param(f'{name}_auto_split_locus', SplitLocus.Early))
    options = {name: {'ordering_fn': get_optional_tuning_param(f'{name}_ordering_fn', maximize_symbol_reuse),
                       'early_ordering_fn': get_optional_tuning_param(f'{name}_early_ordering_fn', None)}}
    f = gf.functions[name]
    f.add_eqn(a, sin(src) * cos(src) + pull_out(src * src))
    f.add_eqn(v[li], [a * sin(src), a * cos(src), sin(src) * cos(src)])
    f.add_eqn(b, v[l0] + v[l1] * sin(src))
    f.add_eqn(c, b + v[l2] * sin(src) * cos(src))
    gf.bake(do_cse=True, temporary_promotion_strategy=promote_all(), functions=options)

thorn("PROBETWOTHORNSA", "f1")
thorn("PROBETWOTHORNSB", "f2")
"""


def test_recipe_args(r: Recipes) -> None:
    """The recipe's own arguments reach the probe, are recorded in the probe file, and must match without a recipe."""
    path = r.write('args', f"""\
        import sys
        get_tuning_param('{LOCUS}', None)
        report_split_positions(get_tuning_param('{HARD}', None), 'rhs', 7 if sys.argv[1:] == ['--wide'] else 3)
        bake_finished()
        """)
    checkpoint = r.path('args.jsonl')
    tuner = CombinatorialSplitTuner()
    assert len(build_experiment(tuner, path, checkpoint, record_probe=True, recipe_args=['--wide']).in_params) == 14
    with open(probe_sidecar_path(checkpoint)) as fh:
        assert json.load(fh)['recipe_args'] == ['--wide']
    assert len(build_experiment(tuner, None, checkpoint, recipe_args=['--wide']).in_params) == 14
    expect_error(RuntimeError, lambda: build_experiment(tuner, None, checkpoint), 'recipe arguments', "['--wide']")
    #  With a recipe, other arguments are a changed probe: an error once the checkpoint has entries.
    write_jsonl(checkpoint, [{'target': -1.0, 'params': {f'split_{i}': 0 for i in range(7)}}])
    expect_error(RuntimeError, lambda: build_experiment(tuner, path, checkpoint), 'recipe arguments were',
                 'misinterpreted')
    assert_idle()


def test_split_tuner_param_names() -> None:
    """probe._SPLIT_TUNER_PARAMS names every fixed out-param a split tuner can provide."""
    order = add_eqn_order([0])
    for tuner in (CombinatorialSplitTuner(prefix='p_', early_ordering_fn=order, pre_population_ordering_fn=order,
                                          ordering_fn=order),
                  CutPositionSplitTuner(1, locus=SplitLocus.Early, prefix='p_', early_ordering_fn=order,
                                        pre_population_ordering_fn=order, ordering_fn=order)):
        assert set(tuner.probe_params()) == {'p_' + name for name in probe._SPLIT_TUNER_PARAMS}


def test_prefixed_separate_bakes(r: Recipes) -> None:
    """Prefixed split tuners for functions baked separately: a ranked early ordering function for the function baked
    first or second, or for both, derives the same order as the same function alone without a prefix; and a prefixed
    CutPositionSplitTuner probes its own function."""
    path = r.write('two_thorns', TWO_THORN_RECIPE, prelude='')
    alone = write_real_recipe(r, 'two_thorns_alone')
    _, expected = derive_add_eqn_order(alone, prioritize_rare_symbols, {LOCUS: SplitLocus.Early})
    rank = rank_by_post_population(prioritize_rare_symbols)
    for n, ranked in enumerate((['f1'], ['f2'], ['f1', 'f2'])):
        study = CompositeTuner(*(CombinatorialSplitTuner(prefix=f'{f}_', early_ordering_fn=rank if f in ranked else None)
                                 for f in ('f1', 'f2')))
        checkpoint = r.path(f'two_thorns_{n}.jsonl')
        e = build_experiment(study, path, checkpoint, record_probe=True)
        assert_idle()
        with open(probe_sidecar_path(checkpoint)) as fh:
            recorded = json.load(fh)
        assert recorded['counts'] == {f'f1_{HARD}': 4, f'f2_{HARD}': 4}
        assert recorded['derived_orders'] == {f'{f}_early_ordering_fn': list(expected) for f in ranked}
        assert len(e.in_params) == 16
    cuts = CompositeTuner(CombinatorialSplitTuner(prefix='f1_'),
                          CutPositionSplitTuner(1, locus=SplitLocus.Early, prefix='f2_'))
    e = build_experiment(cuts, path, r.path('two_thorns_cuts.jsonl'), record_probe=True)
    assert [n for n in e.in_params if n.startswith('f2_')] == ['f2_cut_position_0', 'f2_cut_kind_0',
                                                                'f2_cut_retain_percentile_0']
    assert e.in_params['f2_cut_position_0'].resolve_domain({}).coord_distribution({})[1:] == (0, 3)
    #  A prefix must be an identifier ending in '_'; the split-param hints recognize prefixed names.
    for bad in ('rhs', '1_', 'a-b_'):
        expect_error(ValueError, lambda: CombinatorialSplitTuner(prefix=bad), "ending in '_'")
    e = CombinatorialSplitTuner(n_vars=1, prefix='rhs_').get_experiment()
    expect_error(UndeclaredParamError, lambda: e.check_declared({'rhs_split_0': 0, 'rhs_split_1': 0}, 'x'),
                 'For prefixed split params', 'more split positions')


def test_strict_checkpoint(r: Recipes) -> None:
    e = CombinatorialSplitTuner(n_vars=2).get_experiment()
    good = {'split_0': 1, 'soft_retain_percentile_0': 0.5, 'split_1': 2}
    one_based = {'split_1': 1, 'soft_retain_percentile_1': 0.5, 'split_2': 2}

    checkpoint = r.path('strict.jsonl')
    write_jsonl(checkpoint, [{'target': -1.0, 'params': good}])
    assert CheckpointedOptimizer(f=None, experiment=e, checkpoint_file=checkpoint).n_checkpoint_loaded == 1

    with open(checkpoint, 'a') as fh:
        fh.write(json.dumps({'target': -2.0, 'params': one_based}) + '\n')
    err = expect_error(UndeclaredParamError, lambda: CheckpointedOptimizer(f=None, experiment=e,
                                                                           checkpoint_file=checkpoint),
                       f'{checkpoint}:2', 'split_2', 'scripts/shift_checkpoint_indices.py', '1-based',
                       '--n-positions')
    assert err.names == ['split_2']
    e.check_declared(good, 'good')

    #  The split hint is only given for split params.
    other = Experiment()
    other.add_in_param('a', (0, 1))
    err = expect_error(UndeclaredParamError, lambda: other.check_declared({'a': 0, 'b': 1}, 'entry'), 'b')
    assert 'shift_checkpoint_indices' not in str(err)

    #  A param whose condition holds must be present; one whose condition fails may be absent, or present (and
    #  ignored). The hint depends on the kind of param.
    expect_error(MissingParamError, lambda: e.check_declared({'split_0': 1, 'split_1': 0}, 'entry'),
                 'entry', 'soft_retain_percentile_0', 'predates')
    expect_error(MissingParamError, lambda: e.check_declared({'split_0': 0}, 'entry'), 'split_1',
                 'fewer split positions')
    missing = expect_error(MissingParamError, lambda: other.check_declared({}, 'entry'), 'a')
    assert 'split' not in str(missing) and missing.names == ['a'], missing
    e.check_declared({'split_0': 0, 'split_1': 0}, 'entry')
    e.check_declared({'split_0': 0, 'soft_retain_percentile_0': 0.5, 'split_1': 0}, 'entry')

    #  Each coordinate must lie in the range its domain gives it for this trial. Integral floats (as in old
    #  checkpoints) are accepted.
    e.check_declared({'split_0': 2.0, 'split_1': 0.0}, 'entry')
    expect_error(ValueError, lambda: e.check_declared({'split_0': 3, 'split_1': 0}, 'entry'),
                 'entry has split_0 = 3', 'an integer in [0, 2]')
    expect_error(ValueError, lambda: e.check_declared({'split_0': 1.5, 'split_1': 0}, 'entry'), 'split_0 = 1.5')
    expect_error(ValueError, lambda: e.check_declared({'split_0': 1, 'soft_retain_percentile_0': 1.5, 'split_1': 0},
                                                      'entry'), 'a number in [0.0, 1.0]')
    cuts = CutPositionSplitTuner(2, locus=SplitLocus.PostPopulation).get_probed_experiment(ProbeResult({HARD: 5}))
    descending = {'cut_position_0': 3, 'cut_position_1': 1, 'cut_kind_0': 2, 'cut_kind_1': 2}
    expect_error(ValueError, lambda: cuts.check_declared(descending, 'entry'), 'cut_position_1 = 1', '[4, 4]')
    write_jsonl(checkpoint, [{'target': -1.0, 'params': descending}])
    expect_error(ValueError, lambda: CheckpointedOptimizer(f=None, experiment=cuts, checkpoint_file=checkpoint),
                 f'{checkpoint}:1', 'cut_position_1')
    write_jsonl(checkpoint, [{'target': -1.0, 'params': {'split_0': 0}}])
    expect_error(MissingParamError, lambda: CheckpointedOptimizer(f=None, experiment=e, checkpoint_file=checkpoint),
                 f'{checkpoint}:1', 'split_1')


def run_shift_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, SHIFT_SCRIPT, *args], capture_output=True, text=True)


def test_shift_script(r: Recipes) -> None:
    #  A 1-based checkpoint with 15 positions, for a recipe that has 13.
    old = r.path('old15.jsonl')
    params: dict[str, Any] = {'other_1': 7}
    for k in range(1, 16):
        params[f'split_{k}'] = 1
        params[f'soft_retain_percentile_{k}'] = k / 100
    write_jsonl(old, [{'target': -1.0, 'params': params}, {'target': -2.0, 'params': params}])

    e13 = CombinatorialSplitTuner(n_vars=13).get_experiment()
    e13.add_in_param('other_1', (0, 10))

    shifted = r.path('shifted15.jsonl')
    assert run_shift_script(old, shifted).returncode == 0
    [entry, _] = read_jsonl(shifted)
    assert entry['params']['split_0'] == 1 and entry['params']['soft_retain_percentile_14'] == 0.15
    assert entry['params']['other_1'] == 7  # not a split key: unchanged
    expect_error(UndeclaredParamError, lambda: e13.check_declared(entry['params'], 'entry'), 'split_13', 'split_14',
                 '--n-positions')

    trimmed = r.path('trimmed13.jsonl')
    result = run_shift_script(old, trimmed, '--n-positions', '13')
    assert result.returncode == 0, result.stderr
    assert 'shifting 60 keys and dropping 8 keys with an index >= 13, 4 of which were splits' in result.stderr, \
        result.stderr
    for entry in read_jsonl(trimmed):
        assert entry['params']['soft_retain_percentile_0'] == 0.01
        assert entry['params']['soft_retain_percentile_12'] == 0.13
        e13.check_declared(entry['params'], 'entry')

    #  Refuses an already 0-based file, an existing output, and an existing probe file for the output.
    result = run_shift_script(trimmed, r.path('twice.jsonl'))
    assert result.returncode == 2 and f'{trimmed}:1:' in result.stderr and 'already 0-based' in result.stderr, \
        result.stderr
    assert not os.path.exists(r.path('twice.jsonl'))
    result = run_shift_script(old, trimmed)
    assert result.returncode == 2 and 'already exists' in result.stderr, result.stderr
    fresh = r.path('fresh.jsonl')
    with open(probe_sidecar_path(fresh), 'w') as fh:
        fh.write('{}')
    result = run_shift_script(old, fresh)
    assert result.returncode == 2 and 'probe.json already exists' in result.stderr and not os.path.exists(fresh)

    #  Warns when there is nothing to shift, and the dropped splits count only nonzero split keys.
    no_splits = r.path('no_splits.jsonl')
    write_jsonl(no_splits, [{'target': -1.0, 'params': {'a': 1}}])
    result = run_shift_script(no_splits, r.path('no_splits.0based.jsonl'))
    assert result.returncode == 0 and 'Warning' in result.stderr and 'shifting 0 keys' in result.stderr, result.stderr
    zeros = r.path('zeros.jsonl')
    write_jsonl(zeros, [{'target': -1.0, 'params': {'split_1': 1, 'split_2': 0, 'split_3': 2}}])
    result = run_shift_script(zeros, r.path('zeros.0based.jsonl'), '--n-positions', '1')
    assert 'dropping 2 keys with an index >= 1, 1 of which were splits' in result.stderr, result.stderr
    assert 'Warning' not in result.stderr


#  A real recipe. The vector add_eqn is one early element but three scalar equations, the pull_out adds a temporary
#  before population, and global CSE adds more after it, so the three loci have different counts.
REAL_RECIPE = """\
from EinsteinEngine import *
from EinsteinEngine.intermediate.split_locus import SplitLocus
from EinsteinEngine.tuning.tuning import get_optional_tuning_param, get_tuning_param

gf = ThornDef("TestProbe", {thorn!r})
v = gf.decl("v", [li])
a, b, c, src = gf.decl("a", []), gf.decl("b", []), gf.decl("c", []), gf.decl("src", [])
fn = gf.create_function("probe_fn", ScheduleBin.Evolve,
                        auto_hard_split_predicate=get_tuning_param('auto_hard_split_predicate', None),
                        auto_soft_split_predicate=get_tuning_param('auto_soft_split_predicate', None),
                        {locus_kwarg})
fn.add_eqn(a, sin(src) * cos(src) + pull_out(src * src))
fn.add_eqn(v[li], [a * sin(src), a * cos(src), sin(src) * cos(src)])
fn.add_eqn(b, v[l0] + v[l1] * sin(src))
fn.add_eqn(c, b + v[l2] * sin(src) * cos(src))
gf.bake(do_cse=True, temporary_promotion_strategy=promote_all(),
        ordering_fn=get_optional_tuning_param('ordering_fn', maximize_symbol_reuse),
        early_ordering_fn=get_optional_tuning_param('early_ordering_fn', None))
open({marker!r}, 'w').close()  # not reached while probing: the probe stops the recipe at the end of bake
"""

READS_LOCUS = "auto_split_locus=get_optional_tuning_param('auto_split_locus', SplitLocus.Early)"


def write_real_recipe(r: Recipes, name: str, locus_kwarg: str = READS_LOCUS) -> str:
    return r.write(name, REAL_RECIPE.format(thorn=f'PROBE{name.upper()}', locus_kwarg=locus_kwarg,
                                            marker=r.marker(name)), prelude='')


def queried_positions(path: str, fixed: dict[str, Any]) -> list[int]:
    """Run the recipe with a hard predicate that records the positions it is queried at, and never splits."""
    calls: list[int] = []

    def hard(i: int) -> bool:
        calls.append(i)
        return False
    tuning._tuning_params = {HARD: hard, SOFT: None, **fixed}
    try:
        runpy.run_path(path, run_name='__main__')
    finally:
        tuning._tuning_params = None
    return calls


def test_real_recipe_loci(r: Recipes) -> None:
    counts: dict[SplitLocus, int] = {}
    for locus in SplitLocus:
        variants: list[dict[str, Any]] = [{LOCUS: locus}, {LOCUS: locus, 'early_ordering_fn': add_eqn_order([3, 2, 1, 0])}]
        for fixed in variants:
            name = f'real_{locus.name.lower()}_{len(fixed)}'
            path = write_real_recipe(r, name)
            tuner = CombinatorialSplitTuner(locus=locus, early_ordering_fn=fixed.get('early_ordering_fn'))
            assert tuner.probe_params() == fixed
            result = probe_recipe(path, [HARD, SOFT], tuner.probe_params())
            assert not os.path.exists(r.marker(name)), "the probe did not stop the recipe"
            assert_idle()

            calls = queried_positions(path, fixed)
            assert os.path.exists(r.marker(name))
            assert calls == list(range(len(calls))), calls
            assert result == {HARD: len(calls), SOFT: len(calls)}, (locus, result, len(calls))
            counts.setdefault(locus, len(calls))
            assert counts[locus] == len(calls), "reordering the groups changed the number of positions"

    #  4 add_eqn calls at the early locus; 6 scalar equations plus the pull-out temporary at the pre-population locus;
    #  more after global CSE.
    assert counts[SplitLocus.Early] == 4, counts
    assert counts[SplitLocus.PrePopulation] == 7, counts
    assert counts[SplitLocus.PostPopulation] > 7, counts


def test_real_recipe_must_read_locus(r: Recipes) -> None:
    #  A recipe that ignores auto_split_locus would be probed (and tuned) at the wrong locus.
    path = write_real_recipe(r, 'real_no_locus', locus_kwarg='')
    expect_error(RuntimeError, lambda: probe_recipe(path, [HARD], CombinatorialSplitTuner().probe_params()),
                 'never read', LOCUS)
    assert_idle()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        recipes = Recipes(tmp)
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            test_probe_counts(recipes)
            test_probe_extra_params_and_normal_completion(recipes)
            test_probe_errors(recipes)
            test_bake_finished_waits_for_every_placeholder(recipes)
            test_probe_result()
            test_describe_param()
            test_describe_param_hash_seed(recipes)
            test_tuning_params()
            test_trial_failures()
            test_load_best_params(recipes)
            test_tuner_api()
            test_sidecar(recipes)
            test_derived_order_sidecar(recipes)
            test_probed_names_must_be_out_params(recipes)
            test_probe_restores_name_counters(recipes)
            test_combinatorial_tuner()
            test_cut_position_tuner()
            test_prefix()
            test_prefixed_study(recipes)
            test_prefixed_separate_bakes(recipes)
            test_split_tuner_param_names()
            test_recipe_args(recipes)
            test_strict_checkpoint(recipes)
            test_shift_script(recipes)
            test_real_recipe_loci(recipes)
            test_real_recipe_must_read_locus(recipes)
        finally:
            os.chdir(cwd)
    print("All tuning probe tests passed.")
