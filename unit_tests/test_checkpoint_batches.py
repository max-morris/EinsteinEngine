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

"""Tests for batch ask/tell on CheckpointedOptimizer: reproducible batches, results appended to the same checkpoint
by another process, the refusals that keep a checkpoint consistent, pending batches, and the readers of the
checkpoint (read_checkpoint, generate_best, plot_tuning)."""

import dataclasses
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable

from EinsteinEngine.tuning import tuning
from EinsteinEngine.tuning.checkpointed_optimizer import FAILED_VALUE, Batch, CheckpointedOptimizer, read_checkpoint
from EinsteinEngine.tuning.experiment import Discrete, Experiment
from EinsteinEngine.tuning.generate_best import load_best_params
from EinsteinEngine.tuning.generate_best import main as generate_best_main
from EinsteinEngine.tuning.plot_tuning import load_jsonl, map_coords_to_values, plot
from EinsteinEngine.tuning.tune_composite import DiscreteKnobTuner
from EinsteinEngine.tuning.tuning import build_experiment


def expect_error[E: BaseException](kind: type[E], fn: Callable[[], Any], *fragments: str) -> E:
    try:
        fn()
    except kind as e:
        for fragment in fragments:
            assert fragment in str(e), f"expected {fragment!r} in error message {str(e)!r}"
        return e
    raise AssertionError(f"expected {kind.__name__}")


def experiment() -> Experiment:
    e = Experiment()
    e.add_in_param('x', (0, 20))
    e.add_in_param('odd', (1, 9), constraint=lambda v: v % 2 == 1)  # rejects about half the draws
    e.add_in_param('knob', Discrete(range(3)))
    e.add_in_param('rate', (0.0, 1.0), lambda params: params['knob'] == 2)
    e.add_out_param('y', lambda params: params['x'] * 10)
    return e


def objective(params: dict[str, Any]) -> float:
    return -abs(int(params['x']) - 13) - 0.1 * int(params['odd'])


def lines(path: str) -> list[dict[str, Any]]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def opt(checkpoint: str, **kwargs: Any) -> CheckpointedOptimizer:
    return CheckpointedOptimizer.for_batches(experiment(), checkpoint, **kwargs)


def test_ask_is_reproducible(directory: str) -> None:
    checkpoint = os.path.join(directory, 'repro.jsonl')
    first = opt(checkpoint).ask(5, seed=11)
    assert opt(checkpoint).ask(5, seed=11) == first and first.n_history == 0 and len(first.params) == 5
    assert all(p['odd'] % 2 == 1 for p in first.params), first.params  # infeasible draws were redrawn
    assert len({json.dumps(p, sort_keys=True) for p in first.params}) == 5  # no repeats
    assert opt(checkpoint).ask(5, seed=12).params != first.params
    unseeded = opt(checkpoint).ask(2)
    assert isinstance(unseeded.seed, int) and unseeded.seed >= 0
    assert opt(checkpoint, random_state=7).ask(2).seed == 7  # the optimizer's random_state is the default seed
    assert Batch.from_json(first.to_json()) == first and hash(first) == hash(Batch.from_json(first.to_json()))
    assert first.n_startup_trials == 10 and first.optuna_version and first.pending == ()
    expect_error(ValueError, lambda: Batch.from_json('{"seed": 1}'), 'keys')
    edited = json.loads(first.to_json())
    edited['params'][0]['x'] = (edited['params'][0]['x'] + 1) % 21
    expect_error(ValueError, lambda: Batch.from_json(json.dumps(edited)), 'does not match')
    edited = json.loads(first.to_json())
    edited['seed'] = True
    expect_error(ValueError, lambda: Batch.from_json(json.dumps(edited)), 'wrong type')
    expect_error(ValueError, lambda: opt(checkpoint).ask(0), 'at least one')
    #  The id depends on the trials: a longer batch from the same seed shares its first trials, not its id.
    longer = opt(checkpoint).ask(8, seed=11)
    assert longer.params[:5] == first.params and longer.id != first.id
    expect_error(ValueError, lambda: opt(checkpoint).maximize(), 'ask and tell only')
    assert not os.path.exists(checkpoint)  # asking writes nothing


def test_rounds(directory: str) -> None:
    """Rounds of ask in one process and tell in another, past TPE's startup trials, as a batched driver runs them."""
    checkpoint = os.path.join(directory, 'rounds.jsonl')
    told = 0
    for round_number in range(5):
        batch = Batch.from_json(opt(checkpoint, n_startup_trials=6).ask(4, seed=100 + round_number).to_json())
        assert batch.n_history == told
        #  A batch past the startup trials is still reproducible from the seed and the checkpoint prefix.
        assert opt(checkpoint, n_startup_trials=6).ask(4, seed=100 + round_number) == batch
        seen = {json.dumps(e['params'], sort_keys=True) for e in lines(checkpoint)} if told else set()
        assert not seen & {json.dumps(p, sort_keys=True) for p in batch.params}  # no trial repeats the history
        teller = opt(checkpoint, n_startup_trials=6)
        targets: list[float | None] = [objective(p) for p in batch.params]
        if round_number == 1:
            targets[0] = float('nan')  # a failed run
        teller.tell(batch, targets)
        told += 4
        #  The teller's own study includes the batch it was told.
        assert teller.max is not None and teller.max['target'] == max(e['target'] for e in lines(checkpoint))
    entries = lines(checkpoint)
    assert len(entries) == 20 and all(set(e) == {'target', 'params', 'batch', 'trial'} for e in entries)
    assert entries[4]['target'] == FAILED_VALUE and [e['trial'] for e in entries[:4]] == [0, 1, 2, 3]
    assert len({e['batch'] for e in entries}) == 5
    #  The readers of the checkpoint are unchanged: replay, generate_best's best entry, and plot_tuning.
    assert len(read_checkpoint(experiment(), checkpoint)) == 20
    best_target, best_params = load_best_params(checkpoint)
    assert best_target == max(e['target'] for e in entries)
    assert experiment().recipe_facing_args(best_params) == {'y': best_params['x'] * 10}
    records = map_coords_to_values(load_jsonl(Path(checkpoint)), experiment())
    plot(records, 'rounds', Path(directory) / 'rounds.png')
    assert (Path(directory) / 'rounds.png').stat().st_size > 0


def test_tell_refusals_and_partial_tells(directory: str) -> None:
    checkpoint = os.path.join(directory, 'refusals.jsonl')
    optimizer = opt(checkpoint)
    batch = optimizer.ask(3, seed=5)
    expect_error(ValueError, lambda: optimizer.tell(batch, [1.0, 2.0]), '3 trials but 2 targets')
    bad = dataclasses.replace(batch, params=(*batch.params[:2], {'x': 1}))
    expect_error(Exception, lambda: optimizer.tell(bad, [1.0, 2.0, 3.0]), 'Trial 2')
    assert not os.path.exists(checkpoint)  # refused before writing anything
    #  A trial that did not run (None) is not written, and can be told later; a told trial cannot be told again.
    optimizer.tell(batch, [1.0, None, 3.0])
    assert [(e['trial'], e['target']) for e in lines(checkpoint)] == [(0, 1.0), (2, 3.0)]
    expect_error(RuntimeError, lambda: optimizer.tell(batch, [1.0, None, None]), 'Trials [0]', 'already been told')
    opt(checkpoint).tell(batch, [None, 2.0, None])
    assert [e['trial'] for e in lines(checkpoint)] == [0, 2, 1]

    #  The first batch of another study (an empty checkpoint) cannot be told here.
    other = os.path.join(directory, 'other.jsonl')
    first_of_other = opt(other).ask(2, seed=6)
    expect_error(RuntimeError, lambda: optimizer.tell(first_of_other, [0.0, 0.0]), 'another study')
    #  Nor one whose study has the same file name but another probe sidecar (another probe or search space).
    elsewhere = os.path.join(directory, 'elsewhere')
    os.mkdir(elsewhere)
    twin = os.path.join(elsewhere, 'refusals.jsonl')
    asked = opt(twin).ask(1, seed=7)
    with open(twin + '.probe.json', 'w') as fh:
        fh.write('{"counts": {}, "probe_params": {}}\n')
    expect_error(RuntimeError, lambda: opt(twin).tell(asked, [0.0]), 'another study')
    #  A batch asked against entries that have since changed.
    stale = optimizer.ask(1, seed=8)
    stale = dataclasses.replace(stale, history='0' * 64)
    expect_error(RuntimeError, lambda: optimizer.tell(stale, [0.0]), 'was not asked against this checkpoint')
    long = dataclasses.replace(stale, n_history=10)
    expect_error(RuntimeError, lambda: optimizer.tell(long, [0.0]), 'fewer than the 10')
    #  Two batches asked against the same prefix may both be told (in either order).
    first = optimizer.ask(2, seed=9)
    second = optimizer.ask(2, seed=10, pending=[first])
    assert second.pending == (first.id,)
    optimizer.tell(second, [0.5, 0.5])
    optimizer.tell(first, [0.25, 0.25])
    assert len(lines(checkpoint)) == 7
    assert not os.path.exists(other)


def test_pending(directory: str) -> None:
    """A batch asked while others are out avoids their trials, and the liar counts them as running."""
    checkpoint = os.path.join(directory, 'pending.jsonl')
    optimizer = opt(checkpoint, n_startup_trials=4)
    optimizer.tell(optimizer.ask(6, seed=1), [float(i) for i in range(6)])
    out = optimizer.ask(6, seed=2)
    blind = optimizer.ask(6, seed=3)
    aware = optimizer.ask(6, seed=3, pending=[out])
    assert aware != blind and aware.pending == (out.id,)
    assert not {json.dumps(p, sort_keys=True) for p in aware.params} & \
        {json.dumps(p, sort_keys=True) for p in out.params}
    #  A pending batch's told trials are history already; one from another study is refused.
    optimizer.tell(out, [1.0, None, None, None, None, None])
    optimizer.ask(2, seed=4, pending=[out])
    other = opt(os.path.join(directory, 'pending-other.jsonl')).ask(1, seed=5)
    expect_error(RuntimeError, lambda: optimizer.ask(1, pending=[other]), 'another study')


def test_constant_liar(directory: str) -> None:
    """The liar is what spreads a batch: without it, TPE past its startup trials proposes near-copies."""
    import optuna
    from optuna.samplers import TPESampler
    checkpoint = os.path.join(directory, 'liar.jsonl')
    optimizer = opt(checkpoint, n_startup_trials=4)
    first = optimizer.ask(8, seed=1)
    optimizer.tell(first, [objective(p) for p in first.params])
    batch = optimizer.ask(8, seed=2)
    xs = {p['x'] for p in batch.params}
    #  The same draws with a sampler without the liar, keeping repeats: far fewer distinct values.
    study = optuna.create_study(direction='maximize', sampler=TPESampler(n_startup_trials=4, seed=2))
    for target, coords, specs in read_checkpoint(experiment(), checkpoint):
        study.add_trial(optuna.trial.create_trial(params=coords, value=target, distributions={
            n: optuna.distributions.IntDistribution(int(s.lo), int(s.hi)) if s.kind is int
            else optuna.distributions.FloatDistribution(float(s.lo), float(s.hi)) for n, s in specs.items()}))
    plain = set()
    for _ in range(8):
        trial = study.ask()
        plain.add(trial.suggest_int('x', 0, 20))
    assert len(xs) > len(plain), (xs, plain)


GENERATE_RECIPE = """\
import json
from EinsteinEngine.tuning.tuning import get_tuning_param
with open({out!r}, 'w') as fh:
    json.dump({{'layout': get_tuning_param('layout', None)}}, fh)
"""


def test_generate_best_batch_trial(directory: str) -> None:
    """generate_best --batch FILE --trial I runs the recipe with one trial of a batch, from the sidecar alone."""
    d = Path(directory) / 'generate'
    d.mkdir()
    out = d / 'out.json'
    recipe = d / 'recipe.py'
    recipe.write_text(GENERATE_RECIPE.format(out=str(out)))
    tuner_file = d / 'tuner.py'
    tuner_file.write_text(textwrap.dedent("""\
        from EinsteinEngine.tuning.tune_composite import DiscreteKnobTuner
        tuner = DiscreteKnobTuner('layout', ['fused', 'split', 'per-component'])
        """))
    checkpoint = str(d / 'study.jsonl')
    tuner = DiscreteKnobTuner('layout', ['fused', 'split', 'per-component'])
    e = build_experiment(tuner, None, checkpoint, record_probe=True)  # writes the sidecar (the search space)
    batch = CheckpointedOptimizer.for_batches(e, checkpoint).ask(3, seed=1)
    batch_file = d / 'batch.json'
    batch_file.write_text(batch.to_json())
    saved = sys.argv
    try:
        for i, params in enumerate(batch.params):
            sys.argv = ['generate_best', str(recipe), str(tuner_file), '--checkpoint-file', checkpoint,
                        '--batch', str(batch_file), '--trial', str(i)]
            generate_best_main()
            assert json.loads(out.read_text()) == {'layout': tuner.choices[int(params['layout'])]}
        #  A batch of another study is refused.
        other = CheckpointedOptimizer.for_batches(e, str(d / 'other.jsonl')).ask(1, seed=2)
        batch_file.write_text(other.to_json())
        sys.argv = ['generate_best', str(recipe), str(tuner_file), '--checkpoint-file', checkpoint,
                    '--batch', str(batch_file), '--trial', '0']
        expect_error(SystemExit, generate_best_main, 'another study')
    finally:
        sys.argv = saved
        tuning._tuning_params = None


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        test_ask_is_reproducible(tmp)
        test_rounds(tmp)
        test_tell_refusals_and_partial_tells(tmp)
        test_pending(tmp)
        test_constant_liar(tmp)
        test_generate_best_batch_trial(tmp)
    print("All checkpoint batch tests passed.")
