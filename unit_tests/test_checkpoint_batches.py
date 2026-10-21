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
by another process, and the refusals that keep a checkpoint consistent."""

import json
import os
import tempfile
from typing import Any, Callable

from EinsteinEngine.tuning.checkpointed_optimizer import FAILED_VALUE, Batch, CheckpointedOptimizer, read_checkpoint
from EinsteinEngine.tuning.experiment import Discrete, Experiment
from EinsteinEngine.tuning.generate_best import _FixedTrial, load_best_params


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


def test_ask_is_reproducible(directory: str) -> None:
    checkpoint = os.path.join(directory, 'repro.jsonl')
    first = CheckpointedOptimizer(None, experiment(), checkpoint).ask(5, seed=11)
    again = CheckpointedOptimizer(None, experiment(), checkpoint).ask(5, seed=11)
    assert first == again and first.n_history == 0 and len(first.params) == 5
    assert all(p['odd'] % 2 == 1 for p in first.params), first.params  # infeasible draws were redrawn
    assert CheckpointedOptimizer(None, experiment(), checkpoint).ask(5, seed=12).params != first.params
    unseeded = CheckpointedOptimizer(None, experiment(), checkpoint).ask(2)
    assert isinstance(unseeded.seed, int) and unseeded.seed >= 0
    assert Batch.from_json(first.to_json()) == first
    expect_error(ValueError, lambda: Batch.from_json('{"seed": 1}'), 'keys')
    expect_error(ValueError, lambda: CheckpointedOptimizer(None, experiment(), checkpoint).ask(0), 'at least one')
    assert not os.path.exists(checkpoint)  # asking writes nothing


def test_rounds(directory: str) -> None:
    """Rounds of ask in one process and tell in another, past TPE's startup trials, as a batched driver runs them."""
    checkpoint = os.path.join(directory, 'rounds.jsonl')
    told = 0
    for round_number in range(4):
        batch = Batch.from_json(CheckpointedOptimizer(None, experiment(), checkpoint, n_startup_trials=6)
                                .ask(4, seed=100 + round_number).to_json())
        assert batch.n_history == told
        #  A batch past the startup trials is still reproducible from the seed and the checkpoint prefix.
        assert CheckpointedOptimizer(None, experiment(), checkpoint, n_startup_trials=6).ask(
            4, seed=100 + round_number) == batch
        teller = CheckpointedOptimizer(None, experiment(), checkpoint, n_startup_trials=6)
        targets = [objective(p) for p in batch.params]
        if round_number == 1:
            targets[0] = float('nan')  # a failed run
        teller.tell(batch, targets)
        told += 4
        #  The teller's own study includes the batch it was told.
        assert teller.max is not None and teller.max['target'] == max(e['target'] for e in lines(checkpoint))
    entries = lines(checkpoint)
    assert len(entries) == 16 and all(set(e) == {'target', 'params', 'batch'} for e in entries)
    assert entries[4]['target'] == FAILED_VALUE
    assert len({e['batch'] for e in entries}) == 4
    #  The readers of the checkpoint are unchanged: replay, generate_best's best entry and its fixed trial.
    assert len(read_checkpoint(experiment(), checkpoint)) == 16
    best_target, best_params = load_best_params(checkpoint)
    assert best_target == max(e['target'] for e in entries)
    assert experiment().suggest_params(_FixedTrial(best_params)).recipe_facing_args == {'y': best_params['x'] * 10}


def test_tell_refusals(directory: str) -> None:
    checkpoint = os.path.join(directory, 'refusals.jsonl')
    optimizer = CheckpointedOptimizer(None, experiment(), checkpoint)
    batch = optimizer.ask(3, seed=5)
    expect_error(ValueError, lambda: optimizer.tell(batch, [1.0, 2.0]), '3 trials but 2 targets')
    bad = Batch(batch.seed, batch.n_history, batch.history, (*batch.params[:2], {'x': 1}))
    expect_error(Exception, lambda: optimizer.tell(bad, [1.0, 2.0, 3.0]), 'Trial 2')
    assert not os.path.exists(checkpoint)  # refused before writing anything
    optimizer.tell(batch, [1.0, 2.0, 3.0])
    expect_error(RuntimeError, lambda: optimizer.tell(batch, [1.0, 2.0, 3.0]), 'already been told')
    assert len(lines(checkpoint)) == 3

    #  A batch asked against another checkpoint, or against entries that have since changed.
    other = os.path.join(directory, 'other.jsonl')
    CheckpointedOptimizer(None, experiment(), other).tell(CheckpointedOptimizer(None, experiment(), other).ask(
        3, seed=6), [0.0, 0.0, 0.0])
    from_other = CheckpointedOptimizer(None, experiment(), other).ask(2, seed=7)
    expect_error(RuntimeError, lambda: optimizer.tell(from_other, [0.0, 0.0]), 'was not asked against')
    long = CheckpointedOptimizer(None, experiment(), other).ask(1, seed=8)
    short = Batch(long.seed, 10, long.history, long.params)
    expect_error(RuntimeError, lambda: optimizer.tell(short, [0.0]), 'fewer than the 10')
    #  Two batches asked against the same prefix may both be told (in either order).
    first = optimizer.ask(2, seed=9)
    second = optimizer.ask(2, seed=10)
    optimizer.tell(second, [0.5, 0.5])
    optimizer.tell(first, [0.25, 0.25])
    assert len(lines(checkpoint)) == 7


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        test_ask_is_reproducible(tmp)
        test_rounds(tmp)
        test_tell_refusals(tmp)
    print("All checkpoint batch tests passed.")
