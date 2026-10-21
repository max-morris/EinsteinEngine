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

"""Checkpointing optimization backed by Optuna's TPESampler.

The checkpoint is a JSON-lines file with one {"target": float, "params": {...}} object per trial. TPE copes with
non-smooth objectives, integer and combinatorial parameters, and noisy timings.

Besides `maximize`, which runs one trial at a time, a study can be driven in batches: `ask(n)` proposes n trials as a
`Batch`, which the caller runs however it likes (for example baked on one machine, built and timed on others), and
`tell(batch, targets)` appends the results to the checkpoint. Each call rebuilds the study from the checkpoint, so the
two may run in different processes; the batch, saved with `Batch.to_json`, carries what links them.
"""

import hashlib
import json
import os
import secrets
import subprocess
import time
import urllib.request
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from EinsteinEngine.tuning.experiment import Coord, CoordSpec, Experiment, InfeasibleParamError

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Sentinel returned to the optimizer for runs that fail outright (-inf), and written to the checkpoint for them.
# Large-but-finite so TPE still learns these are bad regions.
FAILED_VALUE: float = -1e9


def is_failed(target: float) -> bool:
    """Whether a checkpoint target records a failed run (the sentinel, or a non-finite value from older checkpoints)."""
    return not np.isfinite(target) or target == FAILED_VALUE

# Infeasible (pruned) trials don't consume the iteration budget, so cap total
# attempts at this multiple of the budget in case a constraint rejects nearly
# the entire search space.
_MAX_ATTEMPT_FACTOR: int = 100


def _distribution_of(spec: CoordSpec) -> optuna.distributions.BaseDistribution:
    """Build the Optuna distribution for a param's raw-coordinate CoordSpec."""
    if spec.kind is int:
        return optuna.distributions.IntDistribution(int(spec.lo), int(spec.hi))
    return optuna.distributions.FloatDistribution(float(spec.lo), float(spec.hi))


def _add_trials(study: optuna.Study,
                entries: Sequence[tuple[float, dict[str, Coord], dict[str, CoordSpec]]]) -> int:
    """Add checkpoint entries (as read_checkpoint returns them) to a study as completed trials."""
    for target, coords, specs in entries:
        study.add_trial(optuna.trial.create_trial(
            params=coords,
            distributions={name: _distribution_of(spec) for name, spec in specs.items()},
            value=target,
        ))
    return len(entries)


def read_checkpoint(
    experiment: Experiment, checkpoint_file: str
) -> list[tuple[float, dict[str, Coord], dict[str, CoordSpec]]]:
    """Read a JSON-lines checkpoint and replay each entry through ``experiment``.

    Returns ``(target, coords, specs)`` per entry, with failed runs' targets
    replaced by the finite failure value; a missing file has no entries.
    Raises (see ``Experiment.check_declared``) if an entry does not fit the
    experiment exactly, naming the file and line.
    """
    if not os.path.exists(checkpoint_file):
        return []
    entries: list[tuple[float, dict[str, Coord], dict[str, CoordSpec]]] = []
    with open(checkpoint_file) as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            entry: dict[str, Any] = json.loads(line)
            experiment.check_declared(entry['params'], f'{checkpoint_file}:{line_number}')
            raw_target: float = float(entry['target'])
            target = raw_target if np.isfinite(raw_target) else FAILED_VALUE
            # Replay the stored raw coordinates through the experiment so
            # dynamic domains recover their exact per-trial coordinate ranges
            # (for plain Intervals a coordinate is just its value).
            coords, specs = experiment.reconstruct_coords(entry['params'])
            entries.append((target, dict(coords), dict(specs)))
    return entries


def _history_digest(checkpoint_file: str, n_entries: int) -> str:
    """The SHA-256 of the first `n_entries` non-blank lines of a checkpoint, each stripped; raises if it has fewer."""
    digest = hashlib.sha256()
    found = 0
    if n_entries > 0:
        with open(checkpoint_file) as fh:
            for line in fh:
                if found == n_entries:
                    break
                if line := line.strip():
                    digest.update(line.encode() + b'\n')
                    found += 1
    if found < n_entries:
        raise RuntimeError(f"The checkpoint {checkpoint_file} has {found} entries, fewer than the {n_entries} a batch "
                           f"was asked against.")
    return digest.hexdigest()


@dataclass(frozen=True)
class Batch:
    """
    Trials proposed together by `CheckpointedOptimizer.ask`. `params` holds each trial's in-params as the checkpoint
    stores them (raw coordinates); the recipe-facing args follow from them through the Experiment, as generate_best
    gets them. `seed` and the first `n_history` checkpoint entries (whose digest is `history`) determine the batch:
    asking again with them gives the same params.
    """
    seed: int
    n_history: int
    history: str
    params: tuple[dict[str, Coord], ...]

    @property
    def id(self) -> str:
        """Written with each of the batch's entries in the checkpoint, so a batch cannot be told twice."""
        return f'{self.n_history}-{self.seed}-{self.history[:16]}'

    def to_json(self) -> str:
        return json.dumps({'seed': self.seed, 'n_history': self.n_history, 'history': self.history,
                           'params': list(self.params)}, indent=1)

    @staticmethod
    def from_json(text: str) -> 'Batch':
        obj = json.loads(text)
        if not isinstance(obj, dict) or set(obj) != {'seed', 'n_history', 'history', 'params'} or \
                not isinstance(obj['params'], list) or not all(isinstance(p, dict) for p in obj['params']):
            raise ValueError('Expected a JSON object with the keys seed, n_history, history and params (a list of '
                             'objects), as Batch.to_json writes it')
        return Batch(int(obj['seed']), int(obj['n_history']), str(obj['history']), tuple(obj['params']))


class CheckpointedOptimizer:
    """Optuna-backed optimizer that checkpoints every probe to a
    JSON-lines file and resumes from it on construction.
    """

    def __init__(
        self,
        f: Callable[..., float] | None,
        experiment: Experiment,
        checkpoint_file: str,
        random_state: int | None = None,
        verbose: int = 2,
        telegram_verbosity: int = 1,
        n_startup_trials: int = 10,
        **_kwargs: Any,  # absorb legacy kwargs (acquisition_function, etc.)
    ) -> None:
        self._f = f
        self._verbose = verbose
        self._telegram_verbosity = telegram_verbosity
        self._checkpoint_file = checkpoint_file
        self._experiment = experiment
        self._n_startup_trials = n_startup_trials

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
            sampler = TPESampler(
                n_startup_trials=n_startup_trials,
                seed=random_state,
            )
        self._study = optuna.create_study(direction='maximize', sampler=sampler)
        self._n_checkpoint_loaded = self._load_checkpoint()

    @property
    def n_checkpoint_loaded(self) -> int:
        """Number of observations restored from the checkpoint file."""
        return self._n_checkpoint_loaded

    @property
    def max(self) -> dict[str, Any] | None:
        """Best result so far: ``{'target': float, 'params': dict}``."""
        try:
            best = self._study.best_trial
            return {'target': best.value, 'params': dict(best.params)}
        except ValueError:
            return None

    def _load_checkpoint(self) -> int:
        return _add_trials(self._study, read_checkpoint(self._experiment, self._checkpoint_file))

    def ask(self, n: int, *, seed: int | None = None) -> Batch:
        """
        Propose `n` trials at once, from a study rebuilt from the checkpoint as it is now. The sampler is TPE with
        `constant_liar`, so each trial is proposed knowing the earlier ones of the batch are pending, and is seeded with
        `seed` (a random one if None), which the batch records. Trials that violate a parameter constraint are
        rejected and redrawn, as in `maximize`. Nothing is written; save the batch with `Batch.to_json`.
        """
        if n < 1:
            raise ValueError(f"A batch needs at least one trial, got {n}")
        seed = secrets.randbelow(2**31) if seed is None else seed
        history = read_checkpoint(self._experiment, self._checkpoint_file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
            sampler = TPESampler(n_startup_trials=self._n_startup_trials, seed=seed, constant_liar=True)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        _add_trials(study, history)
        proposed: list[dict[str, Coord]] = []
        for _ in range(_MAX_ATTEMPT_FACTOR * n):
            if len(proposed) == n:
                break
            trial = study.ask()
            try:
                params, _ = self._experiment.suggest_params(trial)
            except InfeasibleParamError:
                study.tell(trial, state=TrialState.PRUNED)
                continue
            proposed.append(dict(params))  # left RUNNING: the constant liar counts it as pending
        if len(proposed) < n:
            raise RuntimeError(f"Proposed only {len(proposed)} of {n} trials in {_MAX_ATTEMPT_FACTOR * n} attempts; "
                               f"parameter constraints may reject too much of the search space.")
        return Batch(seed, len(history), _history_digest(self._checkpoint_file, len(history)), tuple(proposed))

    def tell(self, batch: Batch, targets: Sequence[float]) -> None:
        """
        Append a batch's results to the checkpoint, in the batch's order: `targets[i]` is the value (maximized) of
        `batch.params[i]`, and a non-finite target records a failed run. Refuses a batch asked against another
        checkpoint (or one whose first `batch.n_history` entries have changed), a batch already told, and params that
        do not fit the Experiment, before writing anything.
        """
        if len(targets) != len(batch.params):
            raise ValueError(f"The batch has {len(batch.params)} trials but {len(targets)} targets were given")
        if _history_digest(self._checkpoint_file, batch.n_history) != batch.history:
            raise RuntimeError(f"Batch {batch.id} was not asked against the checkpoint {self._checkpoint_file}: its "
                               f"first {batch.n_history} entries differ from the ones the batch was asked against.")
        if os.path.exists(self._checkpoint_file):
            with open(self._checkpoint_file) as fh:
                if any(line.strip() and json.loads(line).get('batch') == batch.id for line in fh):
                    raise RuntimeError(f"Batch {batch.id} has already been told to {self._checkpoint_file}.")
        for i, params in enumerate(batch.params):
            self._experiment.check_declared(params, f'Trial {i} of batch {batch.id}')
        values = [float(t) if np.isfinite(t) else FAILED_VALUE for t in targets]
        entries = [(value, *self._experiment.reconstruct_coords(params)) for value, params in zip(values, batch.params)]
        with open(self._checkpoint_file, 'a') as fh:
            for value, params in zip(values, batch.params):
                fh.write(json.dumps({'target': value, 'params': params, 'batch': batch.id}) + '\n')
        _add_trials(self._study, [(value, dict(coords), dict(specs)) for value, coords, specs in entries])

    def _objective(self, trial: optuna.Trial) -> float:
        try:
            in_args, out_args = self._experiment.suggest_params(trial)
        except InfeasibleParamError as e:
            # Reject the trial before running the (expensive) objective.
            # Pruned trials are not checkpointed and don't feed the sampler.
            raise optuna.TrialPruned(str(e)) from e
        assert self._f is not None

        # Capture the best value so far *before* this trial so we can detect
        # whether this trial sets a new record.
        try:
            prev_best = self._study.best_value
        except ValueError:
            prev_best = None

        run_start = time.perf_counter()
        result = self._f(**out_args)
        elapsed = time.perf_counter() - run_start
        value = float(result) if np.isfinite(result) else FAILED_VALUE

        is_new_best = value != FAILED_VALUE and (prev_best is None or value > prev_best)
        best_value = value if is_new_best else prev_best

        # Total completed iterations including checkpointed/resumed trials, plus
        # this one (which is still RUNNING and so not yet counted by the study).
        total_iterations = sum(
            1 for t in self._study.trials if t.state == TrialState.COMPLETE) + 1

        self._notify(
            value=value,
            is_new_best=is_new_best,
            best_value=best_value,
            elapsed=elapsed,
            total_iterations=total_iterations,
        )

        # Append to the JSONL checkpoint.
        entry: dict[str, Any] = {
            'target': value,
            'params': in_args,
        }
        with open(self._checkpoint_file, 'a') as fh:
            fh.write(json.dumps(entry) + '\n')

        if self._verbose >= 1:
            try:
                best = self._study.best_value
            except ValueError:
                best = float('nan')
            print(f'  trial {trial.number:4d} | value: {value:+.6f} | best: {best:+.6f} | {in_args}')

        return value

    def _notify(
        self,
        *,
        value: float,
        is_new_best: bool,
        best_value: float | None,
        elapsed: float,
        total_iterations: int,
    ) -> None:
        """Send a Telegram message about a completed run, per the configured
        telegram verbosity level.

          0: never send.
          1: send only when a new best is found (legacy behavior).
          2: send after every run: the time found and whether it's a new best;
             if not, also the current best.
          3: as 2, plus how long the run took, how many iterations have run in
             total (including resumed ones), and the current Baton Rouge, LA
             temperature and rain chance.
        """
        v = self._telegram_verbosity
        if v <= 0:
            return

        if v == 1:
            if is_new_best:
                self._telegram_send(f'New best time found: {abs(value)} seconds')
            return

        # v >= 2: report on every run.
        lines: list[str] = []
        if value == FAILED_VALUE:
            lines.append('Run finished but produced no valid time (discarded).')
        else:
            lines.append(f'Run finished: {abs(value)} seconds.')
            lines.append('This is a new best!' if is_new_best else 'This is not a new best.')

        if not is_new_best and best_value is not None:
            lines.append(f'Current best: {abs(best_value)} seconds.')

        if v >= 3:
            lines.append(f'Run took {elapsed:.1f} seconds.')
            lines.append(f'Total iterations so far: {total_iterations}.')
            weather = self._get_baton_rouge_weather()
            if weather is not None:
                temperature, rain_chance = weather
                lines.append(
                    f'Baton Rouge, LA: {temperature}\N{DEGREE SIGN}F, '
                    f'{rain_chance}% chance of rain.')
            else:
                lines.append('Baton Rouge, LA weather: unavailable.')

        self._telegram_send('\n'.join(lines))

    @staticmethod
    def _telegram_send(message: str) -> None:
        """Invoke telegram-send with a message.

        Best-effort: failures to invoke telegram-send must never interrupt the
        optimization run.
        """
        try:
            subprocess.run(
                ['telegram-send', message],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            warnings.warn(f'Failed to send Telegram notification: {exc}')

    @staticmethod
    def _get_baton_rouge_weather() -> tuple[float, float] | None:
        """Return ``(temperature_fahrenheit, rain_chance_percent)`` for Baton
        Rouge, LA, or ``None`` if the lookup fails.

        Best-effort and dependency-free: uses only the standard library
        (urllib) against the keyless Open-Meteo API, and never raises.
        """
        url = (
            'https://api.open-meteo.com/v1/forecast'
            '?latitude=30.4515&longitude=-91.1871'
            '&current=temperature_2m'
            '&hourly=precipitation_probability'
            '&temperature_unit=fahrenheit'
            '&timezone=America%2FChicago'
            '&forecast_days=1'
        )
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            temperature = data['current']['temperature_2m']
            # 'current.time' is minute-resolution ('...THH:MM'); hourly times are
            # hour-resolution ('...THH:00'), so match on the 'YYYY-MM-DDTHH' prefix.
            current_hour = data['current']['time'][:13]
            hourly = data['hourly']
            times = hourly['time']
            probabilities = hourly['precipitation_probability']
            rain_chance = next(
                (p for t, p in zip(times, probabilities) if t.startswith(current_hour)),
                probabilities[0],
            )
            return temperature, rain_chance
        except Exception:
            return None

    def maximize(self, warmup_iterations: int = 10, iterations: int = 20) -> None:
        """Run optimization, deducting already-loaded probes from the budget.

        Trials pruned for violating a parameter constraint do not consume the
        budget: optimization continues until the study holds
        ``warmup_iterations + iterations`` completed trials, subject to a hard
        cap on total attempts.
        """

        n = self._n_checkpoint_loaded
        total = warmup_iterations + iterations
        n_remaining = max(0, total - n)
        if n_remaining == 0:
            return

        self._study.optimize(
            self._objective,
            n_trials=_MAX_ATTEMPT_FACTOR * n_remaining,
            callbacks=[MaxTrialsCallback(total, states=(TrialState.COMPLETE,))],
        )

        n_completed = sum(1 for t in self._study.trials if t.state == TrialState.COMPLETE)
        if n_completed < total:
            warnings.warn(
                f'Stopped after {n_completed}/{total} completed trials; the attempt cap '
                f'({_MAX_ATTEMPT_FACTOR * n_remaining}) was reached. Parameter constraints '
                f'may reject too much of the search space.'
            )
