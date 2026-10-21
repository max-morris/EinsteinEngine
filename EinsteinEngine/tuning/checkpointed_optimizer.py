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
two may run in different processes; the batch, saved with `Batch.to_json`, carries what links them. Told entries also
record the batch id and trial index; `seed` records externally chosen trials (with their source). The readers,
`read_checkpoint`, `generate_best` and `plot_tuning`, ignore these extra keys.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import subprocess
import time
import urllib.request
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
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


def study_digest(experiment: Experiment, checkpoint_file: str) -> str:
    """
    What identifies a study apart from its entries: its probe sidecar's study id, probe counts, probe params and
    derived orders (`ProbeRecord.identity`), and the Experiment's in-param names. The sidecar must exist and have a
    study id: `build_experiment(..., record_probe=True)` writes one. A new study must not copy another's sidecar.
    """
    from EinsteinEngine.tuning.tuning import ProbeRecord, probe_sidecar_path  # imported here: tuning imports the engine
    sidecar = probe_sidecar_path(checkpoint_file)
    if not os.path.exists(sidecar):
        raise RuntimeError(f"Batches need the study's probe file {sidecar}; build the Experiment with "
                           f"build_experiment(..., record_probe=True) first.")
    with open(sidecar) as fh:
        record = ProbeRecord.from_json(fh.read())
    if record.study is None:
        raise RuntimeError(f"The probe file {sidecar} has no study id (it predates them); build the Experiment with "
                           f"build_experiment(..., record_probe=True) once to add one.")
    blob = json.dumps({'record': record.identity(), 'in_params': sorted(experiment.in_params)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_coord(x: object) -> bool:
    return _is_int(x) or isinstance(x, float)


@dataclass(frozen=True)
class Batch:
    """
    Trials proposed together by `CheckpointedOptimizer.ask`. `params` holds each trial's in-params as the checkpoint
    stores them (raw coordinates); `Experiment.recipe_facing_args` gives a trial's recipe-facing args.

    `study` identifies the study it was asked for (checkpoint name, probe sidecar, in-param names), and `n_history`
    and `history` the checkpoint entries it was asked against (their number and SHA-256). Asking again with the same
    `seed`, from a checkpoint holding exactly those entries and with the same `pending` batches, gives the same params
    (with the same `n_startup_trials` and optuna version, also recorded).
    """
    seed: int
    n_history: int
    history: str
    study: str
    n_startup_trials: int
    optuna_version: str
    pending: tuple[str, ...]
    params: tuple[dict[str, Coord], ...]
    #  Draws discarded while asking (infeasible, or repeating a trial); they acted as extra liars.
    rejected: int = 0

    @property
    def id(self) -> str:
        """Unique to these trials. Written with each told entry, so a trial cannot be told twice."""
        blob = json.dumps([self.seed, self.n_history, self.history, self.study, self.n_startup_trials,
                           self.optuna_version, list(self.pending), list(self.params), self.rejected], sort_keys=True)
        return f'{self.n_history}-{self.seed}-{hashlib.sha256(blob.encode()).hexdigest()[:16]}'

    def __hash__(self) -> int:
        return hash(self.id)

    def to_json(self) -> str:
        return json.dumps({'id': self.id, 'seed': self.seed, 'n_history': self.n_history, 'history': self.history,
                           'study': self.study, 'n_startup_trials': self.n_startup_trials,
                           'optuna_version': self.optuna_version, 'pending': list(self.pending),
                           'rejected': self.rejected, 'params': list(self.params)}, indent=1) + '\n'

    @staticmethod
    def from_json(text: str) -> 'Batch':
        obj = json.loads(text)
        keys = {'id', 'seed', 'n_history', 'history', 'study', 'n_startup_trials', 'optuna_version', 'pending',
                'rejected', 'params'}
        if not isinstance(obj, dict) or set(obj) != keys:
            raise ValueError(f'Expected a JSON object with the keys {sorted(keys)}, as Batch.to_json writes it')
        ok = (_is_int(obj['seed']) and _is_int(obj['n_history']) and obj['n_history'] >= 0
              and _is_int(obj['n_startup_trials']) and _is_int(obj['rejected']) and all(isinstance(obj[k], str) for k in
                                                            ('id', 'history', 'study', 'optuna_version'))
              and isinstance(obj['pending'], list) and all(isinstance(x, str) for x in obj['pending'])
              and isinstance(obj['params'], list) and obj['params']
              and all(isinstance(p, dict) and all(isinstance(k, str) and _is_coord(x) for k, x in p.items())
                      for p in obj['params']))
        if not ok:
            raise ValueError('A batch file has a field of the wrong type (see Batch.to_json)')
        batch = Batch(obj['seed'], obj['n_history'], obj['history'], obj['study'], obj['n_startup_trials'],
                      obj['optuna_version'], tuple(obj['pending']), tuple(obj['params']), obj['rejected'])
        if batch.id != obj['id']:
            raise ValueError(f"The batch file's id {obj['id']} does not match its contents ({batch.id}); was it edited?")
        return batch


def _append(checkpoint_file: str, lines: Sequence[str]) -> None:
    """Append JSON lines to a checkpoint in one write (looping on a short one), then fsync."""
    fd = os.open(checkpoint_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        data = ('\n'.join(lines) + '\n').encode()
        while data:
            data = data[os.write(fd, data):]
        os.fsync(fd)
    finally:
        os.close(fd)


class _StoredTrial:
    """A TrialObj that replays stored coordinates (to check a seeded trial's constraints)."""

    def __init__(self, params: Mapping[str, Coord]) -> None:
        self._params = params

    def suggest_int(self, name: str, _lo: int, _hi: int, /) -> int:
        return int(self._params[name])

    def suggest_float(self, name: str, _lo: float, _hi: float, /) -> float:
        return float(self._params[name])


@contextlib.contextmanager
def _locked(checkpoint_file: str, exclusive: bool = True) -> Iterator[None]:
    """Hold a lock on `<checkpoint>.lock`: exclusive for tell's check-then-append, shared for ask's reads."""
    with open(f'{checkpoint_file}.lock', 'a') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


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
        self._random_state = random_state

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

    @classmethod
    def for_batches(cls, experiment: Experiment, checkpoint_file: str, n_startup_trials: int = 10,
                    random_state: int | None = None) -> 'CheckpointedOptimizer':
        """An optimizer for `ask` and `tell` only (no objective to `maximize`)."""
        return cls(None, experiment, checkpoint_file, random_state=random_state, n_startup_trials=n_startup_trials)

    def _told(self) -> set[tuple[str, int]]:
        """The (batch id, trial index) pairs already told to the checkpoint."""
        told: set[tuple[str, int]] = set()
        if os.path.exists(self._checkpoint_file):
            with open(self._checkpoint_file) as fh:
                for line in fh:
                    if line.strip() and 'batch' in (entry := json.loads(line)):
                        told.add((entry['batch'], entry.get('trial', -1)))
        return told

    def ask(self, n: int, *, seed: int | None = None, pending: Sequence[Batch] = ()) -> Batch:
        """
        Propose `n` trials at once, from a study rebuilt from the checkpoint as it is now.

        - The sampler is TPE with `constant_liar`: each trial is proposed knowing that the earlier ones of the batch,
          and the not yet told trials of the `pending` batches (asked earlier for this study), are running. Pass the
          batches still out, or a second batch can repeat their configurations.
        - It is seeded with `seed` (by default the optimizer's `random_state`, or a random seed), which the batch
          records. Below `n_startup_trials` told entries, TPE samples at random.
        - A draw that violates a parameter constraint, or repeats a trial of the checkpoint, of a pending batch or of
          this batch, is redrawn. It stays running in the rebuilt study (which is discarded), so the liar steers the
          later draws away from it; it is not recorded anywhere.

        Nothing is written; save the batch with `Batch.to_json`. Targets are maximized, so tell `-time`.
        """
        if n < 1:
            raise ValueError(f"A batch needs at least one trial, got {n}")
        if seed is None:
            seed = self._random_state if self._random_state is not None else secrets.randbelow(2**31)
        study_id = study_digest(self._experiment, self._checkpoint_file)
        with _locked(self._checkpoint_file, exclusive=False):
            history = read_checkpoint(self._experiment, self._checkpoint_file)
            digest = _history_digest(self._checkpoint_file, len(history))
            told = self._told()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
            sampler = TPESampler(n_startup_trials=self._n_startup_trials, seed=seed, constant_liar=True)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        _add_trials(study, history)
        seen = [coords for _, coords, _ in history]
        for other in pending:
            if other.study != study_id:
                raise RuntimeError(f"Pending batch {other.id} belongs to another study than {self._checkpoint_file}.")
            for i, params in enumerate(other.params):
                if (other.id, i) in told:
                    continue  # told, so already in the history
                self._experiment.check_declared(params, f'Trial {i} of pending batch {other.id}')
                coords, _ = self._experiment.reconstruct_coords(params)
                study.enqueue_trial(dict(coords), skip_if_exists=False)
                self._experiment.suggest_params(study.ask())  # left RUNNING with those params
                seen.append(dict(coords))
        proposed: list[dict[str, Coord]] = []
        rejected = 0
        for _ in range(_MAX_ATTEMPT_FACTOR * n):
            if len(proposed) == n:
                break
            trial = study.ask()
            try:
                params, _ = self._experiment.suggest_params(trial)
            except InfeasibleParamError:
                rejected += 1
                continue  # left RUNNING (with the params it drew), so the liar steers the next draws away from it
            if dict(params) in seen or dict(params) in proposed:
                rejected += 1
                continue  # likewise
            proposed.append(dict(params))  # left RUNNING: the constant liar counts it as pending
        if len(proposed) < n:
            raise RuntimeError(f"Proposed only {len(proposed)} of {n} new trials in {_MAX_ATTEMPT_FACTOR * n} "
                               f"attempts; parameter constraints may reject too much of the search space, or too "
                               f"little of it is left.")
        if rejected > n:
            warnings.warn(f"ask rejected {rejected} draws for a batch of {n} (infeasible, or repeating a trial); each "
                          f"acted as an extra liar, which can push the batch away from the best trials so far.")
        return Batch(seed, len(history), digest, study_id, self._n_startup_trials, optuna.__version__,
                     tuple(b.id for b in pending), tuple(proposed), rejected)

    def tell(self, batch: Batch, targets: Sequence[float | None]) -> None:
        """
        Append a batch's results to the checkpoint: `targets[i]` is the value (maximized, so `-time`) of
        `batch.params[i]`. A non-finite target records a failed run (a bad configuration). None records nothing, for a
        trial that did not run (a lost node, say): tell it later, in another call with the batch. Each entry also
        records the batch id and trial index.

        Refuses, before writing anything: a batch asked for another study (another checkpoint, probe sidecar or
        Experiment) or against other first `batch.n_history` entries; a trial already told; and params that do not fit
        the Experiment. The checks and the append hold a lock on `<checkpoint>.lock`, and the entries are appended in
        one write.
        """
        if len(targets) != len(batch.params):
            raise ValueError(f"The batch has {len(batch.params)} trials but {len(targets)} targets were given")
        if batch.study != study_digest(self._experiment, self._checkpoint_file):
            raise RuntimeError(f"Batch {batch.id} was asked for another study than {self._checkpoint_file} (another "
                               f"checkpoint name, probe sidecar or set of in-params).")
        todo = [i for i, t in enumerate(targets) if t is not None]
        for i in todo:
            self._experiment.check_declared(batch.params[i], f'Trial {i} of batch {batch.id}')
        with _locked(self._checkpoint_file):
            if _history_digest(self._checkpoint_file, batch.n_history) != batch.history:
                raise RuntimeError(f"Batch {batch.id} was not asked against this checkpoint: the first "
                                   f"{batch.n_history} entries of {self._checkpoint_file} differ from the ones it "
                                   f"was asked against.")
            told = self._told()
            if again := sorted(i for i in todo if (batch.id, i) in told):
                raise RuntimeError(f"Trials {again} of batch {batch.id} have already been told to "
                                   f"{self._checkpoint_file}.")
            entries = []
            lines = []
            for i in todo:
                t = targets[i]
                assert t is not None
                value = float(t) if np.isfinite(t) else FAILED_VALUE
                entries.append((value, *self._experiment.reconstruct_coords(batch.params[i])))
                lines.append(json.dumps({'target': value, 'params': batch.params[i], 'batch': batch.id, 'trial': i}))
            if lines:
                _append(self._checkpoint_file, lines)
        _add_trials(self._study, [(value, dict(coords), dict(specs)) for value, coords, specs in entries])

    def seed(self, params: Mapping[str, Coord], value: float, *, source: str) -> None:
        """
        Record an externally chosen trial, for example a hand-picked configuration measured outside the study, so
        that TPE starts from it. `params` are in-params as the checkpoint stores them (raw coordinates), and `value`
        is its target (maximized, so `-time`; non-finite records a failed run). It is checked against the study's
        Experiment like any trial (names, domains, conditions and constraints, which the probe shaped), then
        appended under the lock as `{"target", "params", "seeded": source}` and replayed on resume like any entry.
        Refuses an empty `source`, and the same params from the same source twice.
        """
        if not source:
            raise ValueError("A seeded trial needs a source (where it comes from, e.g. 'round 0: v8')")
        stored = dict(params)
        what = f"The seeded trial from {source!r}"
        try:
            self._experiment.check_declared(stored, what)
            realized, _ = self._experiment.suggest_params(_StoredTrial(stored))
        except InfeasibleParamError as e:
            raise ValueError(f"{what} violates a constraint: {e}") from e
        if extra := sorted(set(stored) - set(realized)):
            raise ValueError(f"{what} has params whose conditions do not hold for its other values: {extra}")
        coords, specs = self._experiment.reconstruct_coords(stored)
        target = float(value) if np.isfinite(value) else FAILED_VALUE
        with _locked(self._checkpoint_file):
            if os.path.exists(self._checkpoint_file):
                with open(self._checkpoint_file) as fh:
                    for line in fh:
                        if line.strip() and (entry := json.loads(line)).get('seeded') == source \
                                and entry['params'] == stored:
                            raise RuntimeError(f"{what} is already in {self._checkpoint_file}.")
            _append(self._checkpoint_file, [json.dumps({'target': target, 'params': stored, 'seeded': source})])
        _add_trials(self._study, [(target, dict(coords), dict(specs))])

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

        if self._f is None:
            raise ValueError('maximize needs an objective; this optimizer was made for ask and tell only')
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
