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

"""Checkpointing Bayesian optimization backed by Optuna + TPESampler.

Provides CheckpointedBayesianOptimization, a drop-in replacement for the
previous bayes_opt-based version.  The checkpoint file format (JSON-lines)
is preserved, so existing checkpoint files and plot_tuning.py are unaffected.

Optuna's TPESampler is used instead of a GP, which makes it robust to:
  - Non-smooth / discontinuous objective landscapes
  - Integer / combinatorial parameter spaces (e.g. BitTwiddleTuner)
  - Noisy objective values
  - scipy.optimize.NonlinearConstraint (converted to Optuna constraints)
"""

import json
import os
import subprocess
import time
import urllib.request
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from EinsteinEngine.tuning.experiment import CoordSpec, Experiment, InfeasibleParamError

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Sentinel returned to the optimizer for runs that fail outright (-inf).
# Large-but-finite so TPE still learns these are bad regions.
_FAILED_VALUE: float = -1e9

# Infeasible (pruned) trials don't consume the iteration budget, so cap total
# attempts at this multiple of the budget in case a constraint rejects nearly
# the entire search space.
_MAX_ATTEMPT_FACTOR: int = 100


def _distribution_of(spec: CoordSpec) -> optuna.distributions.BaseDistribution:
    """Build the Optuna distribution for a param's raw-coordinate CoordSpec."""
    if spec.kind is int:
        return optuna.distributions.IntDistribution(int(spec.lo), int(spec.hi))
    return optuna.distributions.FloatDistribution(float(spec.lo), float(spec.hi))


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
        if not os.path.exists(self._checkpoint_file):
            return 0
        count = 0
        with open(self._checkpoint_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry: dict[str, Any] = json.loads(line)
                raw_target: float = float(entry['target'])
                target = raw_target if np.isfinite(raw_target) else _FAILED_VALUE
                # Replay the stored raw coordinates through the experiment so
                # dynamic domains recover their exact per-trial coordinate ranges
                # (for plain Intervals a coordinate is just its value).
                coords, specs = self._experiment.reconstruct_coords(entry['params'])
                trial = optuna.trial.create_trial(
                    params=dict(coords),
                    distributions={name: _distribution_of(spec) for name, spec in specs.items()},
                    value=target,
                )
                self._study.add_trial(trial)
                count += 1
        return count

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
        value = float(result) if np.isfinite(result) else _FAILED_VALUE

        is_new_best = value != _FAILED_VALUE and (prev_best is None or value > prev_best)
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

        # Append to JSONL checkpoint (same format as before).
        rendered_in_args = {k: v for k, v in in_args.items()}
        entry: dict[str, Any] = {
            'target': value,
            'params': rendered_in_args,
        }
        with open(self._checkpoint_file, 'a') as fh:
            fh.write(json.dumps(entry) + '\n')

        if self._verbose >= 1:
            try:
                best = self._study.best_value
            except ValueError:
                best = float('nan')
            print(f'  trial {trial.number:4d} | value: {value:+.6f} | best: {best:+.6f} | {rendered_in_args}')

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
        if value == _FAILED_VALUE:
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
