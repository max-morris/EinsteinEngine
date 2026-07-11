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
import typing
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

from EinsteinEngine.tuning.experiment import Experiment

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Sentinel returned to the optimizer for runs that fail outright (-inf).
# Large-but-finite so TPE still learns these are bad regions.
_FAILED_VALUE: float = -1e9

def _is_int_param(low: Any, high: Any) -> bool:
    """Return True when both bounds are plain ints (not float)."""
    return isinstance(low, int) and isinstance(high, int)


def _make_distributions(param_bounds: dict[str, tuple[int | float, int | float]]) -> dict[str, optuna.distributions.BaseDistribution]:
    distributions: dict[str, optuna.distributions.BaseDistribution] = {}
    for name, (low, high) in param_bounds.items():
        if _is_int_param(low, high):
            low = typing.cast(int, low)
            high = typing.cast(int, high)
            distributions[name] = optuna.distributions.IntDistribution(low, high)
        else:
            distributions[name] = optuna.distributions.FloatDistribution(float(low), float(high))
    return distributions


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
        n_startup_trials: int = 10,
        **_kwargs: Any,  # absorb legacy kwargs (acquisition_function, etc.)
    ) -> None:
        self._f = f
        self._verbose = verbose
        self._checkpoint_file = checkpoint_file
        self._experiment = experiment
        self._param_bounds = {param.name: param.bounds for param in self._experiment.in_params.values()}
        self._distributions = _make_distributions(self._param_bounds)

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
                trial_params = entry['params']
                trial_dists = _make_distributions({p: b for p, b in self._param_bounds.items() if p in trial_params})
                trial = optuna.trial.create_trial(
                    params=trial_params,
                    distributions=trial_dists,
                    value=target,
                )
                self._study.add_trial(trial)
                count += 1
        return count

    def _objective(self, trial: optuna.Trial) -> float:
        in_args, out_args = self._experiment.suggest_params(trial)
        assert self._f is not None

        # Capture the best value so far *before* this trial so we can detect
        # whether this trial sets a new record.
        try:
            prev_best = self._study.best_value
        except ValueError:
            prev_best = None

        result = self._f(**out_args)
        value = float(result) if np.isfinite(result) else _FAILED_VALUE

        if value != _FAILED_VALUE and (prev_best is None or value > prev_best):
            self._notify_new_best(value)

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

    @staticmethod
    def _notify_new_best(value: float) -> None:
        """Send a Telegram message announcing a new best target.

        Best-effort: failures to invoke telegram-send must never interrupt the
        optimization run.
        """
        message = f'New best time found: {abs(value)} seconds'
        try:
            subprocess.run(
                ['telegram-send', message],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            warnings.warn(f'Failed to send Telegram notification: {exc}')

    def maximize(self, warmup_iterations: int = 10, iterations: int = 20) -> None:
        """Run optimization, deducting already-loaded probes from the budget."""

        n = self._n_checkpoint_loaded
        total = warmup_iterations + iterations
        n_remaining = max(0, total - n)

        self._study.optimize(self._objective, n_trials=n_remaining)
