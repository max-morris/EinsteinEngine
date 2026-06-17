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
import warnings
from collections.abc import Callable, Mapping
from typing import Any, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from scipy.optimize import NonlinearConstraint

# (dependent_param, control_param, active_value, default_value)
# The dependent param is only suggested when control_param == active_value;
# otherwise it is fixed at default_value and excluded from the Optuna trial,
# shrinking the effective search space.
ConditionalParam = tuple[str, str, Any, float]

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Sentinel returned to the optimizer for runs that fail outright (-inf).
# Large-but-finite so TPE still learns these are bad regions.
_FAILED_VALUE: float = -1e9


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_int_param(low: Any, high: Any) -> bool:
    """Return True when both bounds are plain ints (not float)."""
    return isinstance(low, int) and isinstance(high, int)


def _make_distributions(pbounds: dict[str, tuple]) -> dict[str, optuna.distributions.BaseDistribution]:
    distributions: dict[str, optuna.distributions.BaseDistribution] = {}
    for name, (low, high) in pbounds.items():
        if _is_int_param(low, high):
            distributions[name] = optuna.distributions.IntDistribution(low, high)
        else:
            distributions[name] = optuna.distributions.FloatDistribution(float(low), float(high))
    return distributions


def _coerce_params(pbounds: dict[str, tuple], raw: dict[str, Any]) -> dict[str, Any]:
    """Cast raw param values (e.g. floats from JSON) to the right Python type."""
    result: dict[str, Any] = {}
    for name, val in raw.items():
        if name in pbounds:
            low, high = pbounds[name]
            result[name] = int(round(val)) if _is_int_param(low, high) else float(val)
        else:
            result[name] = float(val)
    return result


def _suggest_params(
    trial: optuna.Trial,
    pbounds: dict[str, tuple],
    conditional_params: list[ConditionalParam] | None = None,
) -> dict[str, Any]:
    """Suggest all parameters for *trial*, respecting conditional dependencies.

    For each ``ConditionalParam`` ``(dependent, control, active_value, default)``:
    - If the already-suggested value of *control* equals *active_value*, suggest
      *dependent* normally from its pbounds range.
    - Otherwise, skip suggesting *dependent* in Optuna (so TPE ignores it) and
      return *default* to the objective function.

    Parameters are suggested in two passes so that every control param is
    resolved before its dependents, regardless of dict insertion order.
    """
    dependents: dict[str, tuple[str, Any, float]] = {}  # name -> (control, active_value, default)
    if conditional_params:
        for dep, ctrl, act, dflt in conditional_params:
            dependents[dep] = (ctrl, act, dflt)

    params: dict[str, Any] = {}

    # Pass 1: suggest all non-dependent params (includes all control params).
    for name, (low, high) in pbounds.items():
        if name in dependents:
            continue
        if _is_int_param(low, high):
            params[name] = trial.suggest_int(name, low, high)
        else:
            params[name] = trial.suggest_float(name, float(low), float(high))

    # Pass 2: suggest dependent params only when their condition is met.
    for name, (ctrl, act, dflt) in dependents.items():
        if params.get(ctrl) == act:
            low, high = pbounds[name]
            if _is_int_param(low, high):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, float(low), float(high))
        else:
            # Condition not met: fix to default without entering Optuna's model.
            params[name] = dflt

    return params


def _effective_trial_data(
    params: dict[str, Any],
    distributions: dict[str, optuna.distributions.BaseDistribution],
    conditional_params: list[ConditionalParam] | None,
) -> tuple[dict[str, Any], dict[str, optuna.distributions.BaseDistribution]]:
    """Strip conditional dependent params from *params* and *distributions* when
    their condition is not met, mirroring the behaviour of ``_suggest_params``."""
    if not conditional_params:
        return params, distributions

    filtered_params = dict(params)
    filtered_dists = dict(distributions)
    for dep, ctrl, act, _dflt in conditional_params:
        if ctrl in params and params[ctrl] != act:
            filtered_params.pop(dep, None)
            filtered_dists.pop(dep, None)

    return filtered_params, filtered_dists


def _compute_constraint_violations(
    constraint: Optional[NonlinearConstraint],
    params: dict[str, Any],
) -> list[float]:
    """Evaluate constraint violations for a given params dict.

    Returns the same list of floats that constraints_func would return for an
    Optuna trial with those params.  Used when loading trials from JSONL so
    that system_attrs["constraints"] is populated and Optuna doesn't warn.
    """
    if constraint is None:
        return []
    value = float(constraint.fun(**params))
    violations: list[float] = []
    if constraint.lb > -np.inf:
        violations.append(float(constraint.lb) - value)
    if constraint.ub < np.inf:
        violations.append(value - float(constraint.ub))
    return violations


def _make_constraints_func(constraint: Optional[NonlinearConstraint]):
    """Convert a scipy NonlinearConstraint to an Optuna constraints_func.

    Optuna considers a trial feasible when all returned values are <= 0.
    """
    if constraint is None:
        return None

    def constraints_func(trial: optuna.trial.FrozenTrial) -> list[float]:
        value = float(constraint.fun(**trial.params))
        violations: list[float] = []
        if constraint.lb > -np.inf:
            violations.append(float(constraint.lb) - value)   # must be >= lb
        if constraint.ub < np.inf:
            violations.append(value - float(constraint.ub))   # must be <= ub
        return violations

    return constraints_func


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class CheckpointedBayesianOptimization:
    """Optuna-backed Bayesian optimization that checkpoints every probe to a
    JSON-lines file and resumes from it on construction.

    Parameters
    ----------
    f : callable
        Objective to **maximise**.  Called with keyword arguments matching
        pbounds keys.  Return -inf (or any non-finite value) for invalid runs.
    pbounds : mapping
        ``{name: (low, high)}``.  When both bounds are plain ``int`` values
        the parameter is treated as integer (``suggest_int``); otherwise it
        is treated as continuous (``suggest_float``).
    checkpoint_file : str
        Path to the JSON-lines checkpoint file.  Appended on every probe;
        loaded on construction so interrupted runs resume automatically.
    constraint : NonlinearConstraint, optional
        Constraint function called with the same keyword arguments as *f*.
        Infeasible trials are down-weighted in the TPE model.
    random_state : int, optional
        Seed for the TPE sampler.
    verbose : int
        0 = silent, 1 = one line per trial, 2 = one line per trial + header.
    n_startup_trials : int
        Random exploration trials before TPE starts modelling (default 10).
    """

    def __init__(
        self,
        f: Callable[..., float] | None,
        pbounds: Mapping[str, tuple[float, float]],
        checkpoint_file: str,
        constraint: NonlinearConstraint | None = None,
        random_state: int | None = None,
        verbose: int = 2,
        n_startup_trials: int = 10,
        conditional_params: list[ConditionalParam] | None = None,
        **_kwargs: Any,  # absorb legacy kwargs (acquisition_function, etc.)
    ) -> None:
        self._f = f
        self._pbounds = dict(pbounds)
        self._constraint = constraint
        self._verbose = verbose
        self._checkpoint_file = checkpoint_file
        self._conditional_params = conditional_params or []
        self._distributions = _make_distributions(self._pbounds)

        constraints_func = _make_constraints_func(constraint)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
            sampler = TPESampler(
                constraints_func=constraints_func,
                n_startup_trials=n_startup_trials,
                seed=random_state,
            )
        self._study = optuna.create_study(direction='maximize', sampler=sampler)
        self._n_checkpoint_loaded = self._load_checkpoint()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

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
                params = _coerce_params(self._pbounds, entry['params'])
                violations = _compute_constraint_violations(self._constraint, params)
                system_attrs = {'constraints': violations} if violations else {}
                trial_params, trial_dists = _effective_trial_data(
                    params, self._distributions, self._conditional_params
                )
                trial = optuna.trial.create_trial(
                    params=trial_params,
                    distributions=trial_dists,
                    value=target,
                    system_attrs=system_attrs,
                )
                self._study.add_trial(trial)
                count += 1
        return count

    def _objective(self, trial: optuna.Trial) -> float:
        params = _suggest_params(trial, self._pbounds, self._conditional_params)
        assert self._f is not None
        result = self._f(**params)
        value = float(result) if np.isfinite(result) else _FAILED_VALUE

        # Append to JSONL checkpoint (same format as before).
        entry: dict[str, Any] = {
            'target': value,
            'params': {k: float(v) for k, v in params.items()},
        }
        with open(self._checkpoint_file, 'a') as fh:
            fh.write(json.dumps(entry) + '\n')

        if self._verbose >= 1:
            try:
                best = self._study.best_value
            except ValueError:
                best = float('nan')
            print(f'  trial {trial.number:4d} | value: {value:+.6f} | best: {best:+.6f} | {params}')

        return value

    def maximize(self, init_points: int = 5, n_iter: int = 25) -> None:
        """Run optimization, deducting already-loaded probes from the budget.

        Parameters
        ----------
        init_points, n_iter : int
            *Total* budget including any probes already in the checkpoint.
        """
        n = self._n_checkpoint_loaded
        total = init_points + n_iter
        n_remaining = max(0, total - n)

        self._study.optimize(self._objective, n_trials=n_remaining)
