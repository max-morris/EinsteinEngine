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

"""Checkpointing support for BayesianOptimization (bayes_opt 3.x).

Provides CheckpointedBayesianOptimization, a drop-in subclass that
automatically saves every probe to a JSON-lines file and resumes from
it on construction.
"""

import json
import os
from collections.abc import Callable, Mapping
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from bayes_opt import BayesianOptimization
from bayes_opt.logger import ScreenLogger
from numpy.random import RandomState
from scipy.optimize import NonlinearConstraint

from bayes_opt.acquisition import AcquisitionFunction
from bayes_opt.domain_reduction import DomainTransformer

if TYPE_CHECKING:
    from bayes_opt.parameter import BayesParameter


class _CheckpointLogger(ScreenLogger):
    """ScreenLogger that also appends each probe to a JSON-lines file."""

    def __init__(self, checkpoint_file: str, verbose: int = 2, is_constrained: bool = False) -> None:
        super().__init__(verbose=verbose, is_constrained=is_constrained)
        self._checkpoint_file = checkpoint_file
        self._writing_enabled = False

    def enable_writing(self) -> None:
        self._writing_enabled = True

    def log_optimization_step(
        self,
        keys: list[str],
        res: dict[str, Any],
        params_config: Mapping[str, "BayesParameter"],
        current_max: dict[str, Any] | None,
    ) -> None:
        super().log_optimization_step(keys, res, params_config, current_max)
        if not self._writing_enabled:
            return
        entry: dict[str, Any] = {
            "target": float(res["target"]),
            "params": {k: float(v) for k, v in res["params"].items()},
        }
        if "constraint" in res:
            cv: NDArray[Any] | float = res["constraint"]
            entry["constraint"] = cv.tolist() if isinstance(cv, np.ndarray) else cv
        with open(self._checkpoint_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


class CheckpointedBayesianOptimization(BayesianOptimization):
    """BayesianOptimization that checkpoints every probe to a JSON-lines file.

    Parameters
    ----------
    checkpoint_file : str
        Path to the checkpoint file. Appended to on every probe; loaded on
        construction if it already exists so that interrupted runs resume
        automatically.
    All other parameters are forwarded to BayesianOptimization.

    Usage
    -----
    Replace BayesianOptimization with CheckpointedBayesianOptimization and
    pass checkpoint_file. Call maximize(init_points, n_iter) with the *total*
    budget; already-completed probes are subtracted automatically.

        optimizer = CheckpointedBayesianOptimization(
            f=my_fn, pbounds=bounds, checkpoint_file="run.json"
        )
        optimizer.maximize(init_points=10, n_iter=20)
    """

    def __init__(
        self,
        f: Callable[..., float] | None,
        pbounds: Mapping[str, tuple[float, float]],
        checkpoint_file: str,
        acquisition_function: AcquisitionFunction | None = None,
        constraint: NonlinearConstraint | None = None,
        random_state: int | RandomState | None = None,
        verbose: int = 2,
        bounds_transformer: DomainTransformer | None = None,
        allow_duplicate_points: bool = False,
    ) -> None:
        super().__init__(
            f=f,
            pbounds=pbounds,
            acquisition_function=acquisition_function,
            constraint=constraint,
            random_state=random_state,
            verbose=verbose,
            bounds_transformer=bounds_transformer,
            allow_duplicate_points=allow_duplicate_points,
        )

        self._checkpoint_file = checkpoint_file
        self._checkpoint_logger = _CheckpointLogger(
            checkpoint_file=checkpoint_file,
            verbose=self.logger.verbose,
            is_constrained=self.is_constrained,
        )
        self.logger = self._checkpoint_logger

        self._n_checkpoint_loaded = self._load_checkpoint()
        self._checkpoint_logger.enable_writing()

    @property
    def n_checkpoint_loaded(self) -> int:
        """Number of observations restored from the checkpoint file."""
        return self._n_checkpoint_loaded

    def _load_checkpoint(self) -> int:
        if not os.path.exists(self._checkpoint_file):
            return 0
        count = 0
        with open(self._checkpoint_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry: dict[str, Any] = json.loads(line)
                raw_cv: list[float] | float | None = entry.get("constraint")
                cv: NDArray[Any] | None = np.array(raw_cv) if raw_cv is not None else None
                self.register(params=entry["params"], target=entry["target"], constraint_value=cv)
                count += 1
        return count

    def maximize(self, init_points: int = 5, n_iter: int = 25) -> None:
        """Maximize, subtracting already-completed probes from the budget.

        Parameters
        ----------
        init_points, n_iter : int
            *Total* budget (including any probes already loaded from the
            checkpoint). Already-completed probes are subtracted automatically.
        """
        n = self._n_checkpoint_loaded
        total = init_points + n_iter
        remaining = max(0, total - n)
        actual_init = max(0, init_points - n)
        actual_iter = remaining - actual_init
        super().maximize(init_points=actual_init, n_iter=actual_iter)
