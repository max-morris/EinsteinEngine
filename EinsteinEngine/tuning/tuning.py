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

import functools
import traceback
import typing
from abc import ABC, abstractmethod
from typing import Any

from EinsteinEngine.common.util import pprint

from EinsteinEngine.tuning.checkpointed_optimizer import CheckpointedOptimizer
from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.tuning.remote_feedback import RemoteFeedbackArgs, do_remote_run

_tuning_params: dict[str, Any] | None = None

def get_tuning_param[T](param_name: str, default: T) -> T:
    if _tuning_params is None:
        return default
    if param_name not in _tuning_params:
        raise RuntimeError(f"Tuning parameter {param_name} not found.")
    return typing.cast(T, _tuning_params[param_name])

class Tuner(ABC):
    @abstractmethod
    def get_experiment(self) -> Experiment:
        ...


def do_tuning[T: Tuner](args: RemoteFeedbackArgs, tuner: T, checkpoint_file: str, warmup_iterations: int = 10, iterations: int = 20, telegram_verbosity: int = 1) -> None:
    optimizer = CheckpointedOptimizer(
        f=functools.partial(do_tuning_run, args=args),
        experiment=tuner.get_experiment(),
        checkpoint_file=checkpoint_file,
        telegram_verbosity=telegram_verbosity,
    )

    if optimizer.n_checkpoint_loaded:
        pprint(f'Resumed from checkpoint: {optimizer.n_checkpoint_loaded} observations loaded from {checkpoint_file}')

    optimizer.maximize(warmup_iterations=warmup_iterations, iterations=iterations)
    assert optimizer.max is not None
    pprint(f'Tuning complete. Best target: {optimizer.max}')


def do_tuning_run(args: RemoteFeedbackArgs, **recipe_facing_args: dict[str, Any]) -> float:
    global _tuning_params
    _tuning_params = recipe_facing_args
    try:
        timing_value = do_remote_run(args, {})
    except RuntimeError as e:
        traceback.print_exception(e)
        # Probably created a bad split (empty loop, loop with no outputs) so discard this result
        return -float('inf')
    finally:
        _tuning_params = None

    return -timing_value  # Optimizer tries to maximize; lower time is better
