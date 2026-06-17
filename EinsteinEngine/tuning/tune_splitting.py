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

import argparse
import functools
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence, Optional

from scipy.optimize import NonlinearConstraint

from EinsteinEngine.common.util import pprint
from EinsteinEngine.tuning.bayes_checkpoint import CheckpointedBayesianOptimization

from EinsteinEngine.tuning.sum_of_cosines import sum_of_cosines

from EinsteinEngine.tuning.remote_feedback import RemoteFeedbackArgs, do_remote_run
from tuning.sum_of_cosines import OwnsZero

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the splitting of a recipe with the remote_feedback module.")
    parser.add_argument("recipe", help="Path to the Einstein Engine recipe.")
    parser.add_argument("--local-path", type=str, default="/home/max/src/EmitCactus/EinsteinEngine/tuning/Cottonmouth/", help="Local path containing generated code.")
    parser.add_argument("--remote-host", type=str, default="qbd", help="Remote host to which generated code should be copied.")
    parser.add_argument("--remote-path", type=str, default="/home/mmorris/project/Cottonmouth/", help="Remote path into which generated code should be copied.")
    parser.add_argument("--remote-cactus-path", type=str, default="/home/mmorris/project/Cactus/", help="Remote path containing the Cactus installation.")
    parser.add_argument("--remote-command", type=str, default="./build.sh && ./run-all.sh", help="Command to build and run on the remote machine.")
    parser.add_argument("--remote-timing-command", type=str, default="./timings.sh", help="Command to run timing on the remote machine.")
    parser.add_argument("--checkpoint-file", type=str, default="split_tuning_checkpt.jsonl", help="File to save/restore Bayesian optimization progress.")

    args: RemoteFeedbackArgs = parser.parse_args()

    #do_tuning(args, SumOfCosinesTuner(order=3), args.checkpoint_file)
    #do_tuning(args, BitTwiddleTuner(max_val=all_ones(15)), args.checkpoint_file)
    do_tuning(args, CombinatorialTuner(n_vars=15), args.checkpoint_file)

def all_ones(n: int) -> int:
    num = 1
    for _ in range(n - 1):
        num = (num << 1) + 1
    return num

class Tuner(ABC):
    @abstractmethod
    def get_p_bounds(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def get_hard_split_predicate(self, **kwargs: Any) -> Callable[[int], bool]:
        ...

    @abstractmethod
    def get_soft_split_predicate(self, **kwargs: Any) -> Callable[[int], bool]:
        ...

    def get_nonlinear_constraints(self) -> Optional[NonlinearConstraint]:
        return None

class CombinatorialTuner(Tuner):
    def __init__(self, n_vars: int) -> None:
        self.n_vars = n_vars

    def get_p_bounds(self) -> dict[str, Any]:
        p_bounds = dict()

        for n in range(1, self.n_vars + 1):
            p_bounds[f'split_{n}'] = (0, 2)
            p_bounds[f'split_{n}'] = (0, 2)

        return p_bounds

    def get_hard_split_predicate(self, **kwargs: Any) -> Callable[[int], bool]:
        return lambda i: int(kwargs[f'split_{i}']) == 2

    def get_soft_split_predicate(self, **kwargs: Any) -> Callable[[int], bool]:
        return lambda i: int(kwargs[f'split_{i}']) == 1

class BitTwiddleTuner(Tuner):
    def __init__(self, max_val: int) -> None:
        self.max_val = max_val

    def get_p_bounds(self) -> dict[str, Any]:
        return {
            'hard_int': (0, self.max_val),
            'soft_int': (0, self.max_val)
        }

    def get_hard_split_predicate(self, **kwargs: Any) -> Callable[[int], bool]:
        return lambda i: ((1 << i) & int(kwargs['hard_int'])) > 0

    def get_soft_split_predicate(self, **kwargs: Any) -> Callable[[int], bool]:
        return lambda i: ((1 << i) & int(kwargs['soft_int'])) > 0

    def get_nonlinear_constraints(self) -> NonlinearConstraint:
        def constraint_fn(hard_int: int, soft_int: int) -> int:
            return abs(int(hard_int) & int(soft_int))

        return NonlinearConstraint(constraint_fn, -0.5, 0.5)

class SumOfCosinesTuner(Tuner):
    def __init__(self, order: int) -> None:
        self.order = order

    def get_p_bounds(self) -> dict[str, Any]:
        p_bounds = dict()

        for degree in range(1, self.order + 1):
            for kind in ['soft', 'hard']:
                p_bounds[f'outer_{degree}_{kind}'] = (0.0, 5)
                p_bounds[f'inner_{degree}_{kind}'] = (0.01, 5)

        return p_bounds

    def get_hard_split_predicate(self, **kwargs) -> Callable[[int], bool]:
        return OwnsZero(
            sum_of_cosines(
                [kwargs[f'outer_{degree}_hard'] for degree in range(1, self.order + 1)],
                [kwargs[f'inner_{degree}_hard'] for degree in range(1, self.order + 1)],
            )
        )

    def get_soft_split_predicate(self, **kwargs: Any) -> Callable[[int], bool]:
        return OwnsZero(
            sum_of_cosines(
                [kwargs[f'outer_{degree}_soft'] for degree in range(1, self.order + 1)],
                [kwargs[f'inner_{degree}_soft'] for degree in range(1, self.order + 1)],
            )
        )

def do_tuning[T: Tuner](args: RemoteFeedbackArgs, tuner: T, checkpoint_file: str) -> None:
    optimizer = CheckpointedBayesianOptimization(
        f=functools.partial(do_tuning_run, tuner=tuner, args=args),
        pbounds=tuner.get_p_bounds(),
        constraint=tuner.get_nonlinear_constraints(),
        checkpoint_file=checkpoint_file,
    )

    if optimizer.n_checkpoint_loaded:
        pprint(f'Resumed from checkpoint: {optimizer.n_checkpoint_loaded} observations loaded from {checkpoint_file}')

    optimizer.maximize(init_points=100, n_iter=2000)
    assert optimizer.max is not None
    pprint(f'Bayesian Optimization result: {optimizer.max}')

# (outer|inner)_{degree}_(soft|hard)
def do_tuning_run[T: Tuner](args: RemoteFeedbackArgs, tuner: T, **kwargs) -> float:
    hard_fn = tuner.get_hard_split_predicate(**kwargs)
    soft_fn = tuner.get_soft_split_predicate(**kwargs)

    try:
        total_time, rhs_time = do_remote_run(args, {
            'split_tuning': True,
            'auto_hard_split_predicate': hard_fn,
            'auto_soft_split_predicate': soft_fn,
        })
    except RuntimeError as e:
        # Probably created a bad split (empty loop, loop with no outputs) so discard this result
        return -float('inf')

    return -rhs_time  # Optimizer tries to maximize; lower time is better

if __name__ == "__main__":
    main()

