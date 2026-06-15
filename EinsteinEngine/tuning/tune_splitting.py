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

from EinsteinEngine.common.util import pprint
from bayes_opt import BayesianOptimization

from EinsteinEngine.tuning.sum_of_cosines import sum_of_cosines

from EinsteinEngine.tuning.remote_feedback import RemoteFeedbackArgs, do_remote_run
from tuning.sum_of_cosines import OwnsZero


# optimizer = BayesianOptimization(
#         f=black_box_score,
#         pbounds={
#             "complexity_weight": (-5.0, 5.0),
#             "rarity_weight": (-0.0, 5.0),
#             "sqrt_rarity_weight": (-5.0, 5.0),
#             "peak_symbol_distance_weight": (-5.0, 0.0),
#             "avg_symbol_distance_weight": (-5.0, 0.0),
#             "symbol_reuse_weight": (0.0, 5.0)
#         },
#         #verbose=0
#     )
#
#     optimizer.maximize(init_points=exploration_iter, n_iter=optimization_iter)
#     assert optimizer.max is not None
#     pprint(f'Bayesian Optimization result: {optimizer.max}')

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

    args: RemoteFeedbackArgs = parser.parse_args()

    do_tuning(args)

def do_tuning(args: RemoteFeedbackArgs) -> None:
    order = 2

    p_bounds = dict()

    for degree in range(1, order + 1):
        for kind in ['soft', 'hard']:
            p_bounds[f'outer_{degree}_{kind}'] = (0.0, 5)
            p_bounds[f'inner_{degree}_{kind}'] = (0.01, 5)

    optimizer = BayesianOptimization(
        f=functools.partial(do_tuning_run, args=args, order=order),
        pbounds=p_bounds
    )

    optimizer.maximize(init_points=10, n_iter=20)
    assert optimizer.max is not None
    pprint(f'Bayesian Optimization result: {optimizer.max}')

# (outer|inner)_{degree}_(soft|hard)
def do_tuning_run(args: RemoteFeedbackArgs, order: int, **kwargs) -> float:
    hard_fn = OwnsZero(
        sum_of_cosines(
            [kwargs[f'outer_{degree}_hard'] for degree in range(1, order + 1)],
            [kwargs[f'inner_{degree}_hard'] for degree in range(1, order + 1)],
        )
    )

    soft_fn = OwnsZero(
        sum_of_cosines(
            [kwargs[f'outer_{degree}_soft'] for degree in range(1, order + 1)],
            [kwargs[f'inner_{degree}_soft'] for degree in range(1, order + 1)],
        )
    )

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

