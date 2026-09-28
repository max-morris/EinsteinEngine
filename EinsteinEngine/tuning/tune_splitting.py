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
from typing import Any, Callable

from EinsteinEngine.intermediate.soft_split_retainment_predicate import SoftSplitRetainmentStrategy, retain_percentile

from EinsteinEngine.tuning.remote_feedback import RemoteFeedbackArgs
from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.tuning.tuning import Tuner, do_tuning


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the splitting of a recipe with the remote_feedback module.")
    parser.add_argument("recipe", help="Path to the Einstein Engine recipe.")
    parser.add_argument("--local-path", type=str, default="/home/max/src/EmitCactus/EinsteinEngine/tuning/Cottonmouth/", help="Local path containing generated code.")
    parser.add_argument("--remote-host", type=str, default="qbd", help="Remote host to which generated code should be copied.")
    parser.add_argument("--remote-path", type=str, default="/home/mmorris/project/Cottonmouth/", help="Remote path into which generated code should be copied.")
    parser.add_argument("--remote-cactus-path", type=str, default="/home/mmorris/project/Cactus/", help="Remote path containing the Cactus installation.")
    parser.add_argument("--remote-command", type=str, default="./build.sh && ./run-all.sh", help="Command to build and run on the remote machine.")
    parser.add_argument("--remote-timing-command", type=str, default="./timings.sh", help="Command to run timing on the remote machine. Must print a single number to optimize for.")
    parser.add_argument("--checkpoint-file", type=str, default="split_tuning_checkpt.jsonl", help="File to save/restore Bayesian optimization progress.")

    args: RemoteFeedbackArgs = parser.parse_args()

    do_tuning(args, CombinatorialSplitTuner(n_vars=15), args.checkpoint_file)


class CombinatorialSplitTuner(Tuner):
    def __init__(self, n_vars: int) -> None:
        self.n_vars = n_vars

    def get_experiment(self) -> Experiment:
        e = Experiment()

        for n in range(1, self.n_vars + 1):
            e.add_in_param(f'split_{n}', (0, 2))
            #  Python has completely insane lexical scoping, so we have to early-bind n in the lambda to capture by value
            e.add_in_param(f'soft_retain_percentile_{n}', (0.0, 1.0), lambda params, n=n: params[f'split_{n}'] == 1)

        def get_hard_split_predicate(params: dict[str, Any]) -> Callable[[int], bool]:
            return lambda i: int(params[f'split_{i}']) == 2

        def get_soft_split_predicate(params: dict[str, Any]) -> Callable[[int], bool|SoftSplitRetainmentStrategy]:
            def f(i: int) -> bool|SoftSplitRetainmentStrategy:
                if int(params[f'split_{i}']) != 1:
                    return False
                return retain_percentile(params[f'soft_retain_percentile_{i}'])
            return f

        e.add_out_param('auto_hard_split_predicate', get_hard_split_predicate)
        e.add_out_param('auto_soft_split_predicate', get_soft_split_predicate)

        return e


if __name__ == "__main__":
    main()

