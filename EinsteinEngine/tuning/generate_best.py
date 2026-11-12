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

"""Counterpart to remote_tuner.py.

Where remote_tuner searches the parameter space, generate_best takes the
best result already recorded in a checkpoint file and runs the recipe once
with the tuning in_params pinned to those maximizing values, producing the
best generated code locally (no remote build/run).

The tuner file is the same one used with remote_tuner: it supplies the
Experiment that maps the checkpoint's in_params to the recipe-facing
out_params. For a tuner that probes the recipe, the probe must match the one
recorded next to the checkpoint, if any; the best entry is checked against
the recorded probe before the (slow) probe runs. If the tuning run measured
the cost model's weights on its target, the recipe runs with those.
"""

import argparse
import json
import math
import os
import runpy
import sys
from contextlib import nullcontext
from typing import Any

from EinsteinEngine.generators.sympy_complexity import use_weights
from EinsteinEngine.tuning import tuning
from EinsteinEngine.tuning.checkpointed_optimizer import is_failed
from EinsteinEngine.tuning.experiment import ParamType
from EinsteinEngine.tuning.microbench import recorded_weights
from EinsteinEngine.tuning.remote_tuner import load_tuner_from_file


class _FixedTrial:
    """A TrialObj whose suggest_* methods return pre-recorded values.

    Passing this to Experiment.suggest_params replays a fixed set of in_params
    through the experiment's conditions and out_param mappings, yielding the
    same recipe-facing args a live trial with those values would have produced.
    """

    def __init__(self, params: dict[str, ParamType]) -> None:
        self._params = params

    def suggest_int(self, name: str, _lo: int, _hi: int, /) -> int:
        return int(self._params[name])

    def suggest_float(self, name: str, _lo: float, _hi: float, /) -> float:
        return float(self._params[name])


def load_best_params(checkpoint_file: str) -> tuple[float, dict[str, Any]]:
    """Return the (target, in_params) of the max-target entry in a checkpoint.

    Failed runs are skipped. The checkpoint format is the JSON-lines format
    written by CheckpointedOptimizer: one ``{"target": float, "params": {...}}``
    object per line.
    """
    best_target = -math.inf
    best_params: dict[str, Any] | None = None
    with open(checkpoint_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry: dict[str, Any] = json.loads(line)
            target = float(entry['target'])
            if is_failed(target):
                continue
            if target > best_target:
                best_target = target
                best_params = entry['params']

    if best_params is None:
        raise RuntimeError(f"No successful run found in checkpoint file {checkpoint_file}.")
    return best_target, best_params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a recipe with its tuning in_params fixed to the best (max-target) values from a checkpoint file.")
    parser.add_argument("recipe", help="Path to the Einstein Engine recipe.")
    parser.add_argument("tuner", help="Path to the Python file providing the Tuner instance (the same file used with remote_tuner).")
    parser.add_argument("--checkpoint-file", type=str, default="split_tuning_checkpt.jsonl", help="Checkpoint file to read the best parameters from.")

    args = parser.parse_args()

    weights = recorded_weights(args.checkpoint_file)
    with use_weights(weights) if weights is not None else nullcontext():
        generate_best(args.recipe, args.tuner, args.checkpoint_file)


def generate_best(recipe: str, tuner: str, checkpoint_file: str) -> None:
    tuner_inst = load_tuner_from_file(tuner)
    best_target, best_params = load_best_params(checkpoint_file)
    source = f"The best entry in {checkpoint_file}"
    if tuner_inst.probe_targets() and os.path.exists(tuning.probe_sidecar_path(checkpoint_file)):
        #  Fail fast, before probing, if the best entry does not fit the recorded probe.
        tuning.build_experiment(tuner_inst, None, checkpoint_file).check_declared(best_params, source)
    experiment = tuning.build_experiment(tuner_inst, recipe, checkpoint_file)
    experiment.check_declared(best_params, source)
    _, recipe_facing_args = experiment.suggest_params(_FixedTrial(best_params))

    print(f"Best target {best_target} found in {checkpoint_file}.")
    print(f"Fixed in_params: {best_params}")

    #  Pin the recipe's tuning params, then run it exactly as remote_feedback
    #  does: the recipe reads these via get_tuning_param, which resolves the
    #  module-global _tuning_params.
    tuning._tuning_params = recipe_facing_args
    sys.argv = [recipe]
    runpy.run_path(recipe, run_name="__main__")


if __name__ == "__main__":
    main()
