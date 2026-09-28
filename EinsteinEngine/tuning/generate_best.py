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
out_params.
"""

import argparse
import json
import math
import runpy
import sys
from typing import Any

from EinsteinEngine.tuning import tuning
from EinsteinEngine.tuning.experiment import ParamType
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

    Non-finite targets (failed runs) are skipped. The checkpoint format is the
    JSON-lines format written by CheckpointedOptimizer: one
    ``{"target": float, "params": {...}}`` object per line.
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
            if not math.isfinite(target):
                continue
            if target > best_target:
                best_target = target
                best_params = entry['params']

    if best_params is None:
        raise RuntimeError(f"No finite target found in checkpoint file {checkpoint_file}.")
    return best_target, best_params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a recipe with its tuning in_params fixed to the best (max-target) values from a checkpoint file.")
    parser.add_argument("recipe", help="Path to the Einstein Engine recipe.")
    parser.add_argument("tuner", help="Path to the Python file providing the Tuner instance (the same file used with remote_tuner).")
    parser.add_argument("--checkpoint-file", type=str, default="split_tuning_checkpt.jsonl", help="Checkpoint file to read the best parameters from.")

    args = parser.parse_args()

    tuner_inst = load_tuner_from_file(args.tuner)
    experiment = tuner_inst.get_experiment()

    best_target, best_params = load_best_params(args.checkpoint_file)
    _, recipe_facing_args = experiment.suggest_params(_FixedTrial(best_params))

    print(f"Best target {best_target} found in {args.checkpoint_file}.")
    print(f"Fixed in_params: {best_params}")

    #  Pin the recipe's tuning params, then run it exactly as remote_feedback
    #  does: the recipe reads these via get_tuning_param, which resolves the
    #  module-global _tuning_params.
    tuning._tuning_params = recipe_facing_args
    sys.argv = [args.recipe]
    runpy.run_path(args.recipe, run_name="__main__")


if __name__ == "__main__":
    main()
