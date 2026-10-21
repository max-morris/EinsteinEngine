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
the recorded probe before the (slow) probe runs.

With --batch FILE --trial I it runs one trial of a batch instead (see
CheckpointedOptimizer.ask), for baking a batch's trials. That uses the probe
recorded next to the checkpoint and does not probe the recipe.
"""

import argparse
import json
import math
import os
import runpy
import sys
from typing import Any

from EinsteinEngine.tuning import tuning
from EinsteinEngine.tuning.checkpointed_optimizer import Batch, is_failed, study_digest
from EinsteinEngine.tuning.experiment import FixedTrial
from EinsteinEngine.tuning.remote_tuner import load_tuner_from_file


#  Kept for callers of the old private name.
_FixedTrial = FixedTrial


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
        description="Run a recipe with its tuning in_params fixed to the best (max-target) values from a checkpoint "
                    "file, or to one trial of a batch (CheckpointedOptimizer.ask).",
        epilog="Arguments after -- are passed to the recipe (its sys.argv[1:]). They must be the ones the probe file "
               "records (the probe ran the recipe with them), since they can change the equations; to choose "
               "where a recipe writes its output, run generate_best in that directory instead.")
    parser.add_argument("recipe", help="Path to the Einstein Engine recipe.")
    parser.add_argument("tuner", help="Path to the Python file providing the Tuner instance (the same file used with remote_tuner).")
    parser.add_argument("--checkpoint-file", type=str, default="split_tuning_checkpt.jsonl", help="Checkpoint file to read the best parameters from.")
    parser.add_argument("--batch", type=str, help="A batch file (Batch.to_json) to take the in_params from, instead of the best entry. The checkpoint's probe file is used and the recipe is not probed, so the probe file must be present.")
    parser.add_argument("--trial", type=int, help="With --batch: the index of the trial to run.")

    #  Arguments after -- are the recipe's own (its sys.argv[1:]), e.g. its output directory.
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    recipe_args = argv[split + 1:]
    args = parser.parse_args(argv[:split])
    if (args.batch is None) != (args.trial is None):
        parser.error("--batch and --trial go together")

    tuner_inst = load_tuner_from_file(args.tuner)
    if args.batch is not None:
        with open(args.batch) as fh:
            batch = Batch.from_json(fh.read())
        if not 0 <= args.trial < len(batch.params):
            parser.error(f"--trial must be in 0..{len(batch.params) - 1}")
        experiment = tuning.build_experiment(tuner_inst, None, args.checkpoint_file, recipe_args=recipe_args)
        if batch.study != study_digest(experiment, args.checkpoint_file):
            sys.exit(f"Batch {batch.id} was asked for another study than {args.checkpoint_file} (another checkpoint "
                     f"name, probe file or set of in-params).")
        fixed = batch.params[args.trial]
        recipe_facing_args = experiment.recipe_facing_args(fixed, f"Trial {args.trial} of batch {batch.id}")
        print(f"Trial {args.trial} of batch {batch.id}.")
    else:
        best_target, fixed = load_best_params(args.checkpoint_file)
        source = f"The best entry in {args.checkpoint_file}"
        if tuner_inst.probe_targets() and os.path.exists(tuning.probe_sidecar_path(args.checkpoint_file)):
            #  Fail fast, before probing, if the best entry does not fit the recorded probe.
            tuning.build_experiment(tuner_inst, None, args.checkpoint_file,
                                    recipe_args=recipe_args).check_declared(fixed, source)
        experiment = tuning.build_experiment(tuner_inst, args.recipe, args.checkpoint_file, recipe_args=recipe_args)
        recipe_facing_args = experiment.recipe_facing_args(fixed, source)
        print(f"Best target {best_target} found in {args.checkpoint_file}.")
    print(f"Fixed in_params: {fixed}")

    #  Pin the recipe's tuning params, then run it exactly as remote_feedback
    #  does: the recipe reads these via get_tuning_param, which resolves the
    #  module-global _tuning_params.
    tuning._tuning_params = recipe_facing_args
    sys.argv = [args.recipe, *recipe_args]
    runpy.run_path(args.recipe, run_name="__main__")


if __name__ == "__main__":
    main()
