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
import runpy
from typing import Protocol

from EinsteinEngine.tuning.remote_feedback import RemoteFeedbackArgs
from EinsteinEngine.tuning.tuning import Tuner, do_tuning


class RemoteTunerArgs(RemoteFeedbackArgs, Protocol):
    tuner: str
    warmup_iterations: int
    iterations: int


def load_tuner_from_file(path: str) -> Tuner:
    """Execute a standalone Python file and extract the Tuner instance it provides.

    The file is executed with runpy and may import anything it needs from
    EinsteinEngine. It must provide a Tuner instance in one of two ways:
    either define a function ``get_tuner()`` returning the instance, or assign
    the instance to a module-level variable named ``tuner``. The file may
    define its own Tuner subclass, but it may also instantiate an existing one
    (e.g. tune_splitting.CombinatorialSplitTuner).

    Unlike recipes, the file is not run as ``__main__``, so an
    ``if __name__ == "__main__":`` block in it will not fire.
    """
    module_globals = runpy.run_path(path)

    get_tuner = module_globals.get("get_tuner")
    tuner_inst = get_tuner() if callable(get_tuner) else module_globals.get("tuner")

    if not isinstance(tuner_inst, Tuner):
        raise RuntimeError(
            f"Tuner file {path} must define a get_tuner() function returning a Tuner instance, "
            f"or assign one to a module-level variable named 'tuner'.")
    return tuner_inst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the splitting of a recipe with the remote_feedback module.")
    parser.add_argument("recipe", help="Path to the Einstein Engine recipe.")
    parser.add_argument("tuner", help="Path to a Python file providing the Tuner instance, via a get_tuner() function or a module-level variable named 'tuner'.")
    parser.add_argument("--local-path", type=str, help="Local path containing generated code.")
    parser.add_argument("--remote-host", type=str, help="Remote host to which generated code should be copied.")
    parser.add_argument("--remote-path", type=str, help="Remote path into which generated code should be copied.")
    parser.add_argument("--remote-cactus-path", type=str, help="Remote path containing the Cactus installation.")
    parser.add_argument("--remote-command", type=str, default="./build.sh && ./run-all.sh", help="Command to build and run on the remote machine. Relative to Cactus path.")
    parser.add_argument("--remote-timing-command", type=str, default="./timings.sh", help="Command to print timing information on the remote machine. Relative to Cactus path.")
    parser.add_argument("--checkpoint-file", type=str, default="split_tuning_checkpt.jsonl", help="File to save/restore tuning progress.")
    parser.add_argument("--warmup-iterations", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations.")

    args: RemoteTunerArgs = parser.parse_args()

    tuner_inst = load_tuner_from_file(args.tuner)

    do_tuning(args, tuner_inst, args.checkpoint_file,
              warmup_iterations=args.warmup_iterations, iterations=args.iterations)

if __name__ == "__main__":
    main()

