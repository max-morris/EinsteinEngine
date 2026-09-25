#  Copyright (C) 2026 Steven R. Brandt and other Einstein Engine contributors.
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

"""Re-measure the top checkpoint configurations several times each.

The search reports one timing per configuration, but the top handful of Z4c
splits come in within a few tenths of a percent of each other -- at or below
the run-to-run spread. A single number cannot say which of them is actually
fastest, or whether the winner was simply the luckiest draw.

This replays chosen configurations through the same Experiment the tuner used
(so the recipe sees byte-identical parameters) and runs each one `--reps`
times, reporting mean, spread and the implied noise. It writes nothing to the
tuning checkpoint: these are repeat measurements of known points, and feeding
them back would bias the optimizer's surrogate.
"""

import argparse
import json
import math
import statistics
from typing import Any, Optional

from EinsteinEngine.tuning import tuning
from EinsteinEngine.tuning.generate_best import _FixedTrial
from EinsteinEngine.tuning.remote_feedback import do_remote_run
from EinsteinEngine.tuning.remote_tuner import load_tuner_from_file


def load_entries(checkpoint_file: str) -> list[dict[str, Any]]:
    """Every finite-target entry in the checkpoint, best first."""
    entries: list[dict[str, Any]] = []
    with open(checkpoint_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry: dict[str, Any] = json.loads(line)
            if math.isfinite(float(entry['target'])):
                entries.append(entry)
    entries.sort(key=lambda e: -float(e['target']))
    return entries


def structure(params: dict[str, Any], n: int = 13) -> str:
    """The discrete none/soft/hard pattern, as a compact digit string."""
    return ''.join(str(int(params.get(f'split_{i}', 0))) for i in range(1, n + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("tuner")
    parser.add_argument("--local-path", type=str, required=True)
    parser.add_argument("--remote-host", type=str, required=True)
    parser.add_argument("--remote-path", type=str, required=True)
    parser.add_argument("--remote-cactus-path", type=str, required=True)
    parser.add_argument("--remote-command", type=str, required=True)
    parser.add_argument("--remote-timing-command", type=str, required=True)
    parser.add_argument("--checkpoint-file", type=str, default="split_tuning_checkpt.jsonl")
    parser.add_argument("--reps", type=int, default=3, help="Measurements per configuration.")
    parser.add_argument("--top", type=int, default=2, help="How many of the best configurations to re-measure.")
    parser.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=False,
                        help="Also measure the unsplit recipe (no tuning params at all). "
                             "validate.sh passes --baseline; use --no-baseline to skip it when "
                             "you already have good baseline reps in the results file.")
    parser.add_argument("--results-file", type=str, default="validation.jsonl")
    args = parser.parse_args()

    experiment = load_tuner_from_file(args.tuner).get_experiment()
    entries = load_entries(args.checkpoint_file)
    if not entries:
        raise SystemExit(f"No finite entries in {args.checkpoint_file}")

    #  label -> recipe-facing args (None means "unsplit": leave _tuning_params
    #  unset so the recipe falls back to its get_tuning_param defaults).
    configs: list[tuple[str, Optional[dict[str, Any]], float]] = []
    if args.baseline:
        configs.append(("baseline/unsplit", None, float('nan')))
    for rank, entry in enumerate(entries[:args.top]):
        _, recipe_facing = experiment.suggest_params(_FixedTrial(entry['params']))
        label = f"rank{rank + 1} [{structure(entry['params'])}]"
        configs.append((label, dict(recipe_facing), -float(entry['target'])))

    results: dict[str, list[float]] = {}
    with open(args.results_file, "a") as out:
        for label, cfg_params, searched in configs:
            times: list[float] = []
            for rep in range(args.reps):
                print(f"\n=== {label} rep {rep + 1}/{args.reps} ===", flush=True)
                tuning._tuning_params = cfg_params
                try:
                    t = do_remote_run(args, {})
                except RuntimeError as exc:
                    print(f"  FAILED: {exc}", flush=True)
                    t = float('nan')
                finally:
                    tuning._tuning_params = None
                times.append(t)
                out.write(json.dumps({"label": label, "rep": rep, "time": t,
                                      "searched": searched}) + "\n")
                out.flush()
                print(f"  {label} rep {rep + 1}: {t:.3f} s", flush=True)
            results[label] = times

    print("\n\n================ VALIDATION ================")
    print(f"{'configuration':28s} {'searched':>9s} {'mean':>8s} {'sd':>7s} {'min':>8s} {'max':>8s}")
    for label, _, searched in configs:
        good = [t for t in results[label] if not math.isnan(t)]
        if not good:
            print(f"{label:28s} {'--':>9s}   all reps failed")
            continue
        mean = statistics.fmean(good)
        sd = statistics.stdev(good) if len(good) > 1 else 0.0
        s = f"{searched:9.3f}" if not math.isnan(searched) else f"{'--':>9s}"
        print(f"{label:28s} {s} {mean:8.3f} {sd:7.3f} {min(good):8.3f} {max(good):8.3f}")
    print("\nA configuration whose mean is within ~1 sd of another is not")
    print("distinguishable at this benchmark length.")


if __name__ == "__main__":
    main()
