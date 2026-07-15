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

"""Plot target vs. parameter values from a bayes_checkpoint JSONL file.

Usage:
    python plot_tuning.py <checkpoint.json> [--out plot.png]
"""

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.tuning.remote_tuner import load_tuner_from_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def map_coords_to_values(records: list[dict[str, Any]], experiment: Experiment) -> list[dict[str, Any]]:
    """Return records with each ``params`` dict remapped from raw search
    coordinates to human-meaningful domain values via the experiment's domains.

    Checkpoints store the coordinates Optuna searched over; for reparameterized
    domains (e.g. Union) those differ from the values the recipe saw.  A record
    that can't be remapped (e.g. the experiment definition has since changed) is
    left with its raw coordinates so the plot still renders.
    """
    mapped: list[dict[str, Any]] = []
    for record in records:
        try:
            values = experiment.reconstruct_values(record["params"])
            mapped.append({**record, "params": dict(values)})
        except Exception as exc:
            warnings.warn(f"Could not map coordinates to values for a record; plotting raw coordinates: {exc}")
            mapped.append(record)
    return mapped


_FAILED_VALUE: float = -1e9


def plot(records: list[dict[str, Any]], title: str, out: Path | None) -> None:
    param_names: list[str] = sorted({k for r in records for k in r["params"]})
    n_params = len(param_names)
    steps = list(range(1, len(records) + 1))
    targets = [r["target"] for r in records]

    # Separate successful and failed runs.
    ok_mask = [t != _FAILED_VALUE for t in targets]
    ok_steps    = [s for s, ok in zip(steps, ok_mask) if ok]
    ok_targets  = [t for t, ok in zip(targets, ok_mask) if ok]
    fail_steps  = [s for s, ok in zip(steps, ok_mask) if not ok]

    running_max = []
    current_max = float("-inf")
    for t, ok in zip(targets, ok_mask):
        if ok:
            current_max = max(current_max, t)
        running_max.append(current_max if current_max != float("-inf") else None)
    ok_running_max_steps   = [s for s, v in zip(steps, running_max) if v is not None]
    ok_running_max_values  = [v for v in running_max if v is not None]

    # Y-axis limits derived from successful runs only.
    if ok_targets:
        y_min, y_max = min(ok_targets), max(ok_targets)
        y_range = y_max - y_min or 1.0
        # Reserve 10 % of the range at the bottom for failed-run markers.
        fail_y = y_min - 0.10 * y_range
        y_lo   = fail_y - 0.02 * y_range
        y_hi   = y_max + 0.05 * y_range
    else:
        fail_y, y_lo, y_hi = 0.0, -1.0, 1.0

    params: dict[str, list[float | None]] = {
        name: [r["params"].get(name) for r in records] for name in param_names
    }

    def _active_mask(name: str) -> list[bool]:
        return [v is not None for v in params[name]]

    # Layout: top row is the full-width target-vs-step plot;
    # below that, one subplot per parameter (param value vs target).
    n_cols = min(n_params, 5)
    n_param_rows = math.ceil(n_params / n_cols)
    n_rows = 1 + n_param_rows

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows), layout="constrained")
    fig.suptitle(title, fontsize=13)

    gs = fig.add_gridspec(n_rows, n_cols)

    # --- Target over iterations (spans all columns) ---
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(ok_steps, ok_targets, "o-", color="steelblue", alpha=0.6, label="target")
    ax_top.plot(ok_running_max_steps, ok_running_max_values, "--", color="tomato",
                linewidth=1.5, label="running max")
    if fail_steps:
        ax_top.plot(fail_steps, [fail_y] * len(fail_steps), "rx", markersize=8,
                    markeredgewidth=1.5, label="failed", zorder=4)
        ax_top.axhline(fail_y, color="salmon", linewidth=0.5, linestyle=":")
    ax_top.set_ylim(y_lo, y_hi)
    ax_top.set_xlabel("Iteration")
    ax_top.set_ylabel("Target")
    ax_top.set_title("Target over iterations")
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    # Colour probes by iteration so scatter plots show exploration order.
    cmap = cm.viridis
    colours = cmap(np.linspace(0, 1, len(records)))
    ok_colours   = [c for c, ok in zip(colours, ok_mask) if ok]
    fail_colours = [c for c, ok in zip(colours, ok_mask) if not ok]

    # --- Per-parameter scatter: param value vs target ---
    param_axes: list[matplotlib.axes.Axes] = []
    for idx, name in enumerate(param_names):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        active = _active_mask(name)
        ok_x      = [v for v, ok, a in zip(params[name], ok_mask, active) if ok and a and v is not None]
        ok_tgts   = [t for t, ok, a in zip(targets,       ok_mask, active) if ok and a]
        ok_cols   = [c for c, ok, a in zip(colours,       ok_mask, active) if ok and a]
        fail_x    = [v for v, ok, a in zip(params[name], ok_mask, active) if not ok and a and v is not None]
        fail_cols = [c for c, ok, a in zip(colours,       ok_mask, active) if not ok and a]

        if ok_x:
            ax.scatter(ok_x, ok_tgts, c=ok_cols, s=40, alpha=0.8, zorder=3)
        if fail_x:
            ax.scatter(fail_x, [fail_y] * len(fail_x), c=fail_cols,
                       marker="x", s=60, linewidths=1.5, alpha=0.8, zorder=4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(name)
        ax.set_ylabel("Target")
        ax.set_title(f"Target vs {name}")
        ax.grid(True, alpha=0.3)
        param_axes.append(ax)

    # Shared colourbar showing iteration order.
    sm = cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=1, vmax=len(records)))
    sm.set_array([])
    fig.colorbar(sm, ax=param_axes, label="Iteration", shrink=0.6)


    if out is not None:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bayes_checkpoint JSONL tuning results.")
    parser.add_argument("checkpoint", type=Path, help="Path to the JSONL checkpoint file.")
    parser.add_argument("--out", type=Path, default=None, help="Save plot to file instead of displaying it.")
    parser.add_argument("--tuner", type=str, default=None,
                        help="Path to the Tuner file (same one used with remote_tuner/generate_best). "
                             "When given, reparameterized params (e.g. Union domains) are plotted as their "
                             "actual values instead of the raw search coordinates stored in the checkpoint.")
    args = parser.parse_args()

    records = load_jsonl(args.checkpoint)
    if not records:
        print("No records found in checkpoint file.")
        return

    if args.tuner is not None:
        experiment = load_tuner_from_file(args.tuner).get_experiment()
        records = map_coords_to_values(records, experiment)

    plot(records, title=args.checkpoint.name, out=args.out)


if __name__ == "__main__":
    main()
