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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot(records: list[dict[str, Any]], title: str, out: Path | None) -> None:
    param_names: list[str] = sorted(records[0]["params"].keys())
    n_params = len(param_names)
    steps = list(range(1, len(records) + 1))
    targets = [r["target"] for r in records]
    running_max = [max(targets[: i + 1]) for i in range(len(targets))]

    params: dict[str, list[float]] = {
        name: [r["params"][name] for r in records] for name in param_names
    }

    # Layout: top row is the full-width target-vs-step plot;
    # below that, one subplot per parameter (param value vs target).
    n_cols = min(n_params, 3)
    n_param_rows = math.ceil(n_params / n_cols)
    n_rows = 1 + n_param_rows

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows), layout="constrained")
    fig.suptitle(title, fontsize=13)

    gs = fig.add_gridspec(n_rows, n_cols)

    # --- Target over iterations (spans all columns) ---
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(steps, targets, "o-", color="steelblue", alpha=0.6, label="target")
    ax_top.plot(steps, running_max, "--", color="tomato", linewidth=1.5, label="running max")
    ax_top.set_xlabel("Iteration")
    ax_top.set_ylabel("Target")
    ax_top.set_title("Target over iterations")
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    # Colour probes by iteration so scatter plots show exploration order.
    cmap = cm.viridis
    colours = cmap(np.linspace(0, 1, len(records)))

    # --- Per-parameter scatter: param value vs target ---
    param_axes: list[plt.Axes] = []
    for idx, name in enumerate(param_names):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.scatter(params[name], targets, c=colours, s=40, alpha=0.8, zorder=3)
        ax.set_xlabel(name)
        ax.set_ylabel("Target")
        ax.set_title(f"Target vs {name}")
        ax.grid(True, alpha=0.3)
        param_axes.append(ax)

    # Shared colourbar showing iteration order.
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=len(records)))
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
    args = parser.parse_args()

    records = load_jsonl(args.checkpoint)
    if not records:
        print("No records found in checkpoint file.")
        return

    plot(records, title=args.checkpoint.name, out=args.out)


if __name__ == "__main__":
    main()
