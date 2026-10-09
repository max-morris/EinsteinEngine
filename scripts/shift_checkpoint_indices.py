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

"""Convert a 1-based split checkpoint to 0-based, into a new file.

Split params used to be named split_1..N and soft_retain_percentile_1..N; they are now 0-based. This rewrites each
split_<int> and soft_retain_percentile_<int> params key to <name>_<int-1>, and copies every other key unchanged:

    python scripts/shift_checkpoint_indices.py old_checkpt.jsonl new_checkpt.jsonl

A checkpoint may also have more split positions than the recipe has now. For example, the old Z4c tuner declared 15
positions, but the Z4c RHS has 13 add_eqn calls, so the positions after the last one were never queried.
--n-positions N drops the (shifted) keys with an index >= N, and reports how many of the dropped split_<int> keys
were a split (nonzero); such a split had no effect if the recipe never queried that position:

    python scripts/shift_checkpoint_indices.py old_checkpt.jsonl new_checkpt.jsonl --n-positions 13

The new file must not have a probe file (<checkpoint>.probe.json) yet. For a tuner that probes the recipe,
remote_tuner writes one when it first resumes from the new file, and generate_best probes the recipe without one.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

SPLIT_KEY_RE = re.compile(r'(split|soft_retain_percentile)_(\d+)')


@dataclass
class Counts:
    """Counts over the converted entries."""
    shifted: int = 0  # split keys, including the dropped ones
    dropped: int = 0  # split keys dropped by --n-positions
    dropped_splits: int = 0  # dropped split_<int> keys with a nonzero value


def shift_key(key: str) -> tuple[str, int] | None:
    """The shifted key and its new index, or None if ``key`` is not a split key."""
    if (m := SPLIT_KEY_RE.fullmatch(key)) is None:
        return None
    index = int(m.group(2)) - 1
    if index < 0:
        raise ValueError(f"Key {key!r} is already 0-based.")
    return f'{m.group(1)}_{index}', index


def shift_params(params: dict[str, Any], n_positions: int | None, counts: Counts) -> dict[str, Any]:
    """The shifted params, without the keys whose (shifted) index is >= ``n_positions``."""
    shifted: dict[str, Any] = {}
    for key, value in params.items():
        if (result := shift_key(key)) is None:
            shifted[key] = value
            continue
        counts.shifted += 1
        if n_positions is not None and result[1] >= n_positions:
            counts.dropped += 1
            if key.startswith('split_') and value != 0:
                counts.dropped_splits += 1
        else:
            shifted[result[0]] = value
    return shifted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="1-based checkpoint (JSON lines).")
    parser.add_argument("output", help="0-based checkpoint to write; neither it nor its probe file may exist.")
    parser.add_argument("--n-positions", type=int, default=None, metavar="N",
                        help="Drop shifted keys whose index is N or more (the recipe's number of split positions).")
    args = parser.parse_args()
    if args.n_positions is not None and args.n_positions < 0:
        parser.error("--n-positions must be non-negative.")
    #  The name of the probe file is EinsteinEngine.tuning.tuning.probe_sidecar_path(output); that module is not
    #  imported, because it imports optuna.
    for path in (args.output, f'{args.output}.probe.json'):
        if os.path.exists(path):
            parser.error(f"{path} already exists.")

    entries = []
    counts = Counts()
    with open(args.input) as src:
        for line_number, line in enumerate(src, start=1):
            if line.strip():
                entry = json.loads(line)
                try:
                    entry['params'] = shift_params(entry['params'], args.n_positions, counts)
                except ValueError as e:
                    parser.error(f"{args.input}:{line_number}: {e} Is the checkpoint already converted?")
                entries.append(entry)
    with open(args.output, 'x') as dst:
        dst.writelines(json.dumps(entry) + '\n' for entry in entries)

    message = f"Wrote {len(entries)} entries to {args.output}, shifting {counts.shifted} keys"
    if args.n_positions is not None:
        message += (f" and dropping {counts.dropped} keys with an index >= {args.n_positions}, "
                    f"{counts.dropped_splits} of which were splits (a nonzero split_<int>)")
    print(message + ".", file=sys.stderr)
    if counts.shifted == 0:
        print(f"Warning: {args.input} has no split_<int> or soft_retain_percentile_<int> keys, so {args.output} is "
              f"a copy of it. Is it a split checkpoint?", file=sys.stderr)


if __name__ == "__main__":
    main()
