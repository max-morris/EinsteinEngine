#!/bin/bash

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

set -euo pipefail

# Tune the Z4c CSE optimizations via remote_tuner, on Deep Bayou. The remote command builds the synced thorn and
# submits a job that runs it once untimed, to warm up the GPU, and then times a run; the timing command prints that
# run's ODESolvers::Solve::rhs time.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"

# The recipes like to place their output into the working dir.
# By setting the working dir here, we avoid trashing "normal" recipe output with our tuner generations.
# This requires us to set PYTHONPATH to the repo root before invoking python.
cd "${SCRIPT_DIR}"

PYTHONPATH="${REPO_ROOT}" python -m EinsteinEngine.tuning.remote_tuner \
    "${REPO_ROOT}/recipes/Cottonmouth/Z4c.py" \
    "${SCRIPT_DIR}/tuner.py" \
    --local-path "${SCRIPT_DIR}/Cottonmouth/" \
    --remote-host db \
    --remote-path /work/mmor115/ee-cse-bench/staging/ \
    --remote-cactus-path /work/mmor115/ee-cse-bench \
    --remote-command ./tuner-command.sh \
    --remote-timing-command ./tuner-time.sh \
    --checkpoint-file "${SCRIPT_DIR}/cse_tuning_checkpt.jsonl" \
    "$@"
