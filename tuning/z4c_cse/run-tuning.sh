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

# Measure the cost model's weights on the V100S nodes the trials run on, once per checkpoint; they are recorded in
# <checkpoint>.microbench.json (see docs/TUNING-QUICKSTART.md). MICROBENCH=0 tunes with the default weights instead,
# e.g. to resume a checkpoint started without measured ones.
microbench=(
    --microbench-remote-path /work/mmor115/ee-cse-bench/microbench/
    --microbench-command 'module load gcc/9.3.0 cuda/12.4.0 && CUDA_ARCH=70 sbatch -A hpc_smalltest2 -p gpu -N 1 -n 1 -c 48 --gres=gpu:1 -t 0:40:00 -o microbench.log run.sh cuda'
)
if [[ "${MICROBENCH:-1}" == 0 ]]; then
    microbench=()
fi

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
    "${microbench[@]}" \
    "$@"
