#!/bin/bash

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

set -euo pipefail

# Run a tuning search on a compute node instead of the login node.
#
# rostam1 is shared with a dozen other users, and a search puts two heavy things
# on whatever machine it runs on: several minutes of sympy per trial, and a
# parallel CUDA build. Both belong on an allocation. The timed runs are
# submitted as their own jobs from inside this one -- verified to work on
# rostam (a medusa job can sbatch to cuda-A100-amd and poll squeue).
#
#   ./submit-search.sh tuner_cuts.py split_tuning_cuts.jsonl
#   ./submit-search.sh tuner_cuts.py split_tuning_cuts.jsonl --warmup 10 --iterations 50
#
# Prints the job id; follow with logs/search-<jobid>.out.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

TUNER="${1:?usage: submit-search.sh <tuner.py> <checkpoint.jsonl> [--warmup N] [--iterations N]}"
CKPT="${2:?usage: submit-search.sh <tuner.py> <checkpoint.jsonl> [--warmup N] [--iterations N]}"
shift 2

WARMUP=10
ITERATIONS=50
BUILD_PARTITION=medusa
BUILD_CPUS=16
DRIVER_WALLTIME=24:00:00

while [ $# -gt 0 ]; do
    case "$1" in
        --warmup)     WARMUP="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --cpus)       BUILD_CPUS="$2"; shift 2 ;;
        --walltime)   DRIVER_WALLTIME="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${SCRIPT_DIR}/logs"

sbatch --parsable \
    -p "${BUILD_PARTITION}" -N 1 -n 1 -c "${BUILD_CPUS}" -t "${DRIVER_WALLTIME}" \
    -J "z4c-search" \
    -o "${SCRIPT_DIR}/logs/search-%j.out" \
    -e "${SCRIPT_DIR}/logs/search-%j.err" \
    --wrap "cd '${SCRIPT_DIR}' && \
            PYTHONPATH='${REPO_ROOT:-/home/sbrandt/repos/EinsteinEngine}' \
            PYTHONUNBUFFERED=1 \
            PARFILE_NAME='${PARFILE_NAME:-gauge_wave_z4c_bench}' \
            WALLTIME='${WALLTIME:-0:30:00}' \
            '${PYTHON}' -u -m EinsteinEngine.tuning.remote_tuner \
                '/home/sbrandt/repos/EinsteinEngine/recipes/Cottonmouth/Z4c.py' \
                '${SCRIPT_DIR}/${TUNER}' \
                --local-path '${SCRIPT_DIR}/Cottonmouth/' \
                --remote-host localhost \
                --remote-path '${STAGE_DIR}/' \
                --remote-cactus-path '${CACTUS_DIR}/' \
                --remote-command '${SCRIPT_DIR}/build-and-submit.sh' \
                --remote-timing-command '${SCRIPT_DIR}/timings.sh' \
                --checkpoint-file '${SCRIPT_DIR}/${CKPT}' \
                --warmup-iterations ${WARMUP} \
                --iterations ${ITERATIONS}"
