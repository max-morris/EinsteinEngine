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

# Build ee_microbench for one target and write results.json next to this script.
# Usage: run.sh cpu|cuda|hip [ee_microbench args...]
# The tuner runs this on its target, directly or through sbatch (which runs a copy of the script, so the directory is
# taken from SLURM_SUBMIT_DIR when set). Makefile variables such as CUDA_ARCH and HIP_ARCH come from the environment.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"

target="${1:-}"
shift || true
case "${target}" in
    cpu) make -B ee_microbench ;;
    cuda) make -B cuda ;;
    hip) make -B hip ;;
    *) echo "usage: run.sh cpu|cuda|hip [ee_microbench args...]" >&2; exit 2 ;;
esac

rm -f results.json
./ee_microbench --json - "$@" > results.json.tmp
mv results.json.tmp results.json
