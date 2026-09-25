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

# This is the --remote-command. remote_feedback.py invokes it as
#   cd ${CACTUS_DIR} && <this script>
# after rsyncing the trial's freshly generated thorn into the staging dir.
#
# It must end up printing simfactory's "Submit finished, job id is <N>" line,
# which is what the tuner parses to find the Slurm job to wait on.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

cd "${CACTUS_DIR}"

# Guard against running a stale binary: if setup-arrangement.sh has not been
# run (or restore-arrangement.sh has since undone it), the build below would
# silently compile the git checkout's thorn and every trial would time the
# same code.
LINK_TARGET=$(readlink -f "${ARRANGEMENT_LINK}")
EXPECTED=$(readlink -f "${STAGE_DIR}/CottonmouthZ4c4m")
if [ "${LINK_TARGET}" != "${EXPECTED}" ]; then
    echo "CottonmouthZ4c4m resolves to '${LINK_TARGET}', expected '${EXPECTED}'." >&2
    echo "Run ./setup-arrangement.sh first." >&2
    exit 1
fi

# NOTE: deliberately no `cp simfactory/mdb/runscripts/... configs/*/RunScript`
# here. run-queue.sh does that for its own CPU config, but configs/sim-a100's
# RunScript and SubmitScript are the cactus-a100 ones -- singularity, the
# /work autofs bind, --mpi=pmi2, UCX_TLS=^posix, --gpus-per-task/--gres.
# Overwriting them with rostam's generic pair breaks the GPU run.

# Splitting changes schedule.ccl as well as the sources, so the CST has to
# re-run; Cactus does that on its own when the .ccl files are newer.
#  Size the build to what Slurm actually gave us. The machine file says
#  makejobs=20, which is right for a whole node but oversubscribes a smaller
#  allocation -- and when this ran on the login node it was 20 parallel CUDA
#  compiles on a box shared with a dozen other users. nproc under-reports here
#  (it said 1 for an 8-CPU allocation), so trust SLURM_CPUS_PER_TASK.
BUILD_JOBS=${SLURM_CPUS_PER_TASK:-4}
./simfactory/bin/sim build "${CONFIG}" --machine="${MACHINE}" -j"${BUILD_JOBS}"

# simfactory refuses to create a simulation that already exists.
rm -rf "${SIM_BASEDIR:?}/${SIM_NAME:?}"

./simfactory/bin/sim create-submit "${SIM_NAME}" \
    --machine="${MACHINE}" \
    --configuration="${CONFIG}" \
    --parfile="${PARFILE}" \
    --queue="${QUEUE}" \
    --procs="${PROCS}" \
    --ppn="${PPN}" \
    --num-threads="${NUM_THREADS}" \
    --walltime="${WALLTIME}"
