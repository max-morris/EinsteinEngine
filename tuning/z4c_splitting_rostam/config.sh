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

# Single source of truth for every path and Slurm setting the tuning run uses.
# Sourced by setup-arrangement.sh, build-and-submit.sh, timings.sh,
# run-tuning.sh and restore-arrangement.sh.

# Where Cactus lives, and where simfactory puts simulations.
CACTUS_DIR=/work/sbrandt/etk/Cactus
SIM_BASEDIR=/work/sbrandt/simulations

# Staging dir the tuner rsyncs each trial's generated thorn into.
#
# Two constraints pin this location:
#
#   * It must NOT be ${CACTUS_DIR}/repos/Cottonmouth. That is a git checkout
#     (branch ET_2026_05), and remote_feedback.py rsyncs with --delete, which
#     would take .git and the other three thorns with it.
#   * It must live INSIDE ${CACTUS_DIR}. cactus-a100.ini builds with
#     `singularity exec ... make` and passes no --bind, so only $PWD (the
#     Cactus dir) is visible inside the build container. /work is an autofs
#     indirect map -- see the long note in cactus-a100.run -- so a staging dir
#     anywhere else under /work simply is not there when CST runs, and the
#     build dies with "Missing file .../param.ccl".
#
# So: a sibling of the Cottonmouth checkout, inside the tree, owned entirely
# by the tuner.
STAGE_DIR=${CACTUS_DIR}/repos/CottonmouthTuning

# The symlink we retarget, and the two things it points at. Both targets are
# relative, matching the other three Cottonmouth links, so they resolve the
# same way inside the container as outside it regardless of bind paths.
ARRANGEMENT_LINK=${CACTUS_DIR}/arrangements/Cottonmouth/CottonmouthZ4c4m
ARRANGEMENT_LINK_TUNING=../../repos/CottonmouthTuning/CottonmouthZ4c4m
ARRANGEMENT_LINK_ORIG=../../repos/Cottonmouth/CottonmouthZ4c4m

# Build/run settings.
#
# These follow ${CACTUS_DIR}/run-hpct-a100.sh, which is the working reference
# for running on rostam's A100 nodes. Three things there are load-bearing:
#
#   * MACHINE. Config sim-a100 belongs to the simfactory machine cactus-a100
#     (optionlist cactus-cuda-mpich.cfg, MPI_DIR=/opt/mpi/mpich), not to
#     rostam. cactus-a100.ini builds through
#     `singularity exec docker://stevenrbrandt/cactus-cuda make`, so every
#     sim invocation has to pass --machine, or simfactory picks rostam and
#     builds/launches the wrong way.
#   * The config's RunScript and SubmitScript are NOT to be overwritten with
#     the generic rostam.run/rostam.sub. cactus-a100.run carries the
#     singularity invocation, the /work autofs bind, --mpi=pmi2 and
#     UCX_TLS=^posix; cactus-a100.sub carries --gpus-per-task/--gres. Copying
#     rostam's over them (as run-queue.sh does for its own CPU config) breaks
#     GPU runs outright.
#   * Node shape. simfactory rejects --ppn above the machine's declared ppn
#     ("Illegal number of requested cores per node ... max-ppn is 32"), so the
#     full-node 128/64 variant noted in run-hpct-a100.sh's comments is not
#     reachable without editing cactus-a100.ini. 32/16 is what that script
#     actually defaults to, and it still gives 2 ranks with one A100 each --
#     the benchmark is GPU-bound, so the unused host cores cost nothing.
CONFIG=sim-a100
MACHINE=cactus-a100

# cuda-A100-amd rather than cuda-A100: the latter spans nasrin (2x A100-80GB,
# 128 cores) and toranj (4x A100-40GB, 64 cores). Which node Slurm happened to
# give you would move the measured time by more than many of the splits do,
# and the optimizer is fitting a surrogate to exactly these numbers. The
# -amd sub-partition is just the nasrin pair, so every trial is measured on
# identical hardware.
QUEUE=cuda-A100-amd

# One rank on one A100, deliberately.
#
# Two ranks on a node abort in UCX before the first iteration:
#   cma_ep.c:81 process_vm_readv(...) returned -1: Operation not permitted
# Each rank runs in its own singularity instance, so the ranks are in separate
# PID namespaces and CMA's cross-process reads are denied. cactus-a100.run
# already sets UCX_TLS=^posix for the sibling shared-memory failure, but that
# just falls back to cma, which is what fails here. Forcing tcp instead
# (UCX_TLS=^posix,cma) would work but puts a slow intra-node transport inside
# the thing being measured.
#
# A single rank sidesteps it and is the better benchmark anyway: the split
# being tuned changes a local kernel, so taking MPI out of the objective
# removes a whole source of run-to-run variance rather than averaging over it.
NODES=1
PPN=16
NUM_THREADS=16  # -> PPN/NUM_THREADS = 1 rank, 1 A100, 16 host threads.
PROCS=$((PPN * NODES))

# The benchmark is a couple of minutes; a short walltime also schedules sooner.
WALLTIME=${WALLTIME:-0:30:00}

SIM_NAME=z4c-tune
#  Overridable so a one-off run can use a different benchmark (e.g. the 256^3
#  variant) without editing this file. timings.sh reads PARFILE_NAME to find
#  the output directory, so the two must be set together.
PARFILE_NAME=${PARFILE_NAME:-gauge_wave_z4c_bench}

# The benchmark parfile must live OUTSIDE ${STAGE_DIR}: remote_feedback.py
# rsyncs --delete into that dir, and the generated tree contains only
# CottonmouthZ4c4m, so anything else parked there is deleted on the first trial.
PARFILE_DIR=/work/sbrandt/etk/tuning
PARFILE=${PARFILE_DIR}/${PARFILE_NAME}.par

# The interpreter used to drive the tuner. rostam's default `python` is 3.9,
# but EinsteinEngine needs >= 3.13, so prefer the repo venv when it exists.
# Override by exporting EE_PYTHON before invoking the scripts.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -n "${EE_PYTHON:-}" ]; then
    PYTHON="${EE_PYTHON}"
elif [ -x "${_REPO_ROOT}/venv/bin/python" ]; then
    PYTHON="${_REPO_ROOT}/venv/bin/python"
else
    PYTHON=python
fi
