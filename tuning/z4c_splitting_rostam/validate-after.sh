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

set -uo pipefail

# Wait for any running search to finish, then re-measure its best configuration
# several times.
#
# The wait is not politeness: a search and a validation share ${STAGE_DIR} and
# the Slurm simulation name, so overlapping them has each clobbering the
# other's generated code and timing output.
#
#   ./validate-after.sh tuner_update_rarity.py split_tuning_update_rarity.jsonl
#   ./validate-after.sh tuner.py split_tuning_checkpt.jsonl 5
#
# The tuner file must be the one that produced the checkpoint -- it supplies
# the ordering_fn the trials were measured under. Validating against a
# different tuner silently re-measures something else.
#
# Any extra environment the tuner needs (e.g. EE_INFLIGHT_K) must be exported
# by the caller; it is passed through.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

TUNER="${1:?usage: validate-after.sh <tuner.py> <checkpoint.jsonl> [reps] [top]}"
CKPT="${2:?usage: validate-after.sh <tuner.py> <checkpoint.jsonl> [reps] [top]}"
REPS="${3:-3}"
TOP="${4:-1}"

cd "${SCRIPT_DIR}"

echo "waiting for the running search to finish..."
while pgrep -f "EinsteinEngine.tuning.remote_tuner" >/dev/null 2>&1; do
    sleep 60
done
echo "search finished at $(date)"

if [ ! -s "${CKPT}" ]; then
    echo "no checkpoint at ${CKPT}; nothing to validate" >&2
    exit 1
fi

BASE=$(basename "${CKPT}" .jsonl)
RESULTS="${SCRIPT_DIR}/validation_${BASE#split_tuning_}.jsonl"

echo "validating top ${TOP} of $(basename "${CKPT}") with ${REPS} reps -> $(basename "${RESULTS}")"

TUNER_FILE="${SCRIPT_DIR}/${TUNER}" \
CHECKPOINT_FILE="${SCRIPT_DIR}/${CKPT}" \
RESULTS_FILE="${RESULTS}" \
    ./validate.sh --no-baseline --top "${TOP}" --reps "${REPS}"

RC=$?
if [ "$RC" -eq 0 ]; then
    telegram-send "Z4c validation of ${BASE} finished; see $(basename "${RESULTS}")" >/dev/null 2>&1 || true
else
    telegram-send "Z4c validation of ${BASE} FAILED (exit ${RC})" >/dev/null 2>&1 || true
fi
exit "$RC"
