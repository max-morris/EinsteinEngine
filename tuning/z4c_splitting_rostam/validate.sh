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

# Re-measure the top configurations from the checkpoint several times each.
# Same wiring as run-tuning.sh; results go to validation.jsonl, and the tuning
# checkpoint is left untouched.
#
#   ./validate.sh                          # baseline + top 2, 3 reps each
#   ./validate.sh --top 3 --reps 5         # more thorough

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
source "${SCRIPT_DIR}/config.sh"

cd "${SCRIPT_DIR}"

# Which tuner/checkpoint to validate against. Defaults to the control search;
# override for a sweep checkpoint, e.g.
#   EE_INFLIGHT_K=0.25 \
#   TUNER_FILE=tuner_inflight.py \
#   CHECKPOINT_FILE=split_tuning_k0p25.jsonl \
#   ./validate.sh --no-baseline --top 1 --reps 3
# The tuner file must be the one that produced the checkpoint: it supplies the
# ordering_fn the trials were measured with, so validating a sweep checkpoint
# against tuner.py would silently re-measure under the wrong ordering.
TUNER_FILE="${TUNER_FILE:-${SCRIPT_DIR}/tuner.py}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${SCRIPT_DIR}/split_tuning_checkpt.jsonl}"
RESULTS_FILE="${RESULTS_FILE:-${SCRIPT_DIR}/validation.jsonl}"

PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${SCRIPT_DIR}/validate.py" \
    "${REPO_ROOT}/recipes/Cottonmouth/Z4c.py" \
    "${TUNER_FILE}" \
    --local-path "${SCRIPT_DIR}/Cottonmouth/" \
    --remote-host localhost \
    --remote-path "${STAGE_DIR}/" \
    --remote-cactus-path "${CACTUS_DIR}/" \
    --remote-command "${SCRIPT_DIR}/build-and-submit.sh" \
    --remote-timing-command "${SCRIPT_DIR}/timings.sh" \
    --checkpoint-file "${CHECKPOINT_FILE}" \
    --results-file "${RESULTS_FILE}" \
    --baseline \
    "$@"
