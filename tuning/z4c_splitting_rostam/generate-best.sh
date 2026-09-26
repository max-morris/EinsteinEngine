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

# Read the checkpoint, pin the best-scoring parameters, and run the recipe once
# locally to bake them into generated code under ./Cottonmouth/. No build, no
# submit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
source "${SCRIPT_DIR}/config.sh"

cd "${SCRIPT_DIR}"

#  Which result to bake. Defaults to the best known configuration
#  (tuner_fixed_order.py + split_tuning_fixed_order.jsonl, 4.933 s at 128^3).
#  Override for a different search, e.g.
#    TUNER_FILE=tuner.py CHECKPOINT_FILE=split_tuning_checkpt.jsonl ./generate-best.sh
#  The tuner file must be the one that produced the checkpoint: it supplies the
#  ordering the trials were measured under, so pairing a checkpoint with the
#  wrong tuner silently bakes something other than what was measured.
TUNER_FILE="${TUNER_FILE:-tuner_fixed_order.py}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-split_tuning_fixed_order.jsonl}"

if [ ! -s "${SCRIPT_DIR}/${CHECKPOINT_FILE}" ]; then
    echo "No checkpoint at ${SCRIPT_DIR}/${CHECKPOINT_FILE}" >&2
    echo "Committed checkpoints in this directory:" >&2
    ls -1 "${SCRIPT_DIR}"/*.jsonl 2>/dev/null | sed 's|.*/|  |' >&2 || echo "  (none)" >&2
    exit 1
fi

echo "baking best of ${CHECKPOINT_FILE} using ${TUNER_FILE}"
PYTHONPATH="${REPO_ROOT}" "${PYTHON}" -m EinsteinEngine.tuning.generate_best \
    "${REPO_ROOT}/recipes/Cottonmouth/Z4c.py" \
    "${SCRIPT_DIR}/${TUNER_FILE}" \
    --checkpoint-file "${SCRIPT_DIR}/${CHECKPOINT_FILE}" \
    "$@"
