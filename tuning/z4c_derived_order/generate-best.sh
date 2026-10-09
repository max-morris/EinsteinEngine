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

# Generate the Z4c code for the best entry of the checkpoint, with the add_eqn groups in the derived order of tuner.py.
# To reproduce the tuned build of the feature/eqn-order-instrumentation branch, convert that branch's
# tuning/z4c_splitting_rostam/split_tuning_fixed_order.jsonl first:
#     python scripts/shift_checkpoint_indices.py <that file> tuning/z4c_derived_order/split_tuning_checkpt.jsonl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"

cd "${SCRIPT_DIR}"

PYTHONPATH="${REPO_ROOT}" python -m EinsteinEngine.tuning.generate_best \
    "${REPO_ROOT}/recipes/Cottonmouth/Z4c.py" \
    "${SCRIPT_DIR}/tuner.py" \
    --checkpoint-file "${SCRIPT_DIR}/split_tuning_checkpt.jsonl" \
    "$@"
