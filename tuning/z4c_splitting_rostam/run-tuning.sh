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

# Tune the Z4c RHS splitting on rostam. Run ./setup-arrangement.sh once first.
#
# Extra flags pass through, e.g.
#   ./run-tuning.sh --iterations 50
#   ./run-tuning.sh --warmup-iterations 0 --iterations 5   # short smoke test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
source "${SCRIPT_DIR}/config.sh"

# The recipe writes its output into the working dir, so run from here to keep
# tuner generations out of the repo's normal generated/ tree. That means
# PYTHONPATH has to carry the repo root.
cd "${SCRIPT_DIR}"

PYTHONPATH="${REPO_ROOT}" "${PYTHON}" -m EinsteinEngine.tuning.remote_tuner \
    "${REPO_ROOT}/recipes/Cottonmouth/Z4c.py" \
    "${SCRIPT_DIR}/tuner.py" \
    --local-path "${SCRIPT_DIR}/Cottonmouth/" \
    --remote-host localhost \
    --remote-path "${STAGE_DIR}/" \
    --remote-cactus-path "${CACTUS_DIR}/" \
    --remote-command "${SCRIPT_DIR}/build-and-submit.sh" \
    --remote-timing-command "${SCRIPT_DIR}/timings.sh" \
    --checkpoint-file "${SCRIPT_DIR}/split_tuning_checkpt.jsonl" \
    "$@"
