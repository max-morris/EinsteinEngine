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

# One-time setup. Run this once before the first ./run-tuning.sh, and run
# ./restore-arrangement.sh when you are done tuning.
#
# All it does is point arrangements/Cottonmouth/CottonmouthZ4c4m at the
# tuner's staging dir instead of at the git checkout, so that each trial's
# regenerated thorn is what gets built. The config's ThornList still says
# "Cottonmouth/CottonmouthZ4c4m" and needs no edit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

if [ ! -d "${CACTUS_DIR}" ]; then
    echo "Cactus dir '${CACTUS_DIR}' does not exist; edit config.sh" >&2
    exit 1
fi

if [ ! -L "${ARRANGEMENT_LINK}" ]; then
    echo "'${ARRANGEMENT_LINK}' is not a symlink." >&2
    echo "Refusing to touch it -- this script only ever retargets a symlink." >&2
    exit 2
fi

CURRENT=$(readlink "${ARRANGEMENT_LINK}")
echo "Current target: ${CURRENT}"

mkdir -p "${STAGE_DIR}/CottonmouthZ4c4m"

# Seed the staging dir from the checkout so the tree is buildable before the
# first trial has generated anything into it.
if [ -z "$(ls -A "${STAGE_DIR}/CottonmouthZ4c4m")" ]; then
    echo "Seeding ${STAGE_DIR}/CottonmouthZ4c4m from the checkout..."
    rsync -a "${CACTUS_DIR}/repos/Cottonmouth/CottonmouthZ4c4m/" \
             "${STAGE_DIR}/CottonmouthZ4c4m/"
fi

ln -sfn "${ARRANGEMENT_LINK_TUNING}" "${ARRANGEMENT_LINK}"
echo "New target:     $(readlink "${ARRANGEMENT_LINK}")"

# The tuner's parfile has to be somewhere simfactory can read it. It goes one
# level above STAGE_DIR, not inside it: STAGE_DIR is the rsync --delete target
# and anything there that the recipe did not generate is wiped each trial.
mkdir -p "${PARFILE_DIR}"
cp "${SCRIPT_DIR}/${PARFILE_NAME}.par" "${PARFILE}"
echo "Installed benchmark parfile at ${PARFILE}"

echo
echo "Setup complete. Run ./restore-arrangement.sh when finished tuning."
