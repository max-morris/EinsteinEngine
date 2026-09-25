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

# Undo setup-arrangement.sh: point the arrangement symlink back at the
# Cottonmouth git checkout. The staging dir is left alone.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

if [ ! -L "${ARRANGEMENT_LINK}" ]; then
    echo "'${ARRANGEMENT_LINK}' is not a symlink; nothing to restore." >&2
    exit 1
fi

echo "Current target: $(readlink "${ARRANGEMENT_LINK}")"
ln -sfn "${ARRANGEMENT_LINK_ORIG}" "${ARRANGEMENT_LINK}"
echo "New target:     $(readlink "${ARRANGEMENT_LINK}")"
echo
echo "Rebuild with './simfactory/bin/sim build ${CONFIG}' to get the checkout's thorn back."
