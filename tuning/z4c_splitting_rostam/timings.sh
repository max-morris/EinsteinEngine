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

# This is the --remote-timing-command. It must print exactly one number, which
# the tuner minimizes.
#
# The number is the total seconds spent in ODESolvers::Solve::rhs, averaged
# over MPI ranks. That timer is used rather than a per-function one because
# hard-splitting renames and multiplies the generated z4c_rhs schedule
# entries, so no per-function timer name is stable across trials. The
# iteration count is fixed by the parfile, so totals are comparable.
#
# Format of AllTimersReadable.txt (TimerReport/src/Output.cc:434):
#   <iteration> <time>\t<timer#>\t<avg> <min> <max>\t<timer name>
# so the average is field 4.
#
# Exiting nonzero here makes remote_feedback.py raise, which the tuner scores
# as -inf and drops -- the correct outcome for a run that crashed or produced
# no timers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

REPORT="${SIM_BASEDIR}/${SIM_NAME}/output-0000/${PARFILE_NAME}/AllTimersReadable.txt"

if [ ! -r "${REPORT}" ]; then
    echo "No timer report at '${REPORT}' -- the run probably failed." >&2
    exit 1
fi

VALUE=$(grep -F 'ODESolvers::Solve::rhs' "${REPORT}" | tail -1 | awk '{print $4}')

if [ -z "${VALUE}" ]; then
    echo "No ODESolvers::Solve::rhs row in '${REPORT}'." >&2
    exit 2
fi

echo "${VALUE}"
